"""Two-stage transfer learning with stratified group 5-fold cross-validation.

Stage 1 trains the classifier head on a frozen backbone; stage 2 unfreezes the
late blocks at a low learning rate. Each fold writes best.pt, history.csv and
oof_preds.csv to its own directory; pooled across folds, oof_preds.csv gives one
held-out prediction per image in the dataset.

    python train.py --smoke          # end-to-end dry run, ~1 min
    python train.py                  # full 5-fold CV, ~20 min on a T4
"""
import argparse
import copy
import random
import time
from datetime import datetime

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import f1_score, recall_score

import config as C
import data as D
import model as M

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
AMP = DEVICE.type == "cuda"
HEALTHY = C.CLASS_TO_IDX["Potato___healthy"]


def set_seed(seed=C.SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# ── Epoch primitives ──────────────────────────────────────────────────────
def train_one_epoch(model, loader, criterion, optimizer, scaler=None):
    """One pass with weight updates. Returns mean training loss."""
    model.train()                                   # BN guard fires here
    total, n = 0.0, 0

    for xb, yb in loader:
        xb = xb.to(DEVICE, non_blocking=True)
        yb = yb.to(DEVICE, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)

        with torch.autocast("cuda", enabled=AMP):
            loss = criterion(model(xb), yb)

        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        total += loss.item() * yb.size(0)
        n += yb.size(0)

    return total / n


@torch.no_grad()
def evaluate(model, loader, criterion):
    """No-update pass. Returns (metrics, probs, ys).

    Loader order is fixed (shuffle=False), so probs rows align with the loader's
    dataset -- that alignment is what makes the OOF predictions joinable to paths.
    """
    model.eval()
    total, n, P, Y = 0.0, 0, [], []

    for xb, yb in loader:
        xb = xb.to(DEVICE, non_blocking=True)
        yb = yb.to(DEVICE, non_blocking=True)

        with torch.autocast("cuda", enabled=AMP):
            logits = model(xb)
            loss = criterion(logits, yb)

        total += loss.item() * yb.size(0)
        n += yb.size(0)
        P.append(torch.softmax(logits.float(), dim=1).cpu())
        Y.append(yb.cpu())

    probs = torch.cat(P).numpy()
    ys = torch.cat(Y).numpy()
    preds = probs.argmax(1)

    metrics = {
        "loss": total / n,
        "acc": float((preds == ys).mean()),
        "macro_f1": f1_score(ys, preds, average="macro", zero_division=0),
        "healthy_recall": recall_score(ys, preds, labels=[HEALTHY],
                                       average="macro", zero_division=0),
    }
    return metrics, probs, ys


# ── Stage loop ────────────────────────────────────────────────────────────
def run_stage(model, train_loader, val_loader, criterion, optimizer, scaler,
              epochs, stage, patience=C.EARLY_STOP_PATIENCE, fold=None,
              verbose=True):
    """Train until val loss stops improving. Returns (history, best_epoch, best_val).

    The model is left holding the BEST weights seen, not the last ones -- without
    the restore, patience N means keeping a model N epochs past its peak.

    Selection watches val_loss rather than macro-F1: the inner validation slice
    holds ~17 healthy images, so F1 moves in ~6-point steps and cannot separate
    real improvement from noise.
    """
    best_loss, best_state, best_epoch, bad = float("inf"), None, -1, 0
    history = []

    for ep in range(1, epochs + 1):
        t0 = time.time()
        tr_loss = train_one_epoch(model, train_loader, criterion, optimizer, scaler)
        val, _, _ = evaluate(model, val_loader, criterion)

        improved = val["loss"] < best_loss
        if improved:
            best_loss, best_epoch, bad = val["loss"], ep, 0
            best_state = copy.deepcopy(
                {k: v.detach().cpu().clone() for k, v in model.state_dict().items()})
        else:
            bad += 1

        history.append({"fold": fold, "stage": stage, "epoch": ep,
                        "train_loss": tr_loss,
                        **{f"val_{k}": v for k, v in val.items()},
                        "lr": optimizer.param_groups[0]["lr"],
                        "secs": round(time.time() - t0, 1)})

        if verbose:
            print(f"  s{stage} e{ep:>2}  train {tr_loss:.4f}  val {val['loss']:.4f}"
                  f"  F1 {val['macro_f1']:.3f}  healthy-R {val['healthy_recall']:.3f}"
                  f"  {time.time() - t0:.0f}s{'  *' if improved else ''}")

        if bad >= patience:
            if verbose:
                print(f"  early stop: no improvement in {patience} epochs")
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    return history, best_epoch, best_loss


# ── Run directories ───────────────────────────────────────────────────────
def make_run_dir(aug_mode=None, tag=None):
    """results/runs/<timestamp>_<aug_mode>/ -- one per configuration."""
    aug = aug_mode or C.AUG_MODE
    name = tag or f"{datetime.now():%Y%m%d_%H%M}_{aug}"
    d = C.RESULTS / "runs" / name
    d.mkdir(parents=True, exist_ok=True)
    return d


def fold_dir(run_dir, fold):
    d = run_dir / f"fold_{fold}"
    d.mkdir(parents=True, exist_ok=True)
    return d


def fold_done(run_dir, fold):
    """True if all three artifacts exist -- used to resume after a disconnect."""
    d = run_dir / f"fold_{fold}"
    return all((d / f).exists() for f in ("best.pt", "history.csv", "oof_preds.csv"))


# ── Fold ──────────────────────────────────────────────────────────────────
def run_fold(df, fold, run_dir, smoke=False, verbose=True):
    """Stage 1 -> stage 2 -> one held-out evaluation. Writes 3 files, returns a row.

    A fresh model per fold is mandatory: carrying weights across folds would mean
    later folds have already seen their own held-out images.
    """
    d = fold_dir(run_dir, fold)
    set_seed(C.SEED + fold)

    tl, vl, el, w = D.get_loaders(df, fold=fold, smoke=smoke)
    net = M.set_stage(M.build_model(), stage=1).to(DEVICE)
    crit = (nn.CrossEntropyLoss(weight=w.to(DEVICE)) if w is not None
            else nn.CrossEntropyLoss())
    scaler = torch.amp.GradScaler(enabled=AMP)

    if verbose:
        print(f"\n── fold {fold} ──  train {len(tl.dataset)}  "
              f"val {len(vl.dataset)}  eval {len(el.dataset)}")

    # stage 1: head only, frozen backbone
    opt = torch.optim.AdamW(M.param_groups(net, stage=1), weight_decay=1e-4)
    h1, be1, bl1 = run_stage(net, tl, vl, crit, opt, scaler,
                             epochs=1 if smoke else C.EPOCHS_HEAD,
                             stage=1, fold=fold, verbose=verbose)

    # stage 2: the optimizer MUST be rebuilt -- it was constructed over stage 1's
    # trainable params, so newly unfrozen weights would otherwise never update
    M.set_stage(net, stage=2)
    opt = torch.optim.AdamW(M.param_groups(net, stage=2), weight_decay=1e-4)
    h2, be2, bl2 = run_stage(net, tl, vl, crit, opt, scaler,
                             epochs=1 if smoke else C.EPOCHS_TUNE,
                             stage=2, fold=fold, verbose=verbose)

    # held-out fold: first and only look
    test, probs, ys = evaluate(net, el, crit)

    pd.DataFrame(h1 + h2).to_csv(d / "history.csv", index=False)

    # paths come from the loader's own dataset, not a re-derived index -- smoke
    # mode subsamples, and any drift here silently misjoins predictions to files
    paths = el.dataset.paths
    assert len(paths) == len(ys), f"path/label mismatch: {len(paths)} vs {len(ys)}"
    assert (el.dataset.ys == ys).all(), "loader order changed between build and eval"

    pd.DataFrame({
        "path": paths, "y_true": ys,
        **{f"p{i}": probs[:, i] for i in range(C.NUM_CLASSES)},
        "y_pred": probs.argmax(1), "confidence": probs.max(1), "fold": fold,
    }).to_csv(d / "oof_preds.csv", index=False)

    M.save_checkpoint(d / "best.pt", net, fold=fold, stage=2, epoch=be2,
                      metrics=test)

    row = {"fold": fold, **{f"test_{k}": v for k, v in test.items()},
           "best_ep_s1": be1, "best_val_s1": bl1,
           "best_ep_s2": be2, "best_val_s2": bl2,
           "n_train": len(tl.dataset), "n_eval": len(el.dataset)}

    if verbose:
        print(f"  HELD-OUT  F1 {test['macro_f1']:.4f}  "
              f"healthy-R {test['healthy_recall']:.4f}  acc {test['acc']:.4f}")
    return row


# ── Cross-validation ──────────────────────────────────────────────────────
def run_cv(run_dir=None, folds=None, smoke=False, verbose=True):
    """All folds, resumable. summary.csv is rewritten as each fold completes."""
    D.setup_data(verbose=verbose)
    df = D.load_manifest()

    run_dir = run_dir or make_run_dir()
    summary = run_dir / "summary.csv"
    folds = range(C.N_FOLDS) if folds is None else folds

    rows = pd.read_csv(summary).to_dict("records") if summary.exists() else []
    done = {r["fold"] for r in rows}

    t0 = time.time()
    for k in folds:
        if k in done and fold_done(run_dir, k):
            if verbose:
                print(f"fold {k}: already complete — skipping")
            continue
        rows = [r for r in rows if r["fold"] != k]
        rows.append(run_fold(df, k, run_dir, smoke=smoke, verbose=verbose))
        rows.sort(key=lambda r: r["fold"])
        pd.DataFrame(rows).to_csv(summary, index=False)   # checkpoint each fold

    s = pd.DataFrame(rows)
    if verbose:
        print(f"\ntotal: {(time.time() - t0) / 60:.1f} min   folds: {len(s)}")
        print(s[["fold", "test_macro_f1", "test_healthy_recall", "test_acc",
                 "best_ep_s1", "best_ep_s2"]].to_string(index=False))
        for m, target in [("test_macro_f1", C.TARGET_MACRO_F1),
                          ("test_healthy_recall", C.TARGET_HEALTHY_RECALL)]:
            mu, sd = s[m].mean(), s[m].std()
            print(f"\n{m:<20} {mu:.4f} ± {sd:.4f}   target {target}   "
                  f"{'PASS' if mu >= target else 'MISS'}   min fold {s[m].min():.4f}")
    return s, run_dir


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--smoke", action="store_true", help="1 epoch/stage on ~64 images")
    p.add_argument("--folds", type=int, nargs="+", help="subset of folds to run")
    p.add_argument("--tag", type=str, help="run directory name (default: timestamp_augmode)")
    p.add_argument("--aug", type=str, choices=["baseline", "aggressive", "greyworld"],
                   help="override config.AUG_MODE for this run")
    a = p.parse_args()

    if a.aug:
        C.AUG_MODE = a.aug
    run_cv(run_dir=make_run_dir(tag=a.tag), folds=a.folds, smoke=a.smoke)


if __name__ == "__main__":
    main()
