"""Analysis of finished cross-validation runs: pooled metrics and Grad-CAM.

Reads artifacts written by train.py and never trains anything, so it runs in
seconds on CPU against any completed run directory.

    python eval.py results/runs/20260821_2337_baseline
"""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")                      # no display in a script context
import matplotlib.cm as cm_maps
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image
from sklearn.metrics import classification_report, confusion_matrix, f1_score

import config as C
import data as D
import model as M

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NAMES = [C.IDX_TO_CLASS[i].replace("Potato___", "") for i in range(C.NUM_CLASSES)]
MEAN = np.array(C.IMAGENET_MEAN).reshape(3, 1, 1)
STD = np.array(C.IMAGENET_STD).reshape(3, 1, 1)


# ── Pooling ───────────────────────────────────────────────────────────────
def pool_oof(run_dir, n_folds=C.N_FOLDS, verify=True):
    """Stack per-fold OOF predictions into one row-per-image frame.

    Every image appears exactly once, predicted by a model that never trained on
    it. A duplicate path means folds overlapped and every metric below is
    inflated, so it is checked rather than assumed.
    """
    run_dir = Path(run_dir)
    oof = pd.concat(
        [pd.read_csv(run_dir / f"fold_{k}" / "oof_preds.csv") for k in range(n_folds)],
        ignore_index=True)

    if verify:
        dups = len(oof) - oof.path.nunique()
        assert dups == 0, f"{dups} duplicate paths — folds overlap"
        probs = oof[[f"p{i}" for i in range(C.NUM_CLASSES)]].values
        assert np.allclose(probs.sum(1), 1), "probabilities do not sum to 1"

    return oof


# ── Metrics ───────────────────────────────────────────────────────────────
def report(oof, summary=None, verbose=True):
    """Confusion matrix, per-class metrics, error breakdown. Returns a dict."""
    y_true, y_pred = oof.y_true.values, oof.y_pred.values
    cm = confusion_matrix(y_true, y_pred, labels=range(C.NUM_CLASSES))
    errs = oof[oof.y_pred != oof.y_true]

    # in an advisory context these two are not symmetric: a missed disease
    # spreads, a false alarm costs one wasted inspection
    missed = int(((errs.y_true != 2) & (errs.y_pred == 2)).sum())
    false_alarm = int(((errs.y_true == 2) & (errs.y_pred != 2)).sum())

    out = {
        "macro_f1": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "accuracy": float((y_pred == y_true).mean()),
        "n_errors": len(errs),
        "disease_missed": missed,
        "false_alarms": false_alarm,
        "confusion": cm,
        "conf_correct": float(oof[oof.y_pred == oof.y_true].confidence.mean()),
        "conf_error": float(errs.confidence.mean()) if len(errs) else float("nan"),
    }

    if verbose:
        print("CONFUSION MATRIX   (rows = true, cols = predicted)\n")
        print(f"{'':>14}" + "".join(f"{n:>14}" for n in NAMES))
        for i, n in enumerate(NAMES):
            print(f"{n:>14}" + "".join(
                f"{cm[i, j]:>14}" if i != j else f"{str(cm[i, j]) + ' ✓':>14}"
                for j in range(C.NUM_CLASSES)))

        print("\n\nPER-CLASS METRICS\n")
        print(classification_report(y_true, y_pred, target_names=NAMES,
                                    digits=4, zero_division=0))
        print(f"pooled macro-F1 : {out['macro_f1']:.4f}")

        if summary is not None:
            s = pd.read_csv(summary) if not isinstance(summary, pd.DataFrame) else summary
            print(f"per-fold mean   : {s.test_macro_f1.mean():.4f} "
                  f"± {s.test_macro_f1.std():.4f}")

        print("\nERROR BREAKDOWN")
        print(f"total errors    : {len(errs)} / {len(oof)}")
        for (t, p), n in (errs.groupby(["y_true", "y_pred"]).size()
                          .sort_values(ascending=False).items()):
            print(f"  {NAMES[t]:>13} -> {NAMES[p]:<13} {n:>3}")
        print(f"\ndisease missed (called healthy)   : {missed}")
        print(f"false alarms  (healthy called sick): {false_alarm}")
        print(f"\nmean confidence on correct : {out['conf_correct']:.3f}")
        print(f"mean confidence on errors  : {out['conf_error']:.3f}")

    return out


# ── Grad-CAM ──────────────────────────────────────────────────────────────
class GradCAM:
    """Class-discriminative heatmaps from the last spatial layer.

    Weights each feature map by how strongly the class score responds to it,
    sums them, keeps positive evidence only, and upsamples to image size.

    Shows WHERE the model drew evidence, never WHAT feature it used -- a model
    keying on a whole-image colour cast produces on-leaf maps indistinguishable
    from one reading lesions. Spatial shortcuts only.
    """

    def __init__(self, model, layer):
        self.model, self.acts, self.grads = model, None, None
        self.h = [
            layer.register_forward_hook(
                lambda m, i, o: setattr(self, "acts", o.detach())),
            layer.register_full_backward_hook(
                lambda m, gi, go: setattr(self, "grads", go[0].detach())),
        ]

    def __call__(self, x, class_idx=None):
        """x: (1,3,H,W) normalised. Returns (cam HxW in [0,1], probs, idx)."""
        self.model.zero_grad(set_to_none=True)
        logits = self.model(x)                       # fp32: AMP off deliberately
        probs = torch.softmax(logits.float(), 1)[0].detach().cpu().numpy()
        idx = int(logits.argmax(1)) if class_idx is None else int(class_idx)

        logits[0, idx].backward()

        w = self.grads.mean(dim=(2, 3), keepdim=True)          # channel importance
        cam = F.relu((w * self.acts).sum(1, keepdim=True))     # positive evidence
        cam = F.interpolate(cam, size=x.shape[-2:], mode="bilinear",
                            align_corners=False)[0, 0]

        cam = cam - cam.min()
        cam = cam / cam.max() if cam.max() > 0 else cam
        return cam.cpu().numpy(), probs, idx

    def close(self):
        for h in self.h:
            h.remove()


def load_fold(run_dir, fold=0):
    """Returns (net, meta, GradCAM, transform, oof) for one fold's checkpoint."""
    run_dir = Path(run_dir)
    net, meta = M.load_checkpoint(run_dir / f"fold_{fold}" / "best.pt", device=DEVICE)
    _, tf_eval = D.build_transforms(meta["aug_mode"])   # from checkpoint, not config
    oof = pd.read_csv(run_dir / f"fold_{fold}" / "oof_preds.csv")
    return net, meta, GradCAM(net, net.features[-1]), tf_eval, oof


def load_tensor(rel_path, tf_eval):
    with Image.open(C.DATA_ROOT / rel_path) as im:
        return tf_eval(im.convert("RGB")).unsqueeze(0).to(DEVICE)


def to_image(x):
    """Undo ImageNet normalisation -> displayable HxWx3 in [0,1]."""
    a = x[0].cpu().numpy() * STD + MEAN
    return np.clip(a, 0, 1).transpose(1, 2, 0)


def overlay(img, cam, alpha=0.45):
    heat = cm_maps.jet(cam)[..., :3]
    return np.clip((1 - alpha) * img + alpha * heat, 0, 1)


def plot_correct(engine, tf_eval, oof, fold=0, per_class=3, save=None):
    """Correct predictions, per_class rows deep. Background heat is the tell."""
    correct = oof[oof.y_pred == oof.y_true]
    picks = [correct[correct.y_true == c].head(per_class) for c in range(C.NUM_CLASSES)]

    fig, axes = plt.subplots(C.NUM_CLASSES, per_class * 2,
                             figsize=(3 * per_class * 2, 3 * C.NUM_CLASSES))
    axes = np.atleast_2d(axes)

    for r, sub in enumerate(picks):
        for i, (_, row) in enumerate(sub.iterrows()):
            x = load_tensor(row.path, tf_eval)
            cam, probs, idx = engine(x)
            img = to_image(x)

            axes[r, i * 2].imshow(img)
            axes[r, i * 2].set_title(NAMES[row.y_true], fontsize=9)
            axes[r, i * 2 + 1].imshow(overlay(img, cam))
            axes[r, i * 2 + 1].set_title(
                f"p={probs[idx]:.3f}  hot={(cam > 0.5).mean():.0%}", fontsize=9)
            for c in (i * 2, i * 2 + 1):
                axes[r, c].axis("off")

    fig.suptitle(f"Grad-CAM — correct predictions, fold {fold} held-out", fontsize=13)
    fig.tight_layout()
    if save:
        fig.savefig(save, dpi=110, bbox_inches="tight")
    return fig


def plot_errors(engine, tf_eval, oof, fold=0, save=None):
    """Every error, with maps for both the predicted and the true class."""
    errs = oof[oof.y_pred != oof.y_true].reset_index(drop=True)
    if not len(errs):
        return None

    fig, axes = plt.subplots(len(errs), 3, figsize=(9, 3 * len(errs)))
    axes = axes.reshape(len(errs), 3)

    for r, (_, row) in enumerate(errs.iterrows()):
        x = load_tensor(row.path, tf_eval)
        cam_p, probs, _ = engine(x, class_idx=row.y_pred)     # evidence for the answer
        cam_t, _, _ = engine(x, class_idx=row.y_true)         # evidence for the truth
        img = to_image(x)

        axes[r, 0].imshow(img)
        axes[r, 0].set_title(f"true {NAMES[row.y_true]}", fontsize=9)
        axes[r, 1].imshow(overlay(img, cam_p))
        axes[r, 1].set_title(f"said {NAMES[row.y_pred]}  p={probs[row.y_pred]:.2f}",
                             fontsize=9)
        axes[r, 2].imshow(overlay(img, cam_t))
        axes[r, 2].set_title(f"for {NAMES[row.y_true]}  p={probs[row.y_true]:.2f}",
                             fontsize=9)
        for c in range(3):
            axes[r, c].axis("off")

    fig.suptitle(f"Grad-CAM — {len(errs)} errors, fold {fold} held-out", fontsize=13)
    fig.tight_layout()
    if save:
        fig.savefig(save, dpi=110, bbox_inches="tight")
    return fig


# ── Entry point ───────────────────────────────────────────────────────────
def analyse(run_dir, fold=0, figures=True, verbose=True):
    """Pool, report, and optionally write Grad-CAM figures into the run folder."""
    run_dir = Path(run_dir)
    oof = pool_oof(run_dir)
    summary = run_dir / "summary.csv"
    out = report(oof, summary=summary if summary.exists() else None, verbose=verbose)

    if figures:
        D.setup_data(verbose=verbose)          # Grad-CAM opens real image files
        net, meta, engine, tf_eval, fold_oof = load_fold(run_dir, fold)
        fig_dir = run_dir / "figures"
        fig_dir.mkdir(exist_ok=True)
        plot_correct(engine, tf_eval, fold_oof, fold,
                     save=fig_dir / f"gradcam_correct_fold{fold}.png")
        plot_errors(engine, tf_eval, fold_oof, fold,
                    save=fig_dir / f"gradcam_errors_fold{fold}.png")
        engine.close()
        plt.close("all")
        if verbose:
            print(f"\nfigures -> {fig_dir}")

    return oof, out


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("run_dir", type=str, help="path to a completed run directory")
    p.add_argument("--fold", type=int, default=0, help="fold to visualise")
    p.add_argument("--no-figures", action="store_true", help="metrics only")
    a = p.parse_args()
    analyse(a.run_dir, fold=a.fold, figures=not a.no_figures)


if __name__ == "__main__":
    main()
