import random
import shutil

import numpy as np
import pandas as pd
import torch
from PIL import Image
from sklearn.model_selection import StratifiedGroupKFold
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torchvision import transforms as T

import config as C


# Session bootstrap 
def setup_data(verbose=True):

    if not (C.DATA_ROOT.exists() and any(C.DATA_ROOT.iterdir())):
        if verbose:
            print(f"restoring {C.SOURCE} -> {C.DATA_ROOT}")
        shutil.copytree(C.SOURCE, C.DATA_ROOT)

    have = {d.name.lower(): d for d in C.DATA_ROOT.iterdir() if d.is_dir()}
    for label in C.CLASS_TO_IDX:
        if (C.DATA_ROOT / label).exists():
            continue
        src = have.get(label.split("___")[-1].lower())
        if src is None:
            raise FileNotFoundError(f"no folder on disk matching '{label}'")
        src.rename(C.DATA_ROOT / label)
        if verbose:
            print(f"  {src.name} -> {label}")

    return C.DATA_ROOT


#  Manifest
def load_manifest(verify=True):
    df = pd.read_csv(C.MANIFEST)
    df["y"] = df.label.map(C.CLASS_TO_IDX)

    if df.y.isna().any():
        bad = sorted(df.loc[df.y.isna(), "label"].unique())
        raise ValueError(f"labels absent from config.CLASS_TO_IDX: {bad}")
    df["y"] = df.y.astype("int64")

    df = df[df.keep].reset_index(drop=True)

    if verify:
        missing = sum(1 for p in df.path if not (C.DATA_ROOT / p).exists())
        if missing:
            raise FileNotFoundError(
                f"{missing}/{len(df)} manifest paths missing — run setup_data()")
    return df


#  Splits 
def get_indices(df, fold, seed=C.SEED):
 
    ev = np.flatnonzero(df.fold.values == fold)
    pool = np.flatnonzero(df.fold.values != fold)

    sgkf = StratifiedGroupKFold(n_splits=C.INNER_SPLITS, shuffle=True,
                                random_state=seed)
    tr_rel, va_rel = next(sgkf.split(pool, df.y.values[pool],
                                     groups=df.group_id.values[pool]))
    return pool[tr_rel], pool[va_rel], ev


def check_fold(df, fold, seed=C.SEED):
    tr, va, ev = get_indices(df, fold, seed)
    G, P = df.group_id.values, df.path.values

    for a, b, name in [(tr, va, "train/val"), (tr, ev, "train/eval"),
                       (va, ev, "val/eval")]:
        assert not (set(G[a]) & set(G[b])), f"fold {fold}: group leak in {name}"
        assert not (set(P[a]) & set(P[b])), f"fold {fold}: path leak in {name}"

    assert len(tr) + len(va) + len(ev) == len(df), f"fold {fold}: rows lost"
    assert np.bincount(df.y.values[ev], minlength=C.NUM_CLASSES).min() > 0, \
        f"fold {fold}: empty class in eval"
    return len(tr), len(va), len(ev)


class GreyWorld:
    def __call__(self, x):                      
        m = x.mean(dim=(1, 2), keepdim=True)
        return (x * m.mean() / m.clamp(min=1e-6)).clamp(0, 1)


def build_transforms(mode=None):
    """Return (train, eval) transform pipelines for the given AUG_MODE."""
    mode = mode or C.AUG_MODE
    if mode not in {"baseline", "aggressive", "greyworld"}:
        raise ValueError(f"unknown AUG_MODE: {mode}")

    gw = [GreyWorld()] if mode == "greyworld" else []
    scale = (0.4, 1.0) if mode == "aggressive" else (0.8, 1.0)
    ratio = (0.85, 1.18) if mode == "aggressive" else (0.9, 1.11)

    train = T.Compose([
        T.RandomResizedCrop(C.IMG_SIZE, scale=scale, ratio=ratio),
        T.RandomHorizontalFlip(),
        T.RandomVerticalFlip(),
        T.RandomRotation(20),
        # Mild on colour: lesion hue carries class signal.
        T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1, hue=0.02),
        T.ToTensor(), *gw,
        T.Normalize(C.IMAGENET_MEAN, C.IMAGENET_STD),
    ])
    evaluate = T.Compose([
        T.Resize(C.IMG_SIZE),
        T.CenterCrop(C.IMG_SIZE),
        T.ToTensor(), *gw,
        T.Normalize(C.IMAGENET_MEAN, C.IMAGENET_STD),
    ])
    return train, evaluate


# Dataset 
class PotatoDataset(Dataset):
    def __init__(self, frame, indices, root=None, transform=None):
        self.paths = frame.path.values[indices]
        self.ys = frame.y.values[indices].astype("int64")
        self.root = root or C.DATA_ROOT
        self.tf = transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        with Image.open(self.root / self.paths[i]) as im:
            x = self.tf(im.convert("RGB"))
        return x, int(self.ys[i])


# Imbalance 
def class_weights(ys, n=C.NUM_CLASSES):
    """sklearn 'balanced' convention: n_samples / (n_classes * count)."""
    counts = np.bincount(ys, minlength=n).astype("float64")
    return torch.tensor(counts.sum() / (n * counts), dtype=torch.float32)


def make_sampler(ys, n=C.NUM_CLASSES):
    """Available for ablation. Not used when BALANCE == 'loss'."""
    counts = np.bincount(ys, minlength=n)
    per_item = (1.0 / counts)[ys]
    return WeightedRandomSampler(torch.as_tensor(per_item, dtype=torch.double),
                                 num_samples=len(ys), replacement=True)


# Loaders
def _seed_worker(worker_id):
    s = torch.initial_seed() % 2 ** 32
    np.random.seed(s)
    random.seed(s)


def get_loaders(df, fold, aug_mode=None, batch_size=None, seed=C.SEED,
                smoke=False, verify=True):
    """Return (train_loader, val_loader, eval_loader, class_weights).

    smoke=True subsamples to ~64 train / 32 val rows for an end-to-end dry run.
    """
    if verify:
        check_fold(df, fold, seed)

    tr, va, ev = get_indices(df, fold, seed)
    if smoke:
        rng = np.random.default_rng(seed)
        tr = rng.choice(tr, size=min(64, len(tr)), replace=False)
        va = rng.choice(va, size=min(32, len(va)), replace=False)
        ev = rng.choice(ev, size=min(32, len(ev)), replace=False)

    tf_tr, tf_ev = build_transforms(aug_mode)
    bs = batch_size or C.BATCH_SIZE

    g = torch.Generator()
    g.manual_seed(seed + fold)
    common = dict(num_workers=C.NUM_WORKERS,
                  pin_memory=torch.cuda.is_available(),
                  worker_init_fn=_seed_worker)

    sampler = None
    if C.BALANCE == "sampler":
        sampler = make_sampler(df.y.values[tr])

    train = DataLoader(PotatoDataset(df, tr, transform=tf_tr),
                       batch_size=bs, shuffle=(sampler is None),
                       sampler=sampler, drop_last=True,   # BatchNorm safety
                       generator=g, **common)
    val = DataLoader(PotatoDataset(df, va, transform=tf_ev),
                     batch_size=bs, shuffle=False, **common)
    evaluate = DataLoader(PotatoDataset(df, ev, transform=tf_ev),
                          batch_size=bs, shuffle=False, **common)

    weights = class_weights(df.y.values[tr]) if C.BALANCE == "loss" else None
    return train, val, evaluate, weights
