"""Project constants. Single source of truth — do not duplicate these values."""
from pathlib import Path

SEED = 42

# ── Paths ────────────────────────────────────────────────────────────────
# PROJECT   = Path("/content/drive/MyDrive/0_potato_project_v1")
import os
_env = os.environ.get("POTATO_ROOT")
PROJECT = Path(_env) if _env else Path(__file__).resolve().parent.parent

SOURCE    = PROJECT / "data" / "potato_raw" / "raw"
RESULTS   = PROJECT / "results"
AUDIT_DIR = RESULTS / "audit"
MANIFEST  = AUDIT_DIR / "manifest.csv"
# DATA_ROOT = Path("/content/potato")          # fast local working copy
DATA_ROOT = Path(os.environ.get("POTATO_DATA", "/content/potato"))

# ── Classes ──────────────────────────────────────────────────────────────
CLASS_TO_IDX = {'Potato___Early_blight': 0, 'Potato___Late_blight': 1, 'Potato___healthy': 2}
IDX_TO_CLASS = {v: k for k, v in CLASS_TO_IDX.items()}
NUM_CLASSES  = 3

# ── Audit settings (02) ──────────────────────────────────────────────────
PHASH_T   = 3               
BORDER_PX = 4               # conservative band for background probe
N_FOLDS   = 5

# ── Audit results (02) — reference values, not inputs ─────────────────────
BG_PROBE_F1       = 0.811   # background-only macro-F1, 4px band
BG_PROBE_BASELINE = 0.212
BG_PROBE_GREY     = 0.586   # greyscale only — confound is chromatic

# ── Data pipeline (03) ───────────────────────────────────────────────────
INNER_SPLITS = 7            # StratifiedGroupKFold within the 4 training folds → ~14% inner val
NUM_WORKERS  = 2
AUG_MODE     = "baseline"   # 'baseline' | 'aggressive' | 'greyworld' — ablation switch
BALANCE      = "loss"       # 'loss' | 'sampler' | 'none' — never two at once
CLASSES_JSON = RESULTS / "classes.json"

# ── Preprocessing ────────────────────────────────────────────────────────
IMG_SIZE       = 224        # MobileNetV3 input; source is native 256
IMAGENET_MEAN  = (0.485, 0.456, 0.406)
IMAGENET_STD   = (0.229, 0.224, 0.225)

# ── Training (04) ────────────────────────────────────────────────────────
BATCH_SIZE   = 32
LR_HEAD      = 1e-3         # stage 1: frozen backbone
LR_BACKBONE  = 1e-5         # stage 2: unfrozen, low LR
EPOCHS_HEAD  = 10
EPOCHS_TUNE  = 15

# ── Success thresholds ───────────────────────────────────────────────────
TARGET_MACRO_F1       = 0.95
TARGET_HEALTHY_RECALL = 0.90   # ~30 held out per fold → 3.3-pt granularity

EARLY_STOP_MONITOR  = "val_loss"   # not macro-F1: 17 healthy in inner val → ~6-pt steps
EARLY_STOP_PATIENCE = 4
