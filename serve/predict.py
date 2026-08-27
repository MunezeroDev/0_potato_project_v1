
import json
import sys
from pathlib import Path

import torch
from PIL import Image, ImageOps

PROJECT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT / "scripts"))

import data as D                       # noqa: E402
from model import load_checkpoint      # noqa: E402

FINAL_DIR = PROJECT / "results" / "final"

PRETTY = {"Potato___Early_blight": "Early Blight",
          "Potato___Late_blight":  "Late Blight",
          "Potato___healthy":      "Healthy"}

DEFAULT_THRESHOLD = 0.90               # fallback only; checkpoint value wins

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def _find_checkpoint() -> Path:
    """Locate the deployable model. Explicit name first, then any .pt in final/."""
    named = FINAL_DIR / "potato_mobilenetv3_final.pt"
    if named.exists():
        return named
    candidates = sorted(FINAL_DIR.glob("*.pt"))
    if not candidates:
        raise FileNotFoundError(
            f"no checkpoint in {FINAL_DIR}. Run notebooks/09_final.ipynb first."
        )
    return candidates[0]


CKPT = _find_checkpoint()

_model, _meta = load_checkpoint(CKPT, device=DEVICE)

# Serving config, taken from the checkpoint 
AUG_MODE = _meta["aug_mode"]
_, _tf = D.build_transforms(AUG_MODE)          # eval pipeline only

CLASSES = [c for c, _ in sorted(_meta["class_to_idx"].items(), key=lambda kv: kv[1])]

CONF_THRESHOLD = float(
    _meta.get("metrics", {}).get("confidence_threshold", DEFAULT_THRESHOLD)
)

# classes.json is a published artefact; it must agree with the checkpoint.
_published = json.loads((PROJECT / "results" / "classes.json").read_text())["classes"]
if _published != CLASSES:
    raise ValueError(
        f"classes.json {_published} disagrees with checkpoint {CLASSES}"
    )

print(f"[predict] loaded {CKPT.name} | aug_mode={AUG_MODE} | "
      f"threshold={CONF_THRESHOLD} | device={DEVICE}")


def predict_pil(img: Image.Image) -> dict:
    img = ImageOps.exif_transpose(img).convert("RGB")
    x = _tf(img).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        p = torch.softmax(_model(x), dim=1)[0].cpu()
    i = int(p.argmax())
    return {
        "label":      PRETTY.get(CLASSES[i], CLASSES[i]),
        "raw_label":  CLASSES[i],
        "index":      i,
        "confidence": float(p[i]),
        "probs":      {PRETTY.get(c, c): float(v) for c, v in zip(CLASSES, p)},
        "abstain":    float(p[i]) < CONF_THRESHOLD,
    }
