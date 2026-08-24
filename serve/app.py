"""Flask server: upload page + /predict endpoint. Logs every submission."""
import csv
import io
import uuid
from datetime import datetime
from pathlib import Path

from flask import Flask, jsonify, render_template, request
from PIL import Image

from predict import AUG_MODE, CKPT, CLASSES, PRETTY, CONF_THRESHOLD, predict_pil

PROJECT = Path(__file__).resolve().parent.parent
LOG_DIR = PROJECT / "results" / "uploads"
LOG_DIR.mkdir(parents=True, exist_ok=True)
LOG_CSV = LOG_DIR / "log.csv"
NAMES = [PRETTY[c] for c in CLASSES]

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 10 * 1024 * 1024        # 10 MB


def log_row(uid, fname, out):
    new = not LOG_CSV.exists()
    with LOG_CSV.open("a", newline="") as f:
        w = csv.writer(f)
        if new:
            w.writerow(["uid", "ts", "filename", "pred", "confidence",
                        "abstain", "ckpt", "aug_mode"] + NAMES)
        w.writerow([uid, datetime.now().isoformat(timespec="seconds"), fname,
                    out["raw_label"], f"{out['confidence']:.6f}", out["abstain"],
                    CKPT.name, AUG_MODE]
                   + [f"{out['probs'][n]:.6f}" for n in NAMES])


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/health")
def health():
    return jsonify({"ok": True, "classes": CLASSES, "aug_mode": AUG_MODE,
                    "checkpoint": CKPT.name, "threshold": CONF_THRESHOLD})


@app.route("/predict", methods=["POST"])
def predict():
    f = request.files.get("image")
    if f is None or f.filename == "":
        return jsonify({"error": "no image supplied"}), 400

    raw = f.read()
    try:
        img = Image.open(io.BytesIO(raw))
        img.load()
    except Exception:
        return jsonify({"error": "unreadable image file"}), 400

    try:
        out = predict_pil(img)
    except Exception as e:
        app.logger.exception("inference failed")
        return jsonify({"error": f"inference failed: {type(e).__name__}"}), 500

    uid = uuid.uuid4().hex[:12]
    try:
        (LOG_DIR / f"{uid}{Path(f.filename).suffix.lower() or '.jpg'}").write_bytes(raw)
        log_row(uid, f.filename, out)
    except Exception:
        app.logger.exception("logging failed")   # never fail a prediction on logging

    out["uid"] = uid
    return jsonify(out)


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, threaded=False)
