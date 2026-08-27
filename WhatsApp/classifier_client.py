"""Thin HTTP client for the internal model API (serve/app.py).
"""
from __future__ import annotations

import logging

import requests

import config

log = logging.getLogger("whatsapp.classifier")


class ClassifierError(RuntimeError):
    """Raised with a message safe to show a user."""


def health() -> dict | None:
    """Return the service's /health payload, or None if it isn't reachable."""
    try:
        r = requests.get(config.HEALTH_URL, timeout=5)
        r.raise_for_status()
        return r.json()
    except Exception:
        return None


def predict(image_bytes: bytes, filename: str = "upload.jpg") -> dict:
    files = {"image": (filename, image_bytes, "image/jpeg")}
    try:
        r = requests.post(config.PREDICT_URL, files=files,
                          timeout=config.PREDICT_TIMEOUT)
    except requests.Timeout as exc:
        raise ClassifierError(
            "The model took too long to answer. Please try again in a moment."
        ) from exc
    except requests.ConnectionError as exc:
        raise ClassifierError(
            "The model service isn't running. Start it with `python serve/app.py`."
        ) from exc
    except requests.RequestException as exc:
        raise ClassifierError(
            f"Could not reach the model service ({type(exc).__name__})."
        ) from exc

    if r.status_code >= 400:
        detail = ""
        try:
            detail = r.json().get("error", "")
        except Exception:
            detail = r.text[:200]
        raise ClassifierError(f"The model rejected that image: {detail or r.status_code}")

    try:
        out = r.json()
    except ValueError as exc:
        raise ClassifierError("The model service returned something unreadable.") from exc

    for key in ("label", "confidence", "probs"):
        if key not in out:
            raise ClassifierError(f"The model reply is missing '{key}'.")
    return out
