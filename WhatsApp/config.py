"""Configuration for the WhatsApp bot. 
"""
from __future__ import annotations

import os


def _bool(name: str, default: bool) -> bool:
    v = os.environ.get(name)
    if v is None:
        return default
    return v.strip().lower() in ("1", "true", "yes", "on")


def _int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, default))
    except (TypeError, ValueError):
        return default


def _float(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, default))
    except (TypeError, ValueError):
        return default


#  Twilio 
ACCOUNT_SID = os.environ.get("TWILIO_ACCOUNT_SID", "")
AUTH_TOKEN = os.environ.get("TWILIO_AUTH_TOKEN", "")

# Sandbox number
WHATSAPP_FROM = os.environ.get("TWILIO_WHATSAPP_FROM", "whatsapp:+14155238886")

VALIDATE_SIGNATURE = _bool("TWILIO_VALIDATE_SIGNATURE", True)

PUBLIC_WEBHOOK_URL = os.environ.get("PUBLIC_WEBHOOK_URL", "")


# Internal model API (serve/app.py)
HEALTH_URL = os.environ.get("HEALTH_URL", "http://127.0.0.1:5000/health")
PREDICT_TIMEOUT = _float("PREDICT_TIMEOUT", 120.0)

MESSAGE_GAP_SECONDS = _float("MESSAGE_GAP_SECONDS", 10.0)


LOW_CONFIDENCE_HINT = _float("LOW_CONFIDENCE_HINT", 0.90)


# Image handling 
MIN_SHORT_EDGE = _int("MIN_SHORT_EDGE", 224)

MAX_LONG_EDGE = _int("MAX_LONG_EDGE", 1600)


MAX_MEDIA_BYTES = _int("MAX_MEDIA_BYTES", 16 * 1024 * 1024)
MEDIA_TIMEOUT = _float("MEDIA_TIMEOUT", 30.0)

# Server 
HOST = os.environ.get("BOT_HOST", "0.0.0.0")
PORT = _int("BOT_PORT", 5005)
DEBUG = _bool("BOT_DEBUG", False)


class ConfigError(RuntimeError):
    pass


def check() -> list[str]:
    """Return a list of human-readable problems. Empty list means good to go."""
    problems = []
    if not ACCOUNT_SID.startswith("AC"):
        problems.append(
            "TWILIO_ACCOUNT_SID is missing or malformed (it starts with 'AC')."
        )
    if not AUTH_TOKEN:
        problems.append("TWILIO_AUTH_TOKEN is not set.")
    if not WHATSAPP_FROM.startswith("whatsapp:"):
        problems.append(
            f"TWILIO_WHATSAPP_FROM must start with 'whatsapp:' (got {WHATSAPP_FROM!r})."
        )
    if VALIDATE_SIGNATURE and not PUBLIC_WEBHOOK_URL:
        problems.append(
            "PUBLIC_WEBHOOK_URL must be set when signature validation is on "
            "(it changes every time ngrok restarts)."
        )
    return problems
