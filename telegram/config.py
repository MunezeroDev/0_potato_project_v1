"""Configuration for the Telegram bot.

Everything here reads from environment variables, which run.py loads from
telegram/.env. Nothing in this file talks to the network.
"""
from __future__ import annotations

import os
import re


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


# ── Telegram ─────────────────────────────────────────────────────────────
# One secret, from @BotFather. Looks like  123456789:AAF-xxxxxxxxxxxxxxxxx
BOT_TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN", "").strip()

API_BASE = f"https://api.telegram.org/bot{BOT_TOKEN}"
FILE_BASE = f"https://api.telegram.org/file/bot{BOT_TOKEN}"

# Long polling: how long Telegram holds the connection open with no news.
# This is not a delay - a message arriving at second 3 returns at second 3.
POLL_TIMEOUT = _int("POLL_TIMEOUT", 25)

# On startup, throw away anything sent while the bot was off. Without this a
# bot that was down overnight wakes up and answers yesterday's photos.
SKIP_BACKLOG = _bool("SKIP_BACKLOG", True)

# How many photos may be analysed at once. A CPU laptop should not run ten.
WORKERS = _int("WORKERS", 2)


# ── Internal model API (serve/app.py) ────────────────────────────────────
PREDICT_URL = os.environ.get("PREDICT_URL", "http://127.0.0.1:5000/predict")
HEALTH_URL = os.environ.get("HEALTH_URL", "http://127.0.0.1:5000/health")
PREDICT_TIMEOUT = _float("PREDICT_TIMEOUT", 120.0)


# ── Conversation pacing ──────────────────────────────────────────────────
MESSAGE_GAP_SECONDS = _float("MESSAGE_GAP_SECONDS", 10.0)

# Shown in the "no confident call" message. Should match the model's own
# threshold, which /health reports.
LOW_CONFIDENCE_HINT = _float("LOW_CONFIDENCE_HINT", 0.90)


# ── Image handling ───────────────────────────────────────────────────────
MIN_SHORT_EDGE = _int("MIN_SHORT_EDGE", 224)
MAX_LONG_EDGE = _int("MAX_LONG_EDGE", 1600)

# Telegram's Bot API refuses to serve downloads above 20 MB, so there is no
# point accepting more than that.
MAX_MEDIA_BYTES = _int("MAX_MEDIA_BYTES", 20 * 1024 * 1024)
MEDIA_TIMEOUT = _float("MEDIA_TIMEOUT", 30.0)


# ── Misc ─────────────────────────────────────────────────────────────────
DEBUG = _bool("BOT_DEBUG", False)

_TOKEN_SHAPE = re.compile(r"^\d{5,}:[A-Za-z0-9_-]{30,}$")


class ConfigError(RuntimeError):
    pass


def check() -> list[str]:
    """Return a list of human-readable problems. Empty list means good to go."""
    problems = []
    if not BOT_TOKEN:
        problems.append(
            "TELEGRAM_BOT_TOKEN is not set. Message @BotFather on Telegram, "
            "send /newbot, and paste the token it gives you into telegram/.env"
        )
    elif not _TOKEN_SHAPE.match(BOT_TOKEN):
        problems.append(
            "TELEGRAM_BOT_TOKEN does not look like a bot token. It should be "
            "digits, a colon, then a long random string - for example "
            "123456789:AAF-abcdefghijklmnopqrstuvwxyz1234567"
        )
    if MESSAGE_GAP_SECONDS < 0:
        problems.append("MESSAGE_GAP_SECONDS cannot be negative.")
    return problems


def redacted_token() -> str:
    """Safe to print in a startup banner or a log."""
    if not BOT_TOKEN:
        return "(not set)"
    head, _, tail = BOT_TOKEN.partition(":")
    return f"{head}:{'*' * 6}{tail[-4:] if len(tail) > 4 else ''}"
