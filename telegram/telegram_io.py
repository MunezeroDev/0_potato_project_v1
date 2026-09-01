
from __future__ import annotations

import logging

import requests

import config

log = logging.getLogger("telegram.io")


class TelegramError(RuntimeError):
    """The Bot API refused or could not be reached."""


class MediaError(RuntimeError):
    """Raised when the photo could not be retrieved. Safe to show a user."""


# Low-level call 
def _api(method: str, _timeout: float = 20.0, **params) -> dict:
    if not config.BOT_TOKEN:
        raise TelegramError("TELEGRAM_BOT_TOKEN is not set.")

    url = f"{config.API_BASE}/{method}"
    try:
        r = requests.post(url, json=params, timeout=_timeout)
    except requests.Timeout as exc:
        raise TelegramError(f"{method} timed out after {_timeout}s") from exc
    except requests.RequestException as exc:
        raise TelegramError(f"{method} failed ({type(exc).__name__})") from exc

    try:
        payload = r.json()
    except ValueError as exc:
        raise TelegramError(f"{method} returned non-JSON (HTTP {r.status_code})") from exc

    if not payload.get("ok"):
        code = payload.get("error_code")
        desc = payload.get("description", "no description")
        if code == 401:
            raise TelegramError(
                "Telegram rejected the bot token (401). Check TELEGRAM_BOT_TOKEN "
                "in telegram/.env against what @BotFather gave you."
            )
        if code == 409:
            raise TelegramError(
                "Another copy of this bot is already polling (409). Close the "
                "other terminal, or delete the webhook, then start again."
            )
        raise TelegramError(f"{method} failed: {code} {desc}")

    return payload.get("result")


# Identity
def get_me() -> dict:
    return _api("getMe", _timeout=15.0)


def delete_webhook() -> None:
    try:
        _api("deleteWebhook", drop_pending_updates=False)
    except TelegramError:
        log.debug("deleteWebhook failed; continuing", exc_info=True)

# Inbound
def get_updates(offset: int | None, timeout: int | None = None) -> list[dict]:
    t = config.POLL_TIMEOUT if timeout is None else timeout
    params = {"timeout": t, "allowed_updates": ["message"]}
    if offset is not None:
        params["offset"] = offset
    # Read timeout must outlast the long poll itself.
    return _api("getUpdates", _timeout=t + 15.0, **params) or []


def newest_update_id() -> int | None:
    updates = get_updates(offset=-1, timeout=0)
    if not updates:
        return None
    return updates[-1]["update_id"]


#  Inbound: which attachment is the photo 
def photo_file_id(message: dict) -> tuple[str | None, str]:

    photos = message.get("photo")
    if photos:
        return photos[-1].get("file_id"), "image/jpeg"

    doc = message.get("document")
    if doc:
        return doc.get("file_id"), (doc.get("mime_type") or "")

    return None, ""


def unsupported_kind(message: dict) -> str | None:
    for key, human in (
        ("video", "a video"),
        ("video_note", "a video message"),
        ("animation", "a GIF"),
        ("voice", "a voice note"),
        ("audio", "an audio file"),
        ("sticker", "a sticker"),
        ("location", "a location"),
        ("contact", "a contact"),
        ("poll", "a poll"),
    ):
        if message.get(key):
            return human
    return None


#  Inbound: download
def fetch_file(file_id: str) -> tuple[bytes, str]:
    try:
        meta = _api("getFile", _timeout=config.MEDIA_TIMEOUT, file_id=file_id)
    except TelegramError as exc:
        if "too big" in str(exc).lower():
            raise MediaError(
                "Telegram won't let bots download files above 20 MB. "
                "Send the photo the normal way rather than as a file."
            ) from exc
        raise MediaError(f"Telegram wouldn't hand over that photo ({exc}).") from exc

    size = meta.get("file_size")
    if size and int(size) > config.MAX_MEDIA_BYTES:
        raise MediaError("That image is too large to process.")

    file_path = meta.get("file_path")
    if not file_path:
        raise MediaError("Telegram didn't say where that photo lives.")

    url = f"{config.FILE_BASE}/{file_path}"
    try:
        r = requests.get(url, timeout=config.MEDIA_TIMEOUT, stream=True)
        if r.status_code == 404:
            raise MediaError(
                "That photo is no longer available from Telegram. Send it again."
            )
        r.raise_for_status()

        declared = r.headers.get("Content-Length")
        if declared and int(declared) > config.MAX_MEDIA_BYTES:
            r.close()
            raise MediaError("That image is too large to process.")

        buf = bytearray()
        for chunk in r.iter_content(64 * 1024):
            buf.extend(chunk)
            if len(buf) > config.MAX_MEDIA_BYTES:
                r.close()
                raise MediaError("That image is too large to process.")

        content_type = (r.headers.get("Content-Type") or "").split(";")[0].strip()
        r.close()

        if not buf:
            raise MediaError("The download came back empty.")
        return bytes(buf), content_type

    except MediaError:
        raise
    except requests.Timeout as exc:
        raise MediaError("Downloading the photo from Telegram timed out.") from exc
    except requests.RequestException as exc:
        raise MediaError(f"Could not download the photo ({type(exc).__name__}).") from exc


#  Outbound 
def send(chat_id: int | str, text: str) -> int | None:

    try:
        msg = _api(
            "sendMessage",
            chat_id=chat_id,
            text=text,
            parse_mode="HTML",
            disable_web_page_preview=True,
        )
        log.info("sent %s to %s (%d chars)", msg.get("message_id"), chat_id, len(text))
        return msg.get("message_id")
    except TelegramError:
        log.exception("outbound send failed to %s", chat_id)
        return None


def send_typing(chat_id: int | str) -> None:
    try:
        _api("sendChatAction", _timeout=10.0, chat_id=chat_id, action="typing")
    except TelegramError:
        pass
