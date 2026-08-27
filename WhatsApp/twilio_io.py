
from __future__ import annotations

import logging

import requests
from twilio.request_validator import RequestValidator
from twilio.rest import Client

import config

log = logging.getLogger("whatsapp.twilio")

_client: Client | None = None


def client() -> Client:
    global _client
    if _client is None:
        _client = Client(config.ACCOUNT_SID, config.AUTH_TOKEN)
    return _client


#  Inbound: signature 
def signature_ok(signature: str | None, form: dict) -> bool:
    """Verify X-Twilio-Signature against the public URL in config."""
    if not config.VALIDATE_SIGNATURE:
        return True
    if not signature:
        log.warning("rejected: no X-Twilio-Signature header")
        return False
    validator = RequestValidator(config.AUTH_TOKEN)
    ok = validator.validate(config.PUBLIC_WEBHOOK_URL, form, signature)
    if not ok:
        log.warning(
            "rejected: signature mismatch. Check PUBLIC_WEBHOOK_URL (%s) matches "
            "the URL in the Twilio console exactly, including /whatsapp.",
            config.PUBLIC_WEBHOOK_URL,
        )
    return ok


# Inbound: media 
class MediaError(RuntimeError):
    """Raised when the photo could not be retrieved."""


def fetch_media(media_url: str) -> tuple[bytes, str]:
    """Download one WhatsApp attachment.

    Returns (bytes, content_type). Raises MediaError with a message that is safe
    to show a user.
    """
    auth = (config.ACCOUNT_SID, config.AUTH_TOKEN)
    try:
        r = requests.get(
            media_url,
            auth=auth,
            timeout=config.MEDIA_TIMEOUT,
            allow_redirects=False,
            stream=True,
        )

        hops = 0
        while r.is_redirect or r.status_code in (301, 302, 303, 307, 308):
            location = r.headers.get("Location")
            if not location or hops >= 5:
                raise MediaError("Twilio media redirect could not be followed.")
            r.close()
            r = requests.get(
                location,
                timeout=config.MEDIA_TIMEOUT,
                allow_redirects=False,
                stream=True,
            )
            hops += 1

        if r.status_code == 404:
            raise MediaError(
                "Twilio says that photo no longer exists (media expires after a while)."
            )
        if r.status_code in (401, 403):
            raise MediaError(
                "Twilio refused the download. Check TWILIO_ACCOUNT_SID / TWILIO_AUTH_TOKEN."
            )
        r.raise_for_status()

        declared = r.headers.get("Content-Length")
        if declared and int(declared) > config.MAX_MEDIA_BYTES:
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
        raise MediaError("Downloading the photo from Twilio timed out.") from exc
    except requests.RequestException as exc:
        raise MediaError(f"Could not download the photo ({type(exc).__name__}).") from exc


#  Outbound 
def send(to: str, body: str) -> str | None:
    """Send one WhatsApp message. Returns the message SID, or None on failure.

    Never raises: a failed follow-up must not take the worker thread down.
    """
    try:
        msg = client().messages.create(
            from_=config.WHATSAPP_FROM, to=to, body=body
        )
        log.info("sent %s to %s (%d chars)", msg.sid, to, len(body))
        return msg.sid
    except Exception:
        log.exception("outbound send failed to %s", to)
        return None
