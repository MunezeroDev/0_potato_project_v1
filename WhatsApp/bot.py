from __future__ import annotations

import logging
import threading
import time
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor

from flask import Flask, Response, jsonify, request

import classifier_client
import config
import imaging
import responder
import twilio_io

logging.basicConfig(
    level=logging.DEBUG if config.DEBUG else logging.INFO,
    format="%(asctime)s %(levelname)-7s %(name)s | %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("whatsapp.bot")

app = Flask(__name__)

# Bounded pool: a CPU laptop should not run ten inferences at once.
POOL = ThreadPoolExecutor(max_workers=2, thread_name_prefix="analyse")

# Twilio retries a webhook if it doesn't like the response.
_seen: "OrderedDict[str, float]" = OrderedDict()
_seen_lock = threading.Lock()
_SEEN_MAX = 500

# Give the greeting a beat to land before message 1/3 follows it.
_MIN_LEAD_SECONDS = 2.0


def _already_handled(message_sid: str) -> bool:
    if not message_sid:
        return False
    with _seen_lock:
        if message_sid in _seen:
            return True
        _seen[message_sid] = time.time()
        while len(_seen) > _SEEN_MAX:
            _seen.popitem(last=False)
    return False


def _twiml(body: str) -> Response:
    """Minimal TwiML. Avoids pulling in the whole MessagingResponse builder."""
    safe = (
        body.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
    )
    xml = f'<?xml version="1.0" encoding="UTF-8"?><Response><Message>{safe}</Message></Response>'
    return Response(xml, mimetype="application/xml")


def _twiml_empty() -> Response:
    return Response(
        '<?xml version="1.0" encoding="UTF-8"?><Response></Response>',
        mimetype="application/xml",
    )


#  The background job 
def analyse_and_reply(sender: str, media_url: str, declared_type: str,
                      profile_name: str, started: float) -> None:
    try:
        try:
            raw, content_type = twilio_io.fetch_media(media_url)
        except twilio_io.MediaError as exc:
            _reply_error(sender, str(exc), started)
            return

        try:
            norm = imaging.normalise(
                raw,
                content_type or declared_type,
                min_short_edge=config.MIN_SHORT_EDGE,
                max_long_edge=config.MAX_LONG_EDGE,
            )
        except imaging.ImageError as exc:
            _reply_error(sender, str(exc), started)
            return

        log.info(
            "analysing for %s | %s %dx%d -> JPEG %dx%d (%.0f KB)",
            sender, norm.source_format, norm.source_width, norm.source_height,
            norm.width, norm.height, len(norm.data) / 1024,
        )

        try:
            result = classifier_client.predict(norm.data, norm.filename)
        except classifier_client.ClassifierError as exc:
            _reply_error(sender, str(exc), started)
            return

        messages = [
            responder.verdict_message(result, profile_name),
            responder.breakdown_message(result, responder.image_quality_note(norm)),
            responder.guidance_message(result),
        ]
        _wait_for_lead(started)
        responder.send_sequence(sender, messages)
        log.info(
            "done for %s | %s @ %.3f (abstain=%s)",
            sender, result.get("label"), result.get("confidence", 0),
            result.get("abstain"),
        )

    except Exception:
        log.exception("unexpected failure while analysing for %s", sender)
        twilio_io.send(
            sender,
            responder.error_message("Something broke on my side while reading it."),
        )


def _wait_for_lead(started: float) -> None:
    elapsed = time.time() - started
    if elapsed < _MIN_LEAD_SECONDS:
        time.sleep(_MIN_LEAD_SECONDS - elapsed)


def _reply_error(sender: str, reason: str, started: float) -> None:
    log.warning("replying with error to %s: %s", sender, reason)
    _wait_for_lead(started)
    twilio_io.send(sender, responder.error_message(reason))


# Routes 
@app.post("/whatsapp")
def whatsapp():
    form = request.form.to_dict()

    if not twilio_io.signature_ok(request.headers.get("X-Twilio-Signature"), form):
        return Response("forbidden", status=403)

    message_sid = form.get("MessageSid", "")
    sender = form.get("From", "")
    profile_name = form.get("ProfileName", "")
    body = (form.get("Body") or "").strip()

    try:
        num_media = int(form.get("NumMedia", "0"))
    except ValueError:
        num_media = 0

    if not sender:
        return Response("bad request", status=400)

    if _already_handled(message_sid):
        log.info("ignoring Twilio retry of %s", message_sid)
        return _twiml_empty()

    log.info(
        "inbound from %s (%s) | media=%d | body=%r",
        sender, profile_name or "no name", num_media, body[:60],
    )

    if num_media == 0:
        return _twiml(responder.help_text(profile_name))

    media_url = form.get("MediaUrl0", "")
    declared_type = form.get("MediaContentType0", "")

    if not media_url:
        return _twiml(
            responder.error_message("Twilio didn't include a link to that attachment.")
        )

    if declared_type and not declared_type.startswith("image/"):
        return _twiml(
            responder.error_message(
                f"I can only read photos, and that arrived as {declared_type}."
            )
        )

    POOL.submit(
        analyse_and_reply, sender, media_url, declared_type, profile_name, time.time()
    )

    extra = ""
    if num_media > 1:
        extra = f"\n\n(You sent {num_media} images - I'm reading the first one.)"
    return _twiml(responder.greeting(profile_name) + extra)


@app.get("/health")
def health():
    """Is the bot up, and can it see the model service?"""
    model = classifier_client.health()
    problems = config.check()
    return jsonify({
        "ok": not problems and model is not None,
        "config_problems": problems,
        "model_service": model or "unreachable",
        "predict_url": config.PREDICT_URL,
        "signature_validation": config.VALIDATE_SIGNATURE,
        "message_gap_seconds": config.MESSAGE_GAP_SECONDS,
        "heic_support": imaging.HEIF_OK,
    })


@app.get("/")
def root():
    return (
        "Potato Leaf Doctor - WhatsApp bridge.\n"
        "Point the Twilio sandbox 'When a message comes in' webhook at "
        "POST /whatsapp on this host.\n"
    ), 200, {"Content-Type": "text/plain"}


if __name__ == "__main__":
    app.run(host=config.HOST, port=config.PORT, debug=config.DEBUG, threaded=True)
