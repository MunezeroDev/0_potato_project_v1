
# The bot itself
from __future__ import annotations

import logging
import time
from concurrent.futures import ThreadPoolExecutor

import classifier_client
import config
import imaging
import responder
import telegram_io

logging.basicConfig(
    level=logging.DEBUG if config.DEBUG else logging.INFO,
    format="%(asctime)s %(levelname)-7s %(name)s | %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("telegram.bot")

# Bounded pool: A CPU laptop should not run ten inferences at once.
POOL = ThreadPoolExecutor(max_workers=config.WORKERS, thread_name_prefix="analyse")

_running = True


#  The background job 
def analyse_and_reply(chat_id, file_id: str, declared_type: str,
                      first_name: str) -> None:
    """Download, normalise, classify, then send the three paced messages."""
    try:
        telegram_io.send_typing(chat_id)

        try:
            raw, content_type = telegram_io.fetch_file(file_id)
        except telegram_io.MediaError as exc:
            telegram_io.send(chat_id, responder.error_message(str(exc)))
            return

        try:
            norm = imaging.normalise(
                raw,
                content_type or declared_type,
                min_short_edge=config.MIN_SHORT_EDGE,
                max_long_edge=config.MAX_LONG_EDGE,
            )
        except imaging.ImageError as exc:
            telegram_io.send(chat_id, responder.error_message(str(exc)))
            return

        log.info(
            "analysing for %s | %s %dx%d -> JPEG %dx%d (%.0f KB)",
            chat_id, norm.source_format, norm.source_width, norm.source_height,
            norm.width, norm.height, len(norm.data) / 1024,
        )

        try:
            result = classifier_client.predict(norm.data, norm.filename)
        except classifier_client.ClassifierError as exc:
            telegram_io.send(chat_id, responder.error_message(str(exc)))
            return

        messages = [
            responder.verdict_message(result, first_name),
            responder.breakdown_message(result, responder.image_quality_note(norm)),
            responder.guidance_message(result),
        ]
        responder.send_sequence(telegram_io.send, chat_id, messages)

        log.info(
            "done for %s | %s @ %.3f (abstain=%s)",
            chat_id, result.get("label"), result.get("confidence", 0),
            result.get("abstain"),
        )

    except Exception:
        log.exception("unexpected failure while analysing for %s", chat_id)
        telegram_io.send(
            chat_id,
            responder.error_message("Something broke on my side while reading it."),
        )


#  One incoming message 
def handle_message(message: dict) -> None:
    chat_id = (message.get("chat") or {}).get("id")
    if chat_id is None:
        return

    sender = message.get("from") or {}
    first_name = sender.get("first_name") or sender.get("username") or ""
    text = (message.get("text") or message.get("caption") or "").strip()

    file_id, declared_type = telegram_io.photo_file_id(message)

    log.info(
        "inbound from %s (%s) | photo=%s | text=%r",
        chat_id, first_name or "no name", bool(file_id), text[:60],
    )

    # Not readable 
    if not file_id:
        kind = telegram_io.unsupported_kind(message)
        if kind:
            telegram_io.send(
                chat_id,
                responder.error_message(f"I can only read photos, and that was {kind}."),
            )
        else:
            telegram_io.send(chat_id, responder.help_text(first_name))
        return

    # Wrong Form ie images
    if declared_type and not declared_type.startswith("image/"):
        telegram_io.send(
            chat_id,
            responder.error_message(
                f"I can only read photos, and that arrived as {declared_type}."
            ),
        )
        return

    telegram_io.send(chat_id, responder.greeting(first_name))
    POOL.submit(analyse_and_reply, chat_id, file_id, declared_type, first_name)


def handle_update(update: dict) -> None:
    message = update.get("message")
    if message:
        handle_message(message)


#  The polling loop 
def run_forever() -> None:
    global _running
    _running = True

    telegram_io.delete_webhook()

    offset = None
    if config.SKIP_BACKLOG:
        try:
            newest = telegram_io.newest_update_id()
            if newest is not None:
                offset = newest + 1
                log.info("skipping messages received while the bot was off")
        except telegram_io.TelegramError:
            log.warning("could not check for a backlog; starting from the beginning")

    backoff = 1.0
    while _running:
        try:
            updates = telegram_io.get_updates(offset)
            backoff = 1.0
        except telegram_io.TelegramError as exc:
            log.warning("%s - retrying in %.0fs", exc, backoff)
            time.sleep(backoff)
            backoff = min(backoff * 2, 60.0)
            continue

        for update in updates:
            offset = update["update_id"] + 1
            try:
                handle_update(update)
            except Exception:
                log.exception("failed to handle update %s", update.get("update_id"))


def stop() -> None:
    global _running
    _running = False
