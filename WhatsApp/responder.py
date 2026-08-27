
from __future__ import annotations

import logging
import threading
import time

import config
import twilio_io

log = logging.getLogger("whatsapp.responder")

DISEASE_ADVICE = {
    "Early Blight": (
        "Early blight usually starts on older, lower leaves as dark spots with "
        "rings inside them. Remove affected leaves, avoid wetting the foliage "
        "when watering, and rotate the crop next season."
    ),
    "Late Blight": (
        "Late blight moves fast in cool, wet weather and can take a whole field "
        "in days. Act quickly: isolate affected plants and get an agronomist to "
        "confirm before the rest of the plot is exposed."
    ),
    "Healthy": (
        "Nothing disease-like was detected. Worth knowing: 'Healthy' is also "
        "where this model lands when the photo isn't a close-up of a single leaf "
        "at all - so if you photographed a whole plant, the soil, or something "
        "else, retake it before trusting this. Keep checking the lower, older "
        "leaves; that is where trouble usually shows up first."
    ),
}


def _first_name(profile_name: str | None) -> str:
    """WhatsApp profile names are free text. Take a usable first word."""
    if not profile_name:
        return ""
    cleaned = " ".join(profile_name.split())
    if not cleaned:
        return ""
    first = cleaned.split(" ")[0]
    return first[:24]


def greeting(profile_name: str | None) -> str:
    name = _first_name(profile_name)
    who = f"Hi {name}" if name else "Hi there"
    return (
        f"{who} - Potato Leaf Doctor here.\n\n"
        "Got your photo. Looking at it now, this takes a few seconds.\n"
        "I'll send you three short messages: the result, the full scores, "
        "then what it means."
    )


def help_text(profile_name: str | None) -> str:
    name = _first_name(profile_name)
    who = f"Hi {name}" if name else "Hi there"
    return (
        f"{who} - Potato Leaf Doctor here.\n\n"
        "Send me a photo of a single potato leaf and I'll tell you whether it "
        "looks like early blight, late blight, or a healthy leaf.\n\n"
        "For the best result: one leaf, filling most of the frame, in daylight, "
        "against a plain background."
    )


def _bar(fraction: float, width: int = 10) -> str:
    filled = max(0, min(width, round(fraction * width)))
    return "█" * filled + "░" * (width - filled)


def verdict_message(result: dict, profile_name: str | None = None) -> str:
    label = result.get("label", "unknown")
    conf = float(result.get("confidence", 0.0))
    abstain = bool(result.get("abstain", False))

    if abstain:
        return (
            "1/3  No confident call.\n\n"
            f"The closest match was *{label}* at {conf:.0%}, which is below the "
            f"{config.LOW_CONFIDENCE_HINT:.0%} bar this model needs before it will "
            "commit to an answer.\n\n"
            "That usually means the photo is blurry, too far away, has several "
            "leaves in it, or shows something the model was never trained on. "
            "Try again with one leaf filling the frame."
        )

    return (
        "1/3  Result\n\n"
        f"*{label}*\n"
        f"Model score: {conf:.1%}\n\n"
        "Full breakdown coming next."
    )


def breakdown_message(result: dict, image_note: str = "") -> str:
    probs: dict = result.get("probs", {})
    ordered = sorted(probs.items(), key=lambda kv: kv[1], reverse=True)

    lines = ["2/3  How the three classes scored", ""]
    for name, value in ordered:
        lines.append(f"{_bar(float(value))}  {float(value):5.1%}  {name}")

    lines.append("")
    lines.append(
        "These are relative model scores across the three classes, not "
        "calibrated probabilities - they always add up to 100%."
    )

    if image_note:
        lines.append("")
        lines.append(image_note)

    return "\n".join(lines)


def guidance_message(result: dict) -> str:
    label = result.get("label", "")
    abstain = bool(result.get("abstain", False))

    if abstain:
        body = (
            "Nothing to act on from this photo. Retake it with one leaf close up "
            "in daylight and send it again."
        )
    else:
        body = DISEASE_ADVICE.get(label, "Have an agronomist confirm this in person.")

    return (
        "3/3  What this means\n\n"
        f"{body}\n\n"
        "Worth knowing: this model was trained on clean laboratory photographs "
        "of single leaves, not field snapshots, and on far fewer healthy examples "
        "than diseased ones. It is a screening aid for a research prototype - "
        "not agronomic advice, and not a basis for spraying decisions.\n\n"
        "Send another leaf photo any time."
    )


def error_message(reason: str) -> str:
    return (
        "Sorry - I couldn't get a result from that one.\n\n"
        f"{reason}\n\n"
        "Send another photo and I'll try again."
    )


def image_quality_note(norm) -> str:
    """A short line about what WhatsApp did to the photo, or '' if unremarkable."""
    notes = []
    if norm.low_resolution:
        notes.append(
            f"Heads up: WhatsApp delivered this at {norm.source_width}x"
            f"{norm.source_height}px, which is small for a reliable read."
        )
    elif norm.downscaled:
        notes.append(
            f"(Resized from {norm.source_width}x{norm.source_height} before analysis.)"
        )
    if norm.source_format not in ("JPEG", "unknown") and not norm.low_resolution:
        notes.append(f"(Received as {norm.source_format}, converted to JPEG.)")
    return " ".join(notes)


# Paced delivery 
def send_sequence(to: str, messages: list[str], gap: float | None = None,
                  sender=None) -> None:
    """Send messages in order, waiting `gap` seconds between them.

    Called on a background thread so the webhook can return immediately - Twilio
    gives a webhook roughly ten seconds before it gives up, which is nowhere near
    enough for a media download plus CPU inference plus a paced conversation.
    """
    gap = config.MESSAGE_GAP_SECONDS if gap is None else gap
    send = sender or twilio_io.send
    for i, body in enumerate(messages):
        if i:
            time.sleep(gap)
        send(to, body)


def send_sequence_async(to: str, messages: list[str], gap: float | None = None,
                        sender=None) -> threading.Thread:
    t = threading.Thread(
        target=send_sequence,
        args=(to, messages, gap, sender),
        name=f"reply-{to}",
        daemon=True,
    )
    t.start()
    return t
