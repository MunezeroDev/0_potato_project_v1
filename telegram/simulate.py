"""Test the bot with a photo from disk. No Telegram account, no network.

    python simulate.py "..\\results\\uploads\\3d76551299be.jpg"
    python simulate.py "..\\results\\uploads\\3d76551299be.jpg" --fake-model

Prints the exact four messages a sender would receive.
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

# config.check() must pass without a real token.
os.environ.setdefault("TELEGRAM_BOT_TOKEN", "123456789:" + "S" * 35)

import bot as botmod         # noqa: E402
import classifier_client     # noqa: E402
import config                # noqa: E402
import telegram_io           # noqa: E402

RULE = "-" * 62


def strip_html(text: str) -> str:
    """Rough render of Telegram HTML for a terminal."""
    for tag in ("<b>", "</b>", "<i>", "</i>", "<pre>", "</pre>", "<code>", "</code>"):
        text = text.replace(tag, "")
    return (text.replace("&lt;", "<").replace("&gt;", ">").replace("&amp;", "&"))


def main() -> int:
    ap = argparse.ArgumentParser(description="Simulate a Telegram photo message.")
    ap.add_argument("image", help="path to a leaf photo on disk")
    ap.add_argument("--name", default="Munezero", help="sender's Telegram first name")
    ap.add_argument("--gap", type=float, default=0.0,
                    help="seconds between result messages (default 0 for speed)")
    ap.add_argument("--fake-model", action="store_true",
                    help="don't call the model service; use a canned result")
    args = ap.parse_args()

    path = Path(args.image)
    if not path.exists():
        print(f"No such file: {path}")
        return 1

    raw = path.read_bytes()
    suffix = path.suffix.lower().lstrip(".") or "jpeg"
    content_type = f"image/{'jpeg' if suffix in ('jpg', 'jpeg') else suffix}"

    # Fake #1: the Telegram download.
    telegram_io.fetch_file = lambda file_id: (raw, content_type)

    # Fake #2: outbound send - print instead.
    def fake_send(to, body):
        print(f"\n>>> to chat {to}\n{strip_html(body)}\n{RULE}")
        return 1

    telegram_io.send = fake_send
    telegram_io.send_typing = lambda chat_id: None
    config.MESSAGE_GAP_SECONDS = args.gap

    if args.fake_model:
        classifier_client.predict = lambda data, filename="x.jpg": {
            "label": "Early Blight",
            "raw_label": "Potato___Early_blight",
            "index": 0,
            "confidence": 0.9731,
            "probs": {"Early Blight": 0.9731, "Late Blight": 0.0201,
                      "Healthy": 0.0068},
            "abstain": False,
            "uid": "simulated",
        }
        print("(using a canned model result)")
    else:
        if classifier_client.health() is None:
            print(f"The model service at {config.HEALTH_URL} is not answering.")
            print("Start it with:  cd serve && python app.py")
            print("Or re-run this with --fake-model to test the messaging only.")
            return 1

    print(RULE)
    print("What the sender receives:")
    print(RULE)

    botmod.handle_update({
        "update_id": 1,
        "message": {
            "message_id": 1,
            "chat": {"id": 5550001111, "type": "private"},
            "from": {"id": 5550001111, "first_name": args.name},
            "photo": [
                {"file_id": "simulated_small", "width": 90, "height": 90},
                {"file_id": "simulated_large", "width": 1280, "height": 1280},
            ],
        },
    })

    botmod.POOL.shutdown(wait=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
