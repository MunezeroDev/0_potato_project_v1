
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

os.environ.setdefault("TWILIO_ACCOUNT_SID", "AC" + "0" * 32)
os.environ.setdefault("TWILIO_AUTH_TOKEN", "simulation")
os.environ["TWILIO_VALIDATE_SIGNATURE"] = "false"

import bot as botmod  # noqa: E402
import classifier_client  # noqa: E402
import config  # noqa: E402
import twilio_io  # noqa: E402

RULE = "-" * 62


def main() -> int:
    ap = argparse.ArgumentParser(description="Simulate a WhatsApp photo message.")
    ap.add_argument("image", help="path to a leaf photo on disk")
    ap.add_argument("--name", default="Munezero", help="sender's WhatsApp profile name")
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

    # Fake #1: Twilio media download.
    twilio_io.fetch_media = lambda url: (raw, content_type)

    # Fake #2: Twilio outbound send - print instead.
    def fake_send(to, body):
        print(f"\n>>> to {to}\n{body}\n{RULE}")
        return "SM_simulated"

    twilio_io.send = fake_send
    import responder
    responder.twilio_io = twilio_io
    config.MESSAGE_GAP_SECONDS = args.gap
    botmod._MIN_LEAD_SECONDS = 0.0

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

    client = botmod.app.test_client()
    resp = client.post("/whatsapp", data={
        "MessageSid": "SMsimulated0001",
        "From": "whatsapp:+15550001111",
        "ProfileName": args.name,
        "NumMedia": "1",
        "MediaUrl0": "https://api.twilio.com/simulated",
        "MediaContentType0": content_type,
        "Body": "",
    })

    print(RULE)
    print("Immediate webhook reply (what Twilio turns into the greeting):")
    print(RULE)
    body = resp.get_data(as_text=True)
    start, end = body.find("<Message>") + 9, body.find("</Message>")
    print(body[start:end].replace("&amp;", "&").replace("&lt;", "<").replace("&gt;", ">"))
    print(RULE)
    print("Follow-up messages:")

    for t in list(botmod.POOL._threads):
        pass
    botmod.POOL.shutdown(wait=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
