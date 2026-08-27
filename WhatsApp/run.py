"""Start the WhatsApp bot.
    python run.py
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent


def load_env(path: Path) -> int:
    """Minimal .env loader - no dependency on python-dotenv."""
    if not path.exists():
        return 0
    loaded = 0
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:   # real env vars win
            os.environ[key] = value
            loaded += 1
    return loaded


def main() -> int:
    n = load_env(HERE / ".env")

    import classifier_client  # noqa: E402
    import config  # noqa: E402
    import imaging  # noqa: E402
    from bot import app  # noqa: E402

    print(f"Potato Leaf Doctor - WhatsApp bridge")
    print(f"  .env values loaded : {n}")
    print(f"  model API          : {config.PREDICT_URL}")
    print(f"  listening on       : http://{config.HOST}:{config.PORT}/whatsapp")
    print(f"  signature checking : {'on' if config.VALIDATE_SIGNATURE else 'OFF'}")
    print(f"  HEIC support       : {'yes' if imaging.HEIF_OK else 'no (pip install pillow-heif)'}")

    problems = config.check()
    if problems:
        print("\nConfiguration problems:")
        for p in problems:
            print(f"  - {p}")
        print("\nFix these in WhatsApp/.env, then start again.")
        return 1

    model = classifier_client.health()
    if model is None:
        print(
            f"\n  ! The model service at {config.HEALTH_URL} is not answering."
            "\n    Start it first:  cd serve && python app.py"
            "\n    Starting anyway - senders will get a clear error until it's up.\n"
        )
    else:
        print(f"  model checkpoint   : {model.get('checkpoint')} "
              f"(aug={model.get('aug_mode')}, threshold={model.get('threshold')})\n")

    app.run(host=config.HOST, port=config.PORT, debug=config.DEBUG, threaded=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
