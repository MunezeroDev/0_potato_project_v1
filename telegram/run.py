
# Start the Telegram bot : python run.py
from __future__ import annotations

import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))


def load_env(path: Path) -> int:
    if not path.exists():
        return 0
    loaded = 0
    for line in path.read_text(encoding="utf-8-sig").splitlines():
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

    import bot                 # noqa: E402
    import classifier_client   # noqa: E402
    import config              # noqa: E402
    import imaging             # noqa: E402
    import telegram_io         # noqa: E402

    print("Potato Leaf Doctor - Telegram bridge")
    print(f"  .env values loaded : {n}")
    print(f"  bot token          : {config.redacted_token()}")
    print(f"  model API          : {config.PREDICT_URL}")
    print(f"  mode               : long polling (no webhook, no ngrok)")
    print(f"  message gap        : {config.MESSAGE_GAP_SECONDS:g}s")
    print(f"  HEIC support       : {'yes' if imaging.HEIF_OK else 'no (pip install pillow-heif)'}")

    problems = config.check()
    if problems:
        print("\nConfiguration problems:")
        for p in problems:
            print(f"  - {p}")
        print("\nFix these in telegram/.env, then start again.")
        return 1


    try:
        me = telegram_io.get_me()
    except telegram_io.TelegramError as exc:
        print(f"\n  ! Telegram refused the connection:\n    {exc}")
        return 1
    username = me.get("username", "?")
    print(f"  bot account        : @{username}  ({me.get('first_name', '')})")
    print(f"  open this to chat  : https://t.me/{username}")

    model = classifier_client.health()
    if model is None:
        print(
            f"\n  ! The model service at {config.HEALTH_URL} is not answering."
            "\n    Start it first:  cd serve && python app.py"
            "\n    Starting anyway - senders will get a clear error until it's up."
        )
    else:
        print(f"  model checkpoint   : {model.get('checkpoint')} "
              f"(aug={model.get('aug_mode')}, threshold={model.get('threshold')})")

    print("\nListening. Send the bot a leaf photo. Ctrl+C to stop.\n")

    try:
        bot.run_forever()
    except KeyboardInterrupt:
        print("\nStopping - waiting for any reply already in flight...")
        bot.stop()
        bot.POOL.shutdown(wait=True)
        print("Stopped.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
