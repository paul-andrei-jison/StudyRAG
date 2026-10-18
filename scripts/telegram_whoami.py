"""Print the Telegram user id of whoever has messaged your bot (so you can put it in .env as SB_TELEGRAM_USER_ID).

  1. Create the bot with @BotFather and put its token in .env as TELEGRAM_BOT_TOKEN.
  2. Open your bot in Telegram and send it any message (for example "hi").
  3. Stop the Study Buddy app, then run:  python scripts/telegram_whoami.py

It only reads; it never replies and never marks anything as read.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from dotenv import load_dotenv  # noqa: E402

import os  # noqa: E402

from studybuddy.bot.whoami import WhoamiError, fetch_updates, senders  # noqa: E402
from studybuddy.config import REPO_ROOT  # noqa: E402


def main() -> int:
    load_dotenv(REPO_ROOT / ".env")
    token = (os.environ.get("TELEGRAM_BOT_TOKEN") or "").strip()
    if not token:
        print("Put TELEGRAM_BOT_TOKEN=<the token from @BotFather> in .env first.")
        return 1
    try:
        found = senders(fetch_updates(token))
    except WhoamiError as exc:
        print(f"[FAIL] {exc}")
        return 1
    if not found:
        print("No messages are waiting. Open your bot in Telegram, send it any message, and run this again.")
        return 1
    print("Messages were sent to your bot by:")
    for s in found:
        who = " ".join(x for x in (s.first_name, f"(@{s.username})" if s.username else None) if x) or "(no name)"
        print(f"  user id {s.user_id}   {who}   {s.count} message(s), chat type: {s.chat_type}")
    print("\nPut YOUR id in .env as:  SB_TELEGRAM_USER_ID=<that number>")
    if len(found) > 1:
        print("More than one account wrote to the bot: use the id that is yours. Anyone else will be ignored once the id is set.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
