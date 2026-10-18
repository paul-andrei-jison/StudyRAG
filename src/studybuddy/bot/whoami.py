"""Find your own Telegram user id without a third-party bot.

Send any message to your new bot, then run `python scripts/telegram_whoami.py`. It reads the waiting messages once (it never
replies, never confirms them) and prints who sent them. Stop the Study Buddy app first: two programs cannot poll one bot at the same time.
"""
from __future__ import annotations

from dataclasses import dataclass

import httpx

API = "https://api.telegram.org"


@dataclass(frozen=True)
class Sender:
    user_id: int
    username: str | None
    first_name: str | None
    chat_type: str
    count: int


class WhoamiError(Exception):
    """A plain-language reason the lookup failed (never contains the token)."""


def senders(updates: list[dict]) -> list[Sender]:
    found: dict[int, Sender] = {}
    for update in updates:
        message = update.get("message") or update.get("edited_message") or update.get("callback_query", {}).get("message") or {}
        user = update.get("message", {}).get("from") or update.get("callback_query", {}).get("from") or message.get("from")
        if not user or "id" not in user:
            continue
        old = found.get(user["id"])
        found[user["id"]] = Sender(user["id"], user.get("username"), user.get("first_name"), (message.get("chat") or {}).get("type", "?"),
                                   (old.count if old else 0) + 1)
    return sorted(found.values(), key=lambda s: -s.count)


def fetch_updates(token: str, client: httpx.Client | None = None) -> list[dict]:
    own = client is None
    client = client or httpx.Client(timeout=20.0)
    try:
        response = client.get(f"{API}/bot{token}/getUpdates", params={"timeout": 0, "allowed_updates": '["message","callback_query"]'})
    except httpx.HTTPError as exc:
        raise WhoamiError(f"Could not reach Telegram ({type(exc).__name__}). Check the internet connection.") from None
    finally:
        if own:
            client.close()
    if response.status_code in (401, 404):
        raise WhoamiError("Telegram does not accept this bot token. Copy it again from @BotFather (/mybots, API Token).")
    if response.status_code == 409:
        raise WhoamiError("Another program is already reading this bot's messages. Stop the Study Buddy app (or any other copy) and try again.")
    if response.status_code != 200:
        raise WhoamiError(f"Telegram answered with an error (HTTP {response.status_code}).")
    return response.json().get("result", [])
