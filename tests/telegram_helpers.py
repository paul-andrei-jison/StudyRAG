"""A fake Telegram Bot API for offline tests: python-telegram-bot is given this as its HTTP layer, so no network is ever used."""
from __future__ import annotations

import json
import time
from collections import deque

from telegram.error import BadRequest, NetworkError
from telegram.request import BaseRequest

BOT_ID = 777
STUDENT = 42


class FakeTelegram(BaseRequest):
    """Answers the Bot API methods the bot uses and records every call (method, parameters)."""

    def __init__(self, token_in_url: str = "123456789:AAHfake_token-value0123456789abcdefg"):
        self.calls: list[tuple[str, dict]] = []
        self.updates: deque[dict] = deque()
        self.files: dict[str, bytes] = {}          # file_id -> content
        self.file_sizes: dict[str, int] = {}
        self.too_big: set[str] = set()
        self.fail_downloads: set[str] = set()
        self.fail_all = False                      # every call raises a network error
        self.fail_get_me = 0                       # number of getMe calls that fail first
        self.invalid_token = False
        self.messages: list[dict] = []             # what the bot sent, as the student would see it
        self.urls: list[str] = []
        self._ids = iter(range(5000, 10_000))
        self._update_ids = iter(range(1, 10_000))

    # ---- BaseRequest ------------------------------------------------------------------------------------

    async def initialize(self) -> None:
        pass

    async def shutdown(self) -> None:
        pass

    @property
    def read_timeout(self):
        return 5.0

    async def do_request(self, url, method, request_data=None, read_timeout=None, write_timeout=None, connect_timeout=None, pool_timeout=None):
        self.urls.append(url)
        if self.fail_all:
            raise NetworkError("fake network is down")
        if "/file/bot" in url:                                         # a file download
            file_id = url.rsplit("/", 1)[-1].replace(".bin", "")
            if file_id in self.fail_downloads:
                raise NetworkError("download interrupted")
            return 200, self.files.get(file_id, b"")
        name = url.rsplit("/", 1)[-1]
        params = dict(request_data.parameters) if request_data else {}
        self.calls.append((name, params))
        if name == "getMe" and self.invalid_token:
            return 401, json.dumps({"ok": False, "error_code": 401, "description": "Unauthorized"}).encode()
        answer = self._answer(name, params)
        return (200 if answer.get("ok") else answer.get("error_code", 400)), json.dumps(answer).encode()      # errors are HTTP errors, as at Telegram

    # ---- the API ----------------------------------------------------------------------------------------

    def _answer(self, name: str, params: dict) -> dict:
        if name == "getMe":
            if self.fail_get_me > 0:
                self.fail_get_me -= 1
                raise NetworkError("fake network is down")
            return {"ok": True, "result": {"id": BOT_ID, "is_bot": True, "first_name": "Study", "username": "study_buddy_test_bot"}}
        if name == "getUpdates":
            batch = []
            while self.updates:
                batch.append(self.updates.popleft())
            if not batch:
                time.sleep(0.02)                                       # a real long poll waits; do not spin
            return {"ok": True, "result": batch}
        if name == "sendMessage":
            message = {"message_id": next(self._ids), "date": int(time.time()), "chat": {"id": int(params["chat_id"]), "type": "private"},
                       "from": {"id": BOT_ID, "is_bot": True, "first_name": "Study"}, "text": params["text"]}
            if "reply_markup" in params:
                message["reply_markup"] = json.loads(params["reply_markup"]) if isinstance(params["reply_markup"], str) else params["reply_markup"]
            if "reply_parameters" in params:
                rp = params["reply_parameters"]
                message["reply_to"] = (json.loads(rp) if isinstance(rp, str) else rp).get("message_id")
            self.messages.append(message)
            return {"ok": True, "result": {k: v for k, v in message.items() if k not in ("reply_markup", "reply_to")}}
        if name == "editMessageText":
            target = next((m for m in self.messages if m["message_id"] == int(params["message_id"])), None)
            if target is None:
                return {"ok": False, "error_code": 400, "description": "Bad Request: message to edit not found"}
            target["text"] = params["text"]
            markup = params.get("reply_markup")
            target["reply_markup"] = (json.loads(markup) if isinstance(markup, str) else markup) if markup else {"inline_keyboard": []}
            return {"ok": True, "result": True}
        if name in ("answerCallbackQuery", "deleteWebhook"):                # start_polling removes any webhook first
            return {"ok": True, "result": True}
        if name == "getFile":
            file_id = params["file_id"]
            if file_id in self.too_big:
                return {"ok": False, "error_code": 400, "description": "Bad Request: file is too big"}
            return {"ok": True, "result": {"file_id": file_id, "file_unique_id": "u" + file_id, "file_size": len(self.files.get(file_id, b"")),
                                           "file_path": f"{file_id}.bin"}}
        return {"ok": False, "error_code": 404, "description": f"unknown method {name}"}

    # ---- helpers for the tests --------------------------------------------------------------------------

    def methods(self, name: str) -> list[dict]:
        return [p for n, p in self.calls if n == name]

    def labels(self, message: dict) -> list[str]:
        return [b["text"] for row in message.get("reply_markup", {}).get("inline_keyboard", []) for b in row]

    def data_of(self, message: dict, label: str) -> str:
        return next(b["callback_data"] for row in message["reply_markup"]["inline_keyboard"] for b in row if b["text"] == label)

    def questions(self) -> list[dict]:
        return [m for m in self.messages if m.get("reply_markup", {}).get("inline_keyboard")]

    # ---- building updates -------------------------------------------------------------------------------

    def _update(self, **body) -> dict:
        return {"update_id": next(self._update_ids), **body}

    def _message(self, user_id, chat_type="private", **fields) -> dict:
        return {"message_id": next(self._ids), "date": int(time.time()), "chat": {"id": user_id, "type": chat_type},
                "from": {"id": user_id, "is_bot": False, "first_name": "S"}, **fields}

    def photo(self, user_id=STUDENT, content=b"\xff\xd8\xff\xe0" + b"\x00" * 60, caption=None, group=None, size=None, chat_type="private") -> dict:
        file_id = f"ph{next(self._ids)}"
        self.files[file_id] = content
        sizes = [{"file_id": file_id + "s", "file_unique_id": "u" + file_id + "s", "width": 90, "height": 90, "file_size": 1000},
                 {"file_id": file_id, "file_unique_id": "u" + file_id, "width": 1280, "height": 960, "file_size": size or len(content)}]
        fields = {"photo": sizes}
        if caption:
            fields["caption"] = caption
        if group:
            fields["media_group_id"] = group
        return self._update(message=self._message(user_id, chat_type, **fields))

    def document(self, name="scan.pdf", mime="application/pdf", content=b"%PDF-1.4\n" + b"x" * 60, user_id=STUDENT, size=None, caption=None) -> dict:
        file_id = f"doc{next(self._ids)}"
        self.files[file_id] = content
        doc = {"file_id": file_id, "file_unique_id": "u" + file_id, "file_name": name, "file_size": size if size is not None else len(content)}
        if mime:
            doc["mime_type"] = mime
        fields = {"document": doc}
        if caption:
            fields["caption"] = caption
        return self._update(message=self._message(user_id, **fields))

    def text(self, body, user_id=STUDENT) -> dict:
        return self._update(message=self._message(user_id, text=body))

    def command(self, body="/start", user_id=STUDENT) -> dict:
        return self._update(message=self._message(user_id, text=body, entities=[{"type": "bot_command", "offset": 0, "length": len(body)}]))

    def video(self, user_id=STUDENT) -> dict:
        return self._update(message=self._message(user_id, video={"file_id": "vid", "file_unique_id": "uvid", "width": 1, "height": 1,
                                                                    "duration": 1, "file_size": 10}))

    def tap(self, message: dict, label: str, user_id=STUDENT) -> dict:
        return self._update(callback_query={"id": str(next(self._ids)), "chat_instance": "c", "data": self.data_of(message, label),
                                            "from": {"id": user_id, "is_bot": False, "first_name": "S"},
                                            "message": {"message_id": message["message_id"], "date": message["date"],
                                                        "chat": message["chat"], "text": message["text"]}})
