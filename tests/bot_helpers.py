"""Fakes for the bot tests: a messenger that records what it is asked to send, a stub ingest service and a clock you can move."""
from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

from studybuddy.bot.flow import Button, DownloadError, Incoming
from studybuddy.errors import StudyBuddyError

JPEG = b"\xff\xd8\xff\xe0" + b"\x00" * 60
PDF = b"%PDF-1.4\n" + b"x" * 60
WHEN = datetime(2026, 10, 18, 9, 30, 0)


def run(coro):
    return asyncio.run(coro)


class Clock:
    def __init__(self, now=1_000_000.0):
        self.now = now

    def __call__(self):
        return self.now

    def advance(self, seconds):
        self.now += seconds


@dataclass
class Sent:
    id: int
    chat_id: int
    text: str
    buttons: list = field(default_factory=list)
    reply_to: int | None = None
    edits: list = field(default_factory=list)       # (text, buttons) after each edit

    @property
    def labels(self):
        return [b.label for row in self.buttons for b in row]

    def data_of(self, label):
        return next(b.data for row in self.buttons for b in row if b.label == label)

    @property
    def now_text(self):
        return self.edits[-1][0] if self.edits else self.text

    @property
    def now_labels(self):
        buttons = self.edits[-1][1] if self.edits else self.buttons
        return [b.label for row in buttons for b in row]

    def now_data_of(self, label):
        buttons = self.edits[-1][1] if self.edits else self.buttons
        return next(b.data for row in buttons for b in row if b.label == label)


class FakeMessenger:
    def __init__(self):
        self.messages: list[Sent] = []
        self.files: dict[str, bytes | Exception] = {}
        self.downloads: list[str] = []
        self._next = 100

    async def send(self, chat_id, text, buttons=None, reply_to=None):
        self._next += 1
        self.messages.append(Sent(self._next, chat_id, text, [list(r) for r in (buttons or [])], reply_to))
        return self._next

    async def edit(self, chat_id, message_id, text, buttons=None):
        message = next(m for m in self.messages if m.id == message_id)
        message.edits.append((text, [list(r) for r in (buttons or [])]))

    async def download(self, file_id):
        self.downloads.append(file_id)
        data = self.files.get(file_id, JPEG + file_id.encode())
        if isinstance(data, Exception):
            raise data
        return data

    def texts(self):
        return [m.now_text for m in self.messages]


class Ticket:
    def __init__(self, job_id):
        self.job_id = job_id


class StubJobs:
    def __init__(self):
        self.ids: list[str] = []

    def load_all(self):
        return [type("J", (), {"id": i}) for i in self.ids]


class StubService:
    """Records submit() calls; behaves like the real one for the arguments the bot uses."""

    def __init__(self):
        self.calls: list[dict] = []
        self.jobs = StubJobs()
        self.error: Exception | None = None

    def submit(self, paths, subject=None, source_kind=None, received_at=None, origin="api", is_album=False, cleanup_sources=True):
        if self.error:
            raise self.error
        if len(paths) > 1 and not is_album:
            raise StudyBuddyError("Several files are one document only as an album.")
        for p in paths:
            assert Path(p).is_file(), f"{p} must exist in the spool when it is submitted"
        job_id = f"job-{len(self.calls) + 1}"
        self.calls.append(dict(paths=[Path(p) for p in paths], subject=subject, source_kind=source_kind, received_at=received_at,
                               origin=origin, is_album=is_album, cleanup_sources=cleanup_sources, job_id=job_id))
        self.jobs.ids.append(job_id)
        return Ticket(job_id)


_counter = iter(range(1, 10_000))


def incoming(kind="photo", caption=None, group=None, size=50_000, name=None, message_id=None, chat_id=42, unique=None, file_id=None, **kw):
    n = next(_counter)
    return Incoming(chat_id=chat_id, message_id=message_id if message_id is not None else n, kind=kind,
                    file_id=file_id or f"F{n}", file_unique_id=unique or f"U{n}", sent_at=WHEN, file_size=size, file_name=name,
                    media_group_id=group, caption=caption, **kw)


def button(message: Sent, label: str) -> str:
    return message.now_data_of(label)
