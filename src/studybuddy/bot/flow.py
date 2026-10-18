"""The bot's conversation, with no Telegram types in it (the adapter translates).

A file arrives -> it is written to the spool at once -> the student is asked which subject (inline buttons) unless the caption names a
known one -> each file is submitted to the ingest service as its OWN job (several photos become one document only if the student taps
"Make these one document") -> when the job ends the student gets a plain-language message.

Everything the bot still owes is in `StateStore` (on disk), so `start()` after a restart simply carries on.
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Callable, Protocol, Sequence

from ..errors import StudyBuddyError
from ..ingest.results import DEFERRED, DUPLICATE, FAILED, INDEXED, IngestResult
from ..ingest.subjects import SubjectError, SubjectList, UNSORTED, clean_subject_name
from .spool import Spool
from .state import (ASKING, AWAITING_OTHER, CLOSED, DONE, DOWNLOADING, OPEN, SUBMITTED, Item, Question, StateStore, new_id)

log = logging.getLogger(__name__)

DOWNLOAD_LIMIT_BYTES = 20 * 1000 * 1000     # what a bot may fetch from Telegram (getFile)
ANSWER_SECONDS = 5 * 60                     # no answer by then: auto-detect
SETTLE_SECONDS = 2.0                        # photos sent together arrive within a moment of each other
RETRY_SECONDS = 60.0
MAX_DOWNLOAD_ATTEMPTS = 5
IMAGE_KINDS = {"photo", "photo-file"}
HANDWRITING = "Handwritten names, dates, scores and answer letters may be read wrongly: check them against the photo."
SUPPORTED = "photos, images sent as a file, and PDFs"
MAX_CAPTION_SUBJECT_CHARS = 40


class DownloadError(Exception):
    """Telegram would not give the file (network trouble, expired file id). Worth retrying."""


class TooBigError(DownloadError):
    """Over the size a bot may download."""


@dataclass(frozen=True)
class Button:
    label: str
    data: str            # callback data, at most 64 bytes


class Messenger(Protocol):
    async def send(self, chat_id: int, text: str, buttons: Sequence[Sequence[Button]] | None = None,
                   reply_to: int | None = None) -> int: ...
    async def edit(self, chat_id: int, message_id: int, text: str, buttons: Sequence[Sequence[Button]] | None = None) -> None: ...
    async def download(self, file_id: str) -> bytes: ...


@dataclass
class Incoming:
    """One file the student sent, as the adapter describes it."""
    chat_id: int
    message_id: int
    kind: str                       # photo | photo-file | pdf | unsupported
    file_id: str
    file_unique_id: str
    sent_at: datetime               # local time of the message
    file_size: int | None = None
    file_name: str | None = None
    media_group_id: str | None = None
    caption: str | None = None
    what: str = "that kind of file"  # for the "unsupported" reply, e.g. "a Word document"


def _mb(size: int) -> str:
    return f"{size / 1_000_000:.1f} MB"


class BotFlow:
    def __init__(self, service, subjects: SubjectList, spool: Spool, state: StateStore, messenger: Messenger,
                 clock: Callable[[], float] = time.time):
        self.service, self.subjects, self.spool, self.state, self.messenger, self.clock = service, subjects, spool, state, messenger, clock

    # ---- start-up --------------------------------------------------------------------------------------

    async def start(self) -> None:
        """After a restart: tidy the spool, then reconcile what was in flight."""
        self.spool.sweep_partials()
        self.state.sweep_done(self.clock())
        live = {job.id for job in self.service.jobs.load_all()}
        for item in self.state.find_item(lambda i: i.stage == SUBMITTED):
            if item.job_id not in live:      # the job ended while the bot was not running: its result was never seen
                await self._finish_item(item, f"{self._label(item)} was processed while I was switched off. Please look for it in your "
                                              f"library (or the Status panel if it failed).")
        for item in self.state.find_item(lambda i: i.stage == DOWNLOADING and i.attempts == 0):
            item.attempts, item.next_attempt_at = 1, self.clock()          # the app stopped mid-download: try again now
            self.state.save_item(item)
        for item in self.state.find_item(lambda i: i.stage == ASKING):
            question = self.state.question(item.question_id) if item.question_id else None
            if question is None or question.stage == CLOSED:               # a crash between "saved" and "asked", or before "submitted"
                item.question_id = None
                self.state.save_item(item)
                await self._attach_question(item)
        await self.tick()

    # ---- a file arrives --------------------------------------------------------------------------------

    async def on_file(self, inc: Incoming) -> None:
        if self.state.already_seen(inc.chat_id, inc.message_id, inc.file_unique_id):
            return
        if inc.kind == "unsupported":
            await self.messenger.send(inc.chat_id, f"I can only file {SUPPORTED}, not {inc.what}. Nothing was saved.", reply_to=inc.message_id)
            return
        if inc.file_size is not None and inc.file_size > DOWNLOAD_LIMIT_BYTES:
            await self.messenger.send(
                inc.chat_id, f"That file is too big for me to download ({_mb(inc.file_size)}; the limit for bots is 20 MB). Send a smaller or "
                             f"compressed copy, split the PDF, or copy it into the inbox folder on the computer. Nothing was saved.",
                reply_to=inc.message_id)
            return
        item = Item(id=new_id(), chat_id=inc.chat_id, message_id=inc.message_id, kind=inc.kind, file_id=inc.file_id,
                    file_unique_id=inc.file_unique_id, file_size=inc.file_size, original_name=inc.file_name,
                    media_group_id=inc.media_group_id, caption=(inc.caption or "").strip() or None,
                    received_at=inc.sent_at.isoformat(timespec="seconds"), created_at=self.clock())
        self.state.save_item(item)
        if await self._download(item):
            await self._attach_question(item)

    async def _download(self, item: Item) -> bool:
        """Fetch the file into the spool. On failure the item stays DOWNLOADING and is retried by `tick()`."""
        try:
            data = await self.messenger.download(item.file_id)
        except TooBigError:
            await self._finish_item(item, f"{self._label(item)} is too big for me to download (the limit for bots is 20 MB). "
                                          f"Send a smaller copy. Nothing was saved.")
            return False
        except DownloadError as exc:
            item.attempts += 1
            item.next_attempt_at = self.clock() + RETRY_SECONDS
            self.state.save_item(item)
            log.warning("download of item %s failed (attempt %d): %s", item.id, item.attempts, exc)
            if item.attempts >= MAX_DOWNLOAD_ATTEMPTS:
                await self._finish_item(item, f"I could not download {self._label(item)} after {item.attempts} tries. Please send it again.")
            elif item.attempts == 1:
                await self.messenger.send(item.chat_id, f"I couldn't download {self._label(item)} yet. I'll keep trying for a few minutes; "
                                                        f"if it doesn't work I'll tell you.", reply_to=item.message_id)
            return False
        saved = self.spool.write(data, item.original_name, fallback=item.kind)
        item.spool_path, item.sha256, item.stage = str(saved.path), saved.sha256, ASKING
        self.state.save_item(item)
        return True

    # ---- the question ----------------------------------------------------------------------------------

    def _caption_subjects(self, items: Sequence[Item]) -> tuple[str | None, str | None]:
        """(a known subject named by a caption, a valid NEW subject name offered by a caption)."""
        for item in items:
            caption = (item.caption or "").strip()
            if caption.casefold().startswith("subject:"):
                caption = caption.split(":", 1)[1].strip()
            if not caption or "\n" in caption:
                continue
            known = self.subjects.find(caption)
            if known:
                return known, None
        for item in items:
            caption = (item.caption or "").strip()
            if caption.casefold().startswith("subject:"):
                caption = caption.split(":", 1)[1].strip()
            if caption and "\n" not in caption and len(caption) <= MAX_CAPTION_SUBJECT_CHARS:
                try:
                    return None, clean_subject_name(caption)
                except SubjectError:
                    continue
        return None, None

    async def _attach_question(self, item: Item) -> None:
        """Put a downloaded item under a question: join the open question of its photo group, or start one."""
        now = self.clock()
        question = None
        if item.media_group_id:
            question = next((q for q in self.state.all_questions()
                             if q.group_key == item.media_group_id and q.stage in (OPEN, AWAITING_OTHER)), None)
        if question is None:
            question = Question(id=new_id(), chat_id=item.chat_id, item_ids=[], created_at=now, group_key=item.media_group_id)
        question.item_ids.append(item.id)
        item.question_id = question.id
        self.state.save_item(item)
        items = self._items_of(question)
        question.caption_subject, question.caption_suggestion = self._caption_subjects(items)
        question.settle_until = now + (SETTLE_SECONDS if item.media_group_id else 0.0)
        self.state.save_question(question)
        if question.asked:
            await self._render(question)                  # a late photo of the same group: update the count
        elif not item.media_group_id:
            await self.ask(question)

    def _items_of(self, question: Question) -> list[Item]:
        items = [self.state.item(i) for i in question.item_ids]
        return sorted((i for i in items if i is not None), key=lambda i: (i.message_id, i.created_at))

    def _keyboard(self, question: Question, items: Sequence[Item]) -> list[list[Button]]:
        rows: list[list[Button]] = []
        if question.caption_suggestion:
            rows.append([Button(f"Add '{question.caption_suggestion}' as a new subject", f"c:{question.id}")])
        names = question.subjects
        for i in range(0, len(names), 2):
            rows.append([Button(name, f"s:{question.id}:{i + j}") for j, name in enumerate(names[i:i + 2])])
        rows.append([Button("Other...", f"o:{question.id}"), Button("Auto-detect", f"a:{question.id}")])
        if len(items) > 1 and all(i.kind in IMAGE_KINDS for i in items):
            rows.append([Button("Keep separate", f"k:{question.id}") if question.album
                         else Button("Make these one document", f"m:{question.id}")])
        return rows

    def _text(self, question: Question, items: Sequence[Item]) -> str:
        n = len(items)
        photos = all(i.kind in IMAGE_KINDS for i in items)
        noun = ("photo" if photos else "file") if n == 1 else (f"{n} photos" if photos else f"{n} files")
        if question.stage == AWAITING_OTHER:
            return "Type the name of the new subject (for example: Astronomy)."
        if question.album and n > 1:
            head = f"These {n} photos will be ONE document. Which subject is it?"
        elif n == 1:
            head = f"Which subject is this {noun} for?"
        else:
            head = f"{noun}: same subject for all?"
        extra = ""
        if question.caption_suggestion:
            extra = f"\n\"{question.caption_suggestion}\" in the caption is not a subject I know."
        return f"{head}{extra}\n(No answer in 5 minutes: I'll let the study buddy pick.)"

    async def ask(self, question: Question) -> None:
        """Send the question (once all its files are saved)."""
        items = self._items_of(question)
        if not items or question.asked:
            return
        if question.caption_subject:                      # the caption already answers it
            await self._finish(question, question.caption_subject, f"Using the subject from your caption: {question.caption_subject}.")
            return
        question.subjects = self.subjects.names()
        question.deadline = self.clock() + ANSWER_SECONDS
        question.message_id = await self.messenger.send(question.chat_id, self._text(question, items), self._keyboard(question, items),
                                                        reply_to=items[0].message_id)
        question.asked = True
        self.state.save_question(question)

    async def _render(self, question: Question) -> None:
        items = self._items_of(question)
        if question.message_id is not None:
            buttons = ([[Button("Back", f"b:{question.id}"), Button("Auto-detect", f"a:{question.id}")]]
                       if question.stage == AWAITING_OTHER else self._keyboard(question, items))
            await self.messenger.edit(question.chat_id, question.message_id, self._text(question, items), buttons)

    # ---- answers ---------------------------------------------------------------------------------------

    async def on_callback(self, chat_id: int, message_id: int, data: str) -> str:
        """A button was tapped. Returns the short notice Telegram shows to the student."""
        parts = data.split(":")
        question = self.state.question(parts[1]) if len(parts) >= 2 else None
        if question is None or question.chat_id != chat_id or question.stage == CLOSED:
            return "That question is already answered."
        action = parts[0]
        if action == "s" and len(parts) == 3 and parts[2].isdigit() and int(parts[2]) < len(question.subjects):
            await self._finish(question, question.subjects[int(parts[2])])
        elif action == "a":
            await self._finish(question, None)
        elif action == "c" and question.caption_suggestion:
            await self._finish(question, question.caption_suggestion)
        elif action == "o":
            question.stage, question.awaiting_since = AWAITING_OTHER, self.clock()
            self.state.save_question(question)
            await self._render(question)
        elif action == "b":
            question.stage, question.awaiting_since = OPEN, 0.0
            self.state.save_question(question)
            await self._render(question)
        elif action in ("m", "k"):
            question.album = action == "m"
            self.state.save_question(question)
            await self._render(question)
        else:
            return "That button is out of date."
        return "Got it."

    async def on_text(self, chat_id: int, text: str) -> None:
        """A text message: the new subject if "Other..." was tapped, otherwise a short hint."""
        waiting = sorted((q for q in self.state.all_questions() if q.chat_id == chat_id and q.stage == AWAITING_OTHER),
                         key=lambda q: q.awaiting_since, reverse=True)
        if not waiting:
            await self.messenger.send(chat_id, f"Send me photos or PDFs of your handouts ({SUPPORTED}). I'll ask which subject, file them "
                                               f"and tell you when they are ready. Ask questions about them in the web app.")
            return
        try:
            name = self.subjects.find(text) or clean_subject_name(text)
        except SubjectError as exc:
            await self.messenger.send(chat_id, f"{exc} Type another name, or tap Back / Auto-detect.")
            return
        await self._finish(waiting[0], name)

    async def tick(self) -> None:
        """Called every second or two: retry downloads, ask settled photo groups, apply the 5-minute fallback."""
        now = self.clock()
        for item in self.state.find_item(lambda i: i.stage == DOWNLOADING and i.next_attempt_at <= now and i.attempts > 0):
            if await self._download(item):
                await self._attach_question(item)
        for question in self.state.all_questions():
            if question.stage == CLOSED:
                continue
            if not question.asked:
                if question.settle_until <= now and all(i.stage == ASKING for i in self._items_of(question)):
                    await self.ask(question)
            elif question.deadline and question.deadline <= now:
                await self._finish(question, None, "No answer in 5 minutes, so I'm letting the study buddy pick the subject.")

    # ---- submit ----------------------------------------------------------------------------------------

    async def _finish(self, question: Question, subject: str | None, note: str | None = None) -> None:
        items = [i for i in self._items_of(question) if i.stage == ASKING]
        question.stage = CLOSED
        self.state.save_question(question)
        if not items:
            return
        album = question.album and len(items) > 1
        groups = [items] if album else [[i] for i in items]
        queued = 0
        problem = None
        for group in groups:
            try:
                ticket = self.service.submit(
                    [Path(i.spool_path) for i in group], subject=subject, source_kind=[self._source_kind(i) for i in group],
                    received_at=datetime.fromisoformat(group[0].received_at), origin="bot", is_album=len(group) > 1,
                    cleanup_sources=True)
            except StudyBuddyError as exc:
                problem = str(exc)
                continue
            for i in group:
                i.job_id, i.subject, i.stage = ticket.job_id, subject, SUBMITTED
                self.state.save_item(i)
            queued += 1
        summary = self._summary(subject, queued, len(items), album, note, problem)
        if question.message_id is not None:
            await self.messenger.edit(question.chat_id, question.message_id, summary, [])
        else:
            await self.messenger.send(question.chat_id, summary)

    @staticmethod
    def _source_kind(item: Item) -> str:
        return item.kind if item.kind in ("photo", "photo-file", "pdf") else "document"

    @staticmethod
    def _summary(subject: str | None, queued: int, total: int, album: bool, note: str | None, problem: str | None) -> str:
        what = "subject: " + subject if subject else "subject: auto-detect"
        if album:
            line = f"Queued {total} photos as one document ({what})."
        else:
            line = f"Queued {queued} document{'s' if queued != 1 else ''} ({what})."
        if problem:
            line += f" {total - queued} could not be queued ({problem}); they are safe in the _Unsorted folder."
        line += " I'll message you when each is ready."
        return f"{note}\n{line}" if note else line

    # ---- results ---------------------------------------------------------------------------------------

    def _label(self, item: Item) -> str:
        if item.original_name:
            return f"'{item.original_name}'"
        return {"photo": "your photo", "photo-file": "your image", "pdf": "your PDF"}.get(item.kind, "your file")

    async def on_result(self, result: IngestResult) -> None:
        items = self.state.find_item(lambda i: i.job_id == result.job_id and i.stage == SUBMITTED)
        if not items:
            return                                       # a job from the web app or the inbox, not ours
        items.sort(key=lambda i: i.message_id)
        first = items[0]
        if result.status == DEFERRED:       # the service reports a deferral once per job
            wait = f" in about {max(1, round(result.retry_in_seconds / 60))} minute(s)" if result.retry_in_seconds else " shortly"
            await self.messenger.send(first.chat_id, "The reading model isn't reachable right now (is LM Studio running?). "
                                                     f"Your file is saved and I'll try again automatically{wait}.", reply_to=first.message_id)
            return
        text = self.describe(result, len(items))
        await self.messenger.send(first.chat_id, text, reply_to=first.message_id)
        for item in items:
            item.stage, item.finished_at = DONE, self.clock()
            self.state.save_item(item)

    async def _finish_item(self, item: Item, text: str) -> None:
        await self.messenger.send(item.chat_id, text, reply_to=item.message_id)
        item.stage, item.finished_at = DONE, self.clock()
        self.state.save_item(item)

    @staticmethod
    def describe(result: IngestResult, count: int = 1) -> str:
        if result.status == INDEXED:
            name = Path(result.final_path).name if result.final_path else (result.title or "your document")
            where = result.subject or UNSORTED
            if result.subject_source == "fallback" or where == UNSORTED:
                where = f"{UNSORTED} (I couldn't tell the subject; change it in the Library tab)"
            elif result.subject_source == "auto":
                where += " (auto-detected)"
            lines = [f"Filed: {name}", f"Subject: {where}", f"Title: {result.title}" if result.title else "",
                     f"{result.page_count} page{'s' if result.page_count != 1 else ''}."]
            for page in result.flagged_pages:
                reasons = "; ".join(page.reasons) if page.reasons else "hard to read"
                lines.append(f"{'Check page' if page.needs_attention else 'Note: page'} {page.page_number}: {reasons}.")
            if any(p.needs_attention for p in result.flagged_pages):
                lines.append("For a clearer read, retake the photo in good light, or send it as a file (full resolution).")
            lines += [f"Note: {w}." if not w.endswith(".") else f"Note: {w}" for w in result.warnings]
            lines.append(HANDWRITING)
            return "\n".join(line for line in lines if line)
        if result.status == DUPLICATE:
            known = result.existing or {}
            title = known.get("title") or result.title or "an earlier document"
            subject = known.get("subject") or result.subject or "your library"
            return (f"I already have this one: '{title}' ({subject}, {result.page_count} page{'s' if result.page_count != 1 else ''}). "
                    f"I did not add it again; the new copy is kept in {result.duplicate_dir or 'the duplicates folder'}.")
        reason = (result.error or "something went wrong").rstrip(".")
        kept = f" Your file is safe in {result.failed_dir}; retry it from the Status panel." if result.failed_dir else ""
        return f"I couldn't read this one: {reason}.{kept}"
