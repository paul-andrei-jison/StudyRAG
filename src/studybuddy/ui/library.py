"""The logic behind the Library tab: list documents, show their pages, change a subject, remove a document from the index.

Every change goes through the ingest service's single writer with `try_exclusive`, so it never overlaps a document being
read, and never makes the screen wait: if the worker is busy the student is told to try again shortly.
"""
from __future__ import annotations

import json
from concurrent.futures import TimeoutError as FutureTimeout
from dataclasses import dataclass, field
from datetime import datetime, timezone

from PIL import Image

from ..errors import StudyBuddyError
from ..index.index import Index
from ..ingest.relocate import JOURNAL_DIRNAME, Relocator
from ..ingest.results import FLAG_REASONS, IGNORED_FLAGS
from ..ingest.service import IngestService
from ..ingest.subjects import UNSORTED, clean_subject_name
from ..reader.quality import ATTENTION_FLAGS, ATTENTION_QUALITY
from .files import LibraryGuard

BUSY_MESSAGE = "Study Buddy is busy reading a paper. Try again in a moment."
ACTION_TIMEOUT_SECONDS = 15.0


@dataclass
class ActionResult:
    ok: bool
    message: str
    busy: bool = False


@dataclass
class PageInfo:
    page_number: int
    quality: float | None
    reasons: list[str]            # plain language; empty when the page read fine
    hard_to_read: bool
    image_path: str

    @property
    def flagged(self) -> bool:
        return bool(self.reasons) or self.hard_to_read


@dataclass
class DocumentRow:
    id: int
    subject: str
    title: str
    added: str
    pages: int
    summary: str
    flagged: int
    path: str


@dataclass
class DocumentDetail:
    row: DocumentRow
    pages: list[PageInfo] = field(default_factory=list)


def _local_time(stamp: str) -> str:
    try:
        when = datetime.fromisoformat(stamp)
        if when.tzinfo is None:
            when = when.replace(tzinfo=timezone.utc)
        return when.astimezone().strftime("%Y-%m-%d %H:%M")
    except ValueError:
        return stamp


def _page_info(row) -> PageInfo:
    try:
        flags = [f for f in json.loads(row["flags"] or "[]") if f not in IGNORED_FLAGS]
    except ValueError:
        flags = []
    quality = row["quality"]
    reasons = [(row["error"] or FLAG_REASONS["error"]) if f == "error" else FLAG_REASONS.get(f, f) for f in flags]
    hard = (quality is not None and quality < ATTENTION_QUALITY) or bool(ATTENTION_FLAGS & set(flags))
    return PageInfo(row["page_number"], quality, reasons, hard, row["image_path"])


class LibraryView:
    def __init__(self, service: IngestService, guard: LibraryGuard):
        self.service, self.guard = service, guard
        self.subjects = service.subjects
        settings = service.settings
        self.relocator = Relocator(service.index, settings.library_dir, settings.data_dir / JOURNAL_DIRNAME)

    @property
    def index(self) -> Index:
        return self.service.index

    # ---- reading (any thread: the store takes its own read lock) ---------------------------------

    def rows(self, subject: str | None = None) -> list[DocumentRow]:
        wanted = (subject or "").strip().casefold()
        out = []
        for doc in self.index.documents():
            if wanted and (doc.subject or "").casefold() != wanted:
                continue
            pages = [_page_info(p) for p in self.index.store.pages(doc.id)]
            out.append(DocumentRow(doc.id, doc.subject, doc.title, _local_time(doc.created_at), doc.page_count,
                                   doc.summary, sum(1 for p in pages if p.flagged), doc.file_path))
        return sorted(out, key=lambda r: r.id, reverse=True)   # newest first

    def detail(self, doc_id: int) -> DocumentDetail | None:
        row = next((r for r in self.rows() if r.id == doc_id), None)
        if row is None:
            return None
        return DocumentDetail(row, [_page_info(p) for p in self.index.store.pages(doc_id)])

    def page_image(self, doc_id: int, page_number: int) -> Image.Image:
        """The stored original of one page, through the library guard (raises a StudyBuddyError that explains why not)."""
        for page in self.index.store.pages(doc_id):
            if page["page_number"] == page_number:
                return self.guard.render_page(page["image_path"], page_number)
        raise StudyBuddyError(f"Document {doc_id} has no page {page_number}.")

    # ---- changing (through the single writer) --------------------------------------------------

    def change_subject(self, doc_id: int, raw_subject: str) -> ActionResult:
        """Give a document another subject (a new name is allowed and joins the subject list) and move its file(s) to that
        subject's folder. All or nothing: if anything cannot be done the document, its file and the index stay as they were."""
        try:
            subject = self.subjects.find(raw_subject) or clean_subject_name(raw_subject)
        except StudyBuddyError as exc:
            return ActionResult(False, str(exc))

        outcome = self._run(lambda index: self.relocator.relocate(doc_id, subject))
        if isinstance(outcome, ActionResult):
            return outcome
        if subject != UNSORTED:
            self.subjects.add(subject)   # only now that it worked, so a failed change leaves no stray subject
        title = next((d.title for d in self.index.documents() if d.id == doc_id), "") or "The document"
        if not outcome.moved:
            return ActionResult(True, f"'{title}' is already in {subject}.")
        message = f"Moved '{title}' to {subject}. Its file is now in the {subject} folder."
        if outcome.warnings:
            message += " (" + "; ".join(outcome.warnings) + ")"
        return ActionResult(True, message)

    def delete(self, doc_id: int, confirmed: bool = False) -> ActionResult:
        """Remove a document from the index. The original file in the library is never deleted."""
        if not confirmed:
            return ActionResult(False, "Nothing was removed.")

        def action(index: Index):
            doc = index.store.get_document(doc_id)
            if doc is None:
                return None
            index.delete_document(doc_id)
            return doc

        outcome = self._run(action)
        if isinstance(outcome, ActionResult):
            return outcome
        if outcome is None:
            return ActionResult(False, "That paper no longer exists in Study Buddy.")
        return ActionResult(True, f"Removed '{outcome.title}' from Study Buddy. Your original file is still in your folder.")

    def _run(self, action):
        try:
            future = self.service.try_exclusive(action)
        except StudyBuddyError as exc:
            return ActionResult(False, str(exc))
        if future is None:
            return ActionResult(False, BUSY_MESSAGE, busy=True)
        try:
            return future.result(ACTION_TIMEOUT_SECONDS)
        except FutureTimeout:
            return ActionResult(False, "That is taking longer than expected. It will finish in the background; look again in a moment.",
                                busy=True)
        except StudyBuddyError as exc:
            return ActionResult(False, str(exc))
        except Exception as exc:
            return ActionResult(False, f"That did not work: {exc}. Try again.")


def filter_rows(rows: list[DocumentRow], query: str | None) -> list[DocumentRow]:
    """Papers whose title, subject or summary contain every word of the search (any case). An empty search keeps all."""
    words = (query or "").casefold().split()
    if not words:
        return list(rows)
    return [r for r in rows if all(w in f"{r.title} {r.subject} {r.summary}".casefold() for w in words)]
