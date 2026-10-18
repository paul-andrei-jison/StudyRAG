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
from ..ingest.results import FLAG_REASONS, IGNORED_FLAGS
from ..ingest.service import IngestService
from ..ingest.subjects import UNSORTED, clean_subject_name
from ..reader.quality import ATTENTION_FLAGS, ATTENTION_QUALITY
from .files import LibraryGuard

BUSY_MESSAGE = "The study buddy is busy reading a document. Try again shortly."
ACTION_TIMEOUT_SECONDS = 15.0
TABLE_HEADERS = ["ID", "Subject", "Title", "Added", "Pages", "Summary", "Flagged pages", "Path"]


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

    def line(self) -> str:
        quality = "?" if self.quality is None else f"{self.quality:.2f}"
        note = f" - {'; '.join(self.reasons)}" if self.reasons else ""
        return f"page {self.page_number}: reading quality {quality}{note}"


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

    def as_list(self) -> list:
        return [self.id, self.subject, self.title, self.added, self.pages, self.summary, self.flagged, self.path]


@dataclass
class DocumentDetail:
    row: DocumentRow
    pages: list[PageInfo] = field(default_factory=list)

    def describe(self) -> str:
        r = self.row
        lines = [f"{r.title or '(untitled)'}", f"Subject: {r.subject or '-'}   Added: {r.added}   Pages: {r.pages}",
                 f"Path: {r.path}", "", f"Summary: {r.summary or '-'}"]
        flagged = [p for p in self.pages if p.flagged]
        lines += ["", "Flagged pages:" if flagged else "No pages were flagged."]
        lines += [f"  {p.line()}" for p in flagged]
        return "\n".join(lines)


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

    def table(self, subject: str | None = None) -> list[list]:
        return [r.as_list() for r in self.rows(subject)]

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
        """Give a document another subject (a new name is allowed and joins the subject list). The file is not moved."""
        try:
            subject = self.subjects.find(raw_subject) or clean_subject_name(raw_subject)
        except StudyBuddyError as exc:
            return ActionResult(False, str(exc))

        def action(index: Index):
            doc = index.store.get_document(doc_id)
            if doc is None:
                return None
            index.update_document(doc_id, subject=subject)
            return doc

        outcome = self._run(action)
        if isinstance(outcome, ActionResult):
            return outcome
        if outcome is None:
            return ActionResult(False, "That document no longer exists. Refresh the list.")
        if subject != UNSORTED:
            self.subjects.add(subject)   # only now that it worked, so a failed change leaves no stray subject
        return ActionResult(True, f"'{outcome.title}' is now filed under {subject}. "
                                  f"The file itself stays where it is: {outcome.file_path}")

    def delete(self, doc_id: int, confirmed: bool = False) -> ActionResult:
        """Remove a document from the index. The original file in the library is never deleted."""
        if not confirmed:
            return ActionResult(False, "Not removed: the removal was not confirmed.")

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
            return ActionResult(False, "That document no longer exists. Refresh the list.")
        return ActionResult(True, f"'{outcome.title}' was removed from the index. "
                                  f"The original file is still in your library: {outcome.file_path}")

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
            return ActionResult(False, "That is taking longer than expected; it will finish in the background. Refresh in a moment.",
                                busy=True)
        except StudyBuddyError as exc:
            return ActionResult(False, str(exc))
        except Exception as exc:
            return ActionResult(False, f"That did not work ({type(exc).__name__}: {exc}).")
