"""Process one document all-or-nothing: read, describe, file, index. The service runs this one job at a time.

Order matters: duplicates are found before any model call, the originals are copied into the library (verified by
hash) before the index is written, and if anything fails everything this job created is removed again. The
originals are removed from the inbox only after the index write succeeded and the filed copy re-hashed.
"""
from __future__ import annotations

import json
import sqlite3
import time
import traceback
from dataclasses import replace
from datetime import datetime
from pathlib import Path
from typing import Callable, Sequence

from ..config import Settings
from ..errors import (
    IndexMismatchError, IndexStoreError, LLMError, PermanentIngestError, StudyBuddyError, TransientLLMError,
)
from ..index.index import Index
from ..index.store import DocumentMeta
from ..reader.ocr import OcrEngine
from ..reader.pipeline import read_pages
from ..reader.quality import PageRecord
from ..reader.transcribe import Reader
from .files import FileInfo, classify, document_hash, load_document, sha256_file
from .filing import FiledDocument, Filer, FilingError, safe_move
from .jobs import Job
from .metadata import Chats, describe_document
from .results import DEFERRED, DUPLICATE, FAILED, INDEXED, IngestResult, flagged_pages
from .subjects import SubjectError, SubjectList


class PageReadError(StudyBuddyError):
    """One or more pages could not be read, so the document is not indexed (no half documents)."""

    def __init__(self, message: str, transient: bool):
        super().__init__(message)
        self.transient = transient


def classify_exception(exc: Exception) -> tuple[bool, str]:
    """(transient, message). Transient errors keep the job queued; permanent ones move it to failed/."""
    if isinstance(exc, TransientLLMError):
        return True, f"LM Studio is not available right now ({exc})"
    if isinstance(exc, PageReadError):
        return exc.transient, str(exc)
    if isinstance(exc, FilingError):
        return exc.transient, str(exc)
    if isinstance(exc, IndexMismatchError):
        return True, f"{exc} (the document stays queued until this is fixed)"
    if isinstance(exc, (PermanentIngestError, SubjectError, IndexStoreError, LLMError)) or isinstance(exc, StudyBuddyError):
        return False, str(exc)
    if isinstance(exc, (sqlite3.OperationalError, OSError)):
        return True, f"{type(exc).__name__}: {exc}"
    return False, f"unexpected error: {type(exc).__name__}: {exc}"


class Processor:
    def __init__(self, settings: Settings, index: Index, reader: Reader, ocr: OcrEngine | None, chat: Chats,
                 subjects: SubjectList, filer: Filer | None = None, inbox_dir: Path | None = None,
                 persist: Callable[[Job], None] | None = None, clock: Callable[[], float] = time.monotonic):
        self.settings, self.index, self.reader, self.ocr, self.chat = settings, index, reader, ocr, chat
        self.subjects = subjects
        self.filer = filer or Filer(settings.library_dir)
        self.inbox_dir = inbox_dir
        self.persist = persist
        self.clock = clock
        self.failed_dir = settings.data_dir / "failed"
        self.duplicates_dir = settings.data_dir / "duplicates"

    # ---- main entry ------------------------------------------------------------------------

    def process(self, job: Job) -> IngestResult:
        started = self.clock()
        result = IngestResult(status=FAILED, job_id=job.id, files=list(job.files), source_kinds=list(job.kinds),
                              attempts=job.attempts + 1)
        created: list[Path] = []
        committed = False
        try:
            self._clear_stale_staged(job)
            sources = [Path(f) for f in job.files]
            missing = [s.name for s in sources if not s.is_file()]
            if missing:
                raise PermanentIngestError(f"The file(s) are no longer there: {', '.join(missing)}")
            infos = [classify(s) for s in sources]
            album = job.is_album and len(sources) > 1
            hashes = [sha256_file(s) for s in sources]
            doc_hash = document_hash(hashes, album)

            existing_id = self.index.store.find_by_hash(doc_hash)
            if existing_id is not None:  # before any model call: duplicates cost nothing
                return self._duplicate(job, result, existing_id, doc_hash, sources, started)

            loaded = load_document(sources, job.is_album)
            records = read_pages(loaded.pages, self.reader, self.ocr, self.settings.reader_max_edge)
            self._raise_if_pages_failed(records)

            user_subject = self.subjects.resolve(job.subject, add_if_new=True) if job.subject else None
            desc = describe_document(self.chat, [r.text for r in records], self.subjects, user_subject)

            filed = self.filer.file_document(sources, infos, hashes, desc.subject, desc.title,
                                             datetime.fromisoformat(job.received_at), album)
            created = filed.created
            job.staged = [str(p) for p in created]
            self._persist(job)  # a crash from here on can clean up exactly these files

            records = self._with_filed_paths(records, filed, album)
            meta = DocumentMeta(doc_hash, str(filed.final_path), self._document_kind(job, infos), desc.subject,
                                desc.title, desc.summary)
            added = self.index.add_document(meta, records, expected_pages=len(loaded.pages))
            if not added.created:  # indexed by someone else a moment ago
                Filer.rollback(created)
                created = []
                job.staged = []
                self._persist(job)
                return self._duplicate(job, result, added.doc_id, doc_hash, sources, started)
            committed = True

            warnings = list(desc.warnings)
            if job.cleanup_sources:
                warnings += Filer.remove_sources(sources, filed.page_files, hashes)
                self._remove_empty_inbox_folders(sources)
            job.staged = []
            self._persist(job)

            result.status = INDEXED
            result.subject, result.subject_source = desc.subject, desc.subject_source
            result.title, result.summary = desc.title, desc.summary
            result.page_count = len(records)
            result.flagged_pages = flagged_pages(records)
            result.final_path = str(filed.final_path)
            result.pdf_path = str(filed.pdf_path) if filed.pdf_path else None
            result.doc_id = added.doc_id
            result.warnings = warnings
        except Exception as exc:
            if committed:  # the document is in the index: never delete the files it points to
                result.status = INDEXED
                result.warnings.append(f"indexed, but cleanup afterwards failed: {exc}")
            else:
                Filer.rollback(created)
                job.staged = []
                self._persist(job)
                transient, message = classify_exception(exc)
                result.error, result.transient = message, transient
                if transient:
                    result.status = DEFERRED
                else:
                    result.status = FAILED
                    self._quarantine(job, result, exc)
        result.duration_seconds = round(self.clock() - started, 2)
        return result

    # ---- steps -----------------------------------------------------------------------------

    def _clear_stale_staged(self, job: Job) -> None:
        """A previous attempt crashed after filing but before the index write: remove what it left behind."""
        if job.staged:
            Filer.rollback([Path(p) for p in job.staged])
            job.staged = []
            self._persist(job)

    @staticmethod
    def _raise_if_pages_failed(records: Sequence[PageRecord]) -> None:
        failed = [r for r in records if r.error]
        if failed:
            detail = "; ".join(f"page {r.page_number}: {r.error}" for r in failed[:3])
            more = f" (and {len(failed) - 3} more)" if len(failed) > 3 else ""
            raise PageReadError(f"{len(failed)} of {len(records)} page(s) could not be read: {detail}{more}",
                                transient=any(r.error_transient for r in failed))

    @staticmethod
    def _with_filed_paths(records: list[PageRecord], filed: FiledDocument, album: bool) -> list[PageRecord]:
        if album:
            return [replace(r, source_path=str(filed.page_files[i])) for i, r in enumerate(records)]
        return [replace(r, source_path=str(filed.final_path)) for r in records]

    @staticmethod
    def _document_kind(job: Job, infos: Sequence[FileInfo]) -> str:
        kinds = {("pdf" if i.is_pdf else ("photo-file" if k == "document" else k)) for k, i in zip(job.kinds, infos)}
        return kinds.pop() if len(kinds) == 1 else "mixed"

    def _remove_empty_inbox_folders(self, sources: Sequence[Path]) -> None:
        if self.inbox_dir is None:
            return
        inbox = self.inbox_dir.resolve()
        for parent in {s.resolve().parent for s in sources}:
            if parent != inbox and inbox in parent.parents:
                try:
                    parent.rmdir()  # only succeeds when the folder is empty
                except OSError:
                    pass

    def _persist(self, job: Job) -> None:
        if self.persist:
            self.persist(job)

    # ---- outcomes --------------------------------------------------------------------------

    def _duplicate(self, job: Job, result: IngestResult, doc_id: int, doc_hash: str, sources: Sequence[Path],
                   started: float) -> IngestResult:
        row = self.index.store.get_document(doc_id)
        result.status, result.doc_id = DUPLICATE, doc_id
        if row is not None:
            library_copy = Path(row.file_path)
            result.existing = {"doc_id": row.id, "title": row.title, "subject": row.subject, "file_path": row.file_path,
                               "file_exists": library_copy.exists(), "created_at": row.created_at}
            result.title, result.subject, result.page_count = row.title, row.subject, row.page_count
            result.final_path = row.file_path
            if not library_copy.exists():
                result.warnings.append("the filed copy of the earlier document is missing from the library; "
                                       "the new copy is kept in the duplicates folder so nothing is lost")
        if job.cleanup_sources:
            for src in sources:
                try:
                    safe_move(src, self.duplicates_dir, name=f"{doc_hash[:8]}_{src.name}")
                except FilingError as exc:
                    result.warnings.append(str(exc))
            result.duplicate_dir = str(self.duplicates_dir)
            self._remove_empty_inbox_folders(sources)
        result.duration_seconds = round(self.clock() - started, 2)
        return result

    def _quarantine(self, job: Job, result: IngestResult, exc: Exception) -> None:
        """Permanent failure: keep the files in failed/<job>/ with the reason, so nothing is lost and it can be retried."""
        dest = self.failed_dir / job.id
        moved: list[str] = []
        try:
            dest.mkdir(parents=True, exist_ok=True)
            for src in (Path(f) for f in job.files):
                if job.cleanup_sources and src.is_file():
                    try:
                        moved.append(str(safe_move(src, dest)))
                    except FilingError as move_error:
                        result.warnings.append(str(move_error))
                else:
                    moved.append(str(src))
            reason = {
                "job_id": job.id, "error": result.error, "kind": "permanent", "attempts": job.attempts + 1,
                "failed_at": datetime.now().isoformat(timespec="seconds"), "subject": job.subject,
                "origin": job.origin, "received_at": job.received_at, "is_album": job.is_album,
                "source_kinds": job.kinds, "original_files": job.files, "files": moved,
                "cleanup_sources": job.cleanup_sources,
                "detail": "".join(traceback.format_exception(exc))[-4000:] if result.error.startswith("unexpected") else None,
            }
            (dest / "reason.json").write_text(json.dumps(reason, indent=2), encoding="utf-8")
            result.failed_dir = str(dest)
        except OSError as os_error:
            result.warnings.append(f"could not record the failure in {dest}: {os_error}")
