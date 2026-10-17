"""The single-writer ingest service.

One worker thread is the ONLY thing that reads documents into the index or writes to the library. The watcher, the
bot and the UI all go through `submit()` / `run_exclusive()`, so they can never collide. Jobs are persisted
(see jobs.py), so a restart or LM Studio being off loses nothing: transient failures are retried with backoff.
"""
from __future__ import annotations

import json
import logging
import threading
import time
from collections import deque
from concurrent.futures import Future
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Callable, Sequence, TypeVar

from ..config import Settings
from ..errors import StudyBuddyError
from ..index.index import Index
from ..reader.ocr import OcrEngine
from ..reader.transcribe import Reader
from .jobs import Job, JobStore
from .metadata import Chats
from .processor import Processor
from .results import DEFERRED, FAILED, IngestResult
from .subjects import SubjectList

log = logging.getLogger(__name__)
T = TypeVar("T")

Callback = Callable[[IngestResult], None]


@dataclass
class Ticket:
    job_id: str
    reported: Future = field(default_factory=Future)  # the first outcome: indexed, duplicate, failed OR deferred
    final: Future = field(default_factory=Future)     # the last outcome: never `deferred`

    def result(self, timeout: float | None = None) -> IngestResult:
        """Wait for the final outcome (a deferred job keeps waiting until a retry succeeds or fails)."""
        return self.final.result(timeout)


class IngestService:
    def __init__(self, settings: Settings, index: Index, reader: Reader, ocr: OcrEngine | None, chat: Chats,
                 subjects: SubjectList, on_result: Callback | None = None, clock: Callable[[], float] = time.time,
                 poll_interval: float = 1.0, processor: Processor | None = None):
        self.settings = settings
        self.index = index
        self.subjects = subjects
        self.jobs = JobStore(settings.data_dir / "queue", clock=clock)
        self.inbox_dir = settings.data_dir / "inbox"
        self.processor = processor or Processor(settings, index, reader, ocr, chat, subjects,
                                                inbox_dir=self.inbox_dir, persist=self.jobs.save)
        self._callbacks: list[Callback] = [on_result] if on_result else []
        self._poll = poll_interval
        self._cond = threading.Condition()
        self._exclusive: deque[tuple[Callable[[Index], object], Future]] = deque()
        self._tickets: dict[str, Ticket] = {}
        self._thread: threading.Thread | None = None
        self._stopping = False
        self._busy = False

    # ---- lifecycle -------------------------------------------------------------------------

    def start(self) -> "IngestService":
        if self._thread and self._thread.is_alive():
            return self
        self._stopping = False
        self.jobs.recover()  # jobs a crash left `running` go back to pending
        self._thread = threading.Thread(target=self._run, name="ingest-worker", daemon=True)
        self._thread.start()
        return self

    def stop(self, wait: bool = True, timeout: float | None = None) -> None:
        """Stop after the job in flight (it is never abandoned half way). Queued jobs stay on disk."""
        with self._cond:
            self._stopping = True
            self._cond.notify_all()
        if wait and self._thread:
            self._thread.join(timeout)

    def wake(self) -> None:
        with self._cond:
            self._cond.notify_all()

    def add_callback(self, callback: Callback) -> None:
        self._callbacks.append(callback)

    # ---- public API ------------------------------------------------------------------------

    def submit(self, paths: Sequence[Path | str], subject: str | None = None, source_kind: str | Sequence[str] | None = None,
               received_at: datetime | None = None, origin: str = "api", is_album: bool = False,
               cleanup_sources: bool = True) -> Ticket:
        """Queue a document (one file, or several photos with `is_album=True`). Thread-safe, returns at once.

        `subject` is the student's choice (typed subjects are added to the list); None means auto-detect. An invalid
        subject raises SubjectError here, so the caller can say so immediately instead of failing later."""
        if self._stopping:
            raise StudyBuddyError("The ingest service is stopping and takes no new documents.")
        paths = [Path(p) for p in paths]
        if not paths:
            raise ValueError("submit() needs at least one file")
        if isinstance(source_kind, str) or source_kind is None:
            kinds = [source_kind or "document"] * len(paths)
        else:
            kinds = list(source_kind)
        canonical = self.subjects.resolve(subject, add_if_new=True) if subject else None
        with self._cond:  # create and register under the worker's lock: it can never claim a job it has no ticket for
            job = self.jobs.create([str(p) for p in paths], kinds, received_at or datetime.now(), origin=origin,
                                   subject=canonical, is_album=is_album, cleanup_sources=cleanup_sources)
            ticket = Ticket(job.id)
            self._tickets[job.id] = ticket
            self._cond.notify_all()
        return ticket

    def run_exclusive(self, fn: Callable[[Index], T]) -> "Future[T]":
        """Run `fn(index)` on the worker thread, between jobs, so UI actions (delete, change subject, move) can
        never overlap an ingest. The future carries the return value or the exception."""
        future: Future = Future()
        with self._cond:
            if self._stopping:
                raise StudyBuddyError("The ingest service is stopping.")
            self._exclusive.append((fn, future))
            self._cond.notify_all()
        return future

    def status(self) -> dict:
        counts = self.jobs.counts()
        counts["busy"] = self._busy
        counts["exclusive_waiting"] = len(self._exclusive)
        return counts

    def wait_idle(self, timeout: float = 30.0) -> bool:
        """Block until nothing is running, no action is waiting and no job is ready (deferred jobs do not count)."""
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            with self._cond:
                waiting = self._busy or self._exclusive
            wait = self.jobs.seconds_until_next()
            if not waiting and (wait is None or wait > 0):
                return True
            time.sleep(0.01)
        return False

    def retry_failed(self) -> list[Ticket]:
        """Put permanently failed documents (failed/<job>/reason.json) back in the queue."""
        tickets: list[Ticket] = []
        failed_root = self.settings.data_dir / "failed"
        for reason_path in sorted(failed_root.glob("*/reason.json")) if failed_root.exists() else []:
            try:
                info = json.loads(reason_path.read_text(encoding="utf-8"))
                files = [Path(f) for f in info["files"]]
                if not all(f.is_file() for f in files):
                    continue
                tickets.append(self.submit(files, subject=info.get("subject"), source_kind=info.get("source_kinds"),
                                           received_at=datetime.fromisoformat(info["received_at"]),
                                           origin=info.get("origin", "api"), is_album=bool(info.get("is_album")),
                                           cleanup_sources=bool(info.get("cleanup_sources", True))))
                reason_path.rename(reason_path.with_name("reason.json.retried"))
            except (OSError, ValueError, KeyError, StudyBuddyError) as exc:
                log.warning("could not retry %s: %s", reason_path, exc)
        return tickets

    # ---- worker ----------------------------------------------------------------------------

    def _run(self) -> None:
        while True:
            work = None
            with self._cond:
                while not self._stopping:
                    if self._exclusive:
                        work = ("exclusive", self._exclusive.popleft())
                        break
                    job = self.jobs.claim_next()
                    if job is not None:
                        work = ("job", job)
                        break
                    wait = self.jobs.seconds_until_next()
                    self._cond.wait(self._poll if wait is None else max(0.0, min(wait, self._poll)))
                if work is None:
                    return
                self._busy = True
            try:
                if work[0] == "exclusive":
                    self._do_exclusive(*work[1])
                else:
                    self._do_job(work[1])
            except Exception:  # the worker must outlive any single bug
                log.exception("ingest worker error")
            finally:
                with self._cond:
                    self._busy = False
                    self._cond.notify_all()

    def _do_exclusive(self, fn: Callable[[Index], object], future: Future) -> None:
        try:
            future.set_result(fn(self.index))
        except BaseException as exc:
            future.set_exception(exc)

    def _do_job(self, job: Job) -> None:
        try:
            result = self.processor.process(job)
        except Exception as exc:  # process() reports its own errors; this is the last line of defence
            log.exception("processor crashed on job %s", job.id)
            result = IngestResult(status=FAILED, job_id=job.id, files=list(job.files), error=f"unexpected error: {exc}")
        ticket = self._tickets.get(job.id)
        if result.status == DEFERRED:
            result.retry_in_seconds = self.jobs.defer(job, result.error or "transient error")
            result.attempts = job.attempts
            first = not job.deferred_notified
            if first:
                job.deferred_notified = True
                self.jobs.save(job)
            if first:
                self._deliver(ticket, result, final=False)
            return
        self.jobs.delete(job.id)
        self._deliver(ticket, result, final=True)
        with self._cond:
            self._tickets.pop(job.id, None)

    def _deliver(self, ticket: Ticket | None, result: IngestResult, final: bool) -> None:
        if ticket is not None:
            if not ticket.reported.done():
                ticket.reported.set_result(result)
            if final and not ticket.final.done():
                ticket.final.set_result(result)
        for callback in list(self._callbacks):
            try:
                callback(result)
            except Exception:
                log.exception("on_result callback failed")
