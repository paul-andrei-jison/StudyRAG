"""The logic behind the status panel: what is queued, deferred and failed, the latest results, and retry-failed."""
from __future__ import annotations

import json
import threading
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path

from ..ingest.cli import format_result
from ..ingest.results import IngestResult
from ..ingest.service import IngestService

RECENT_RESULTS = 15


@dataclass
class StatusSnapshot:
    queued: list[str] = field(default_factory=list)      # waiting for the worker
    running: list[str] = field(default_factory=list)
    deferred: list[str] = field(default_factory=list)    # LM Studio was unavailable: waiting for a retry
    failed: list[str] = field(default_factory=list)      # kept in failed/ with the reason
    recent: list[str] = field(default_factory=list)      # newest first, one block per document
    failed_names: list[str] = field(default_factory=list)     # file names only (the reasons are in `failed`)
    deferred_names: list[str] = field(default_factory=list)

    def pill(self) -> str:
        """The short text for the corner indicator; empty when there is nothing to say."""
        if self.deferred_names:
            return "Waiting for LM Studio"
        if self.failed_names:
            n = len(self.failed_names)
            return f"{n} need{'s' if n == 1 else ''} a look"
        busy = len(self.running) + len(self.queued)
        return f"{busy} being read" if busy else ""

    def retry_possible(self) -> bool:
        return bool(self.failed_names)

    def headline(self) -> str:
        parts = [f"{len(self.running)} reading", f"{len(self.queued)} queued", f"{len(self.deferred)} waiting to retry",
                 f"{len(self.failed)} failed"]
        return "Status: " + ", ".join(parts)

    def jobs_text(self) -> str:
        lines: list[str] = []
        for title, items in (("Reading now", self.running), ("Queued", self.queued),
                             ("Waiting to retry (LM Studio was not available)", self.deferred), ("Failed", self.failed)):
            if items:
                lines += [f"{title}:"] + [f"  {item}" for item in items] + [""]
        return "\n".join(lines).strip() or "Nothing is waiting."

    def recent_text(self) -> str:
        return "\n\n".join(self.recent) or "No results yet in this session."


def _names(files: list[str]) -> str:
    return ", ".join(Path(f).name for f in files) or "(no files)"


class StatusBoard:
    def __init__(self, service: IngestService, keep: int = RECENT_RESULTS):
        self.service = service
        self._recent: deque[str] = deque(maxlen=keep)
        self._lock = threading.Lock()
        service.add_callback(self._on_result)

    def _on_result(self, result: IngestResult) -> None:
        with self._lock:
            self._recent.appendleft(format_result(result))

    def failed_reasons(self) -> list[tuple[str, str]]:
        """(folder name, text) for every document in failed/ (an unreadable reason.json is still listed)."""
        root = self.service.settings.data_dir / "failed"
        out = []
        for path in sorted(root.glob("*/reason.json")) if root.is_dir() else []:
            try:
                info = json.loads(path.read_text(encoding="utf-8"))
                out.append((path.parent.name, f"{_names(info.get('files', []))} - {info.get('error') or 'unknown error'}"))
            except (OSError, ValueError):
                out.append((path.parent.name, "(the reason file is unreadable; the files are still in that folder)"))
        return out

    def snapshot(self) -> StatusSnapshot:
        jobs = self.service.jobs
        snap = StatusSnapshot()
        now = jobs.clock()
        for job in jobs.load_all():
            names = _names(job.files)
            if job.state == "running":
                snap.running.append(names)
            elif job.attempts > 0:
                wait = max(0, int(job.next_retry_at - now))
                snap.deferred.append(f"{names} - {job.last_error or 'LM Studio unavailable'} (retry in {wait} s, attempt {job.attempts})")
                snap.deferred_names.append(names)
            else:
                snap.queued.append(names)
        snap.failed = [text for _, text in self.failed_reasons()]
        snap.failed_names = [text.split(" - ", 1)[0] for text in snap.failed]
        with self._lock:
            snap.recent = list(self._recent)
        return snap

    def retry_failed(self) -> str:
        before = len(self.failed_reasons())
        if not before:
            return "Nothing needs a look."
        tickets = self.service.retry_failed()
        if not tickets:
            return "These could not be tried again because their files are missing. Add them again on the Add screen."
        if len(tickets) < before:
            return f"Trying {len(tickets)} of {before} again. The rest are missing their files; add them again on the Add screen."
        return f"Trying {before} again." if before > 1 else "Trying it again."
