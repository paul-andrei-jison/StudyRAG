"""Persistent job queue: one JSON file per job in SB_DATA_DIR/queue/.

A job is written to disk before it is queued and deleted only after a final outcome, so a crash, a restart or
LM Studio being off for an hour loses nothing. Transient failures keep the job with a retry time.
"""
from __future__ import annotations

import json
import os
import threading
import time
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

BACKOFF_SECONDS = (30, 60, 120, 300, 900)  # then 15 minutes for every further attempt


def backoff_seconds(attempts: int) -> int:
    """Delay before retry number `attempts` (1-based)."""
    return BACKOFF_SECONDS[min(max(attempts, 1), len(BACKOFF_SECONDS)) - 1]


@dataclass
class Job:
    id: str
    files: list[str]                        # the sources, exactly as received
    kinds: list[str]                        # per file: photo | photo-file | pdf | document
    received_at: str                        # ISO local time the document arrived (used in filenames)
    origin: str = "api"                     # inbox | bot | api
    subject: str | None = None              # None = auto-detect
    is_album: bool = False                  # several photos that form ONE document
    cleanup_sources: bool = True            # remove the sources after the verified copy is filed
    attempts: int = 0
    state: str = "pending"                  # pending | running
    created_at: float = 0.0
    next_retry_at: float = 0.0
    last_error: str | None = None
    deferred_notified: bool = False         # the first deferral is reported once
    staged: list[str] = field(default_factory=list)  # library files this job created (cleaned up on retry)


class JobStore:
    def __init__(self, queue_dir: Path, clock: Callable[[], float] = time.time):
        self.dir = queue_dir
        self.clock = clock
        self._lock = threading.RLock()
        self._seq = 0
        queue_dir.mkdir(parents=True, exist_ok=True)

    # ---- files -----------------------------------------------------------------------------

    def _path(self, job_id: str) -> Path:
        return self.dir / f"{job_id}.json"

    def save(self, job: Job) -> None:
        with self._lock:
            tmp = self._path(job.id).with_suffix(".json.tmp")
            tmp.write_text(json.dumps(asdict(job), indent=2), encoding="utf-8")
            os.replace(tmp, self._path(job.id))  # atomic

    def delete(self, job_id: str) -> None:
        with self._lock:
            self._path(job_id).unlink(missing_ok=True)

    def load_all(self) -> list[Job]:
        """Every job on disk, oldest first. Unreadable files are set aside in queue/corrupt/, never fatal."""
        jobs: list[Job] = []
        with self._lock:
            for path in sorted(self.dir.glob("*.json")):
                try:
                    jobs.append(Job(**json.loads(path.read_text(encoding="utf-8"))))
                except (OSError, ValueError, TypeError):
                    corrupt = self.dir / "corrupt"
                    corrupt.mkdir(exist_ok=True)
                    os.replace(path, corrupt / path.name)
        return sorted(jobs, key=lambda j: (j.created_at, j.id))

    # ---- lifecycle -------------------------------------------------------------------------

    def create(self, files: list[str], kinds: list[str], received_at: datetime | str | None = None,
               origin: str = "api", subject: str | None = None, is_album: bool = False,
               cleanup_sources: bool = True) -> Job:
        if not files or len(files) != len(kinds):
            raise ValueError("a job needs files and one kind per file")
        when = received_at or datetime.now()
        with self._lock:
            self._seq += 1
            seq = self._seq
        job = Job(
            # time, then a per-process sequence: jobs created in the same second still sort in creation order
            id=f"{datetime.now(timezone.utc):%Y%m%dT%H%M%S}-{seq:06d}-{uuid.uuid4().hex[:4]}",
            files=[str(Path(f).resolve()) for f in files], kinds=list(kinds),
            received_at=when if isinstance(when, str) else when.isoformat(timespec="seconds"),
            origin=origin, subject=subject, is_album=is_album, cleanup_sources=cleanup_sources,
            created_at=self.clock(),
        )
        self.save(job)
        return job

    def recover(self) -> list[Job]:
        """After a crash: jobs left `running` go back to `pending`. Returns every job on disk."""
        with self._lock:
            jobs = self.load_all()
            for job in jobs:
                if job.state == "running":
                    job.state = "pending"
                    self.save(job)
            return jobs

    def claim_next(self) -> Job | None:
        """The oldest pending job whose retry time has come, marked `running`."""
        with self._lock:
            now = self.clock()
            for job in self.load_all():
                if job.state == "pending" and job.next_retry_at <= now:
                    job.state = "running"
                    self.save(job)
                    return job
        return None

    def seconds_until_next(self) -> float | None:
        """How long until a pending job becomes ready (0 if one is ready now, None if none are pending)."""
        with self._lock:
            now = self.clock()
            waits = [max(0.0, j.next_retry_at - now) for j in self.load_all() if j.state == "pending"]
        return min(waits) if waits else None

    def defer(self, job: Job, error: str) -> float:
        """A transient failure: keep the job, count the attempt, schedule the retry. Returns the delay."""
        with self._lock:
            job.attempts += 1
            job.state = "pending"
            job.last_error = error
            delay = backoff_seconds(job.attempts)
            job.next_retry_at = self.clock() + delay
            self.save(job)
        return delay

    def pending_paths(self) -> set[str]:
        """Source paths already queued (so the inbox scanner never queues a file twice)."""
        with self._lock:
            return {f for job in self.load_all() for f in job.files}

    def counts(self) -> dict[str, int]:
        jobs = self.load_all()
        return {"queued": sum(1 for j in jobs if j.state == "pending" and j.attempts == 0),
                "deferred": sum(1 for j in jobs if j.state == "pending" and j.attempts > 0),
                "running": sum(1 for j in jobs if j.state == "running")}
