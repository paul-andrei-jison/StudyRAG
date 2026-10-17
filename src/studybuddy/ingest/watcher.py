"""The inbox: files dropped into SB_DATA_DIR/inbox/ are queued for ingest.

Layout:
    inbox/photo.jpg                  -> auto-detect the subject
    inbox/Science/photo.jpg          -> subject Science (a new folder name is added to the subject list)
    inbox/Science/trip/1.jpg 2.jpg   -> ONE album document (2+ photos in a sub-folder), pages in natural name order

`InboxScanner` decides what is ready (a file must stop changing and be openable, a folder must stop gaining files);
it is plain polling logic, tested with a fake clock. `InboxWatcher` uses watchdog only to wake the scanner early;
a periodic rescan means a missed event never matters.
"""
from __future__ import annotations

import json
import logging
import re
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Callable

from ..errors import StudyBuddyError
from .files import sniff
from .filing import FilingError, safe_move
from .service import IngestService, Ticket
from .subjects import SubjectError

log = logging.getLogger(__name__)

SETTLE_SECONDS = 3.0
RESCAN_SECONDS = 5.0
IGNORED_NAMES = {"desktop.ini", "thumbs.db", ".ds_store"}
IGNORED_SUFFIXES = {".tmp", ".temp", ".partial", ".part", ".crdownload", ".download", ".lnk", ".swp"}
ALBUM_TYPES = {"jpeg", "png", "webp", "bmp", "tiff", "heic"}
_NATURAL = re.compile(r"(\d+)")


def natural_key(path: Path) -> list:
    return [int(p) if p.isdigit() else p.lower() for p in _NATURAL.split(str(path))]


def is_ignored(path: Path) -> bool:
    name = path.name
    return (name.startswith((".", "~$")) or name.lower() in IGNORED_NAMES or path.suffix.lower() in IGNORED_SUFFIXES)


@dataclass
class Group:
    files: list[Path]
    subject: str | None
    is_album: bool
    kinds: list[str]
    received_at: datetime


@dataclass
class _Seen:
    signature: tuple          # (size, mtime_ns)
    since: float              # when we first saw exactly this signature


class InboxScanner:
    def __init__(self, inbox: Path, clock: Callable[[], float] = time.time, settle_seconds: float = SETTLE_SECONDS):
        self.inbox = inbox
        self.clock = clock
        self.settle = settle_seconds
        self._files: dict[Path, _Seen] = {}
        self._folders: dict[Path, _Seen] = {}
        self.unsettled = 0

    # ---- readiness -------------------------------------------------------------------------

    def _file_ready(self, path: Path, now: float) -> bool:
        try:
            st = path.stat()
        except OSError:
            self._files.pop(path, None)
            return False
        signature = (st.st_size, st.st_mtime_ns)
        seen = self._files.get(path)
        if seen is None or seen.signature != signature:
            self._files[path] = _Seen(signature, now)  # new or still changing: start the quiet period over
            return False
        if now - seen.since < self.settle:
            return False
        try:
            with path.open("rb") as handle:  # still being written by another program on Windows? then this fails
                handle.read(1)
        except OSError:
            return False
        return True

    def _folder_ready(self, folder: Path, files: list[Path], now: float) -> bool:
        signature = tuple(sorted((str(p.relative_to(folder)), p.stat().st_size) for p in files))
        seen = self._folders.get(folder)
        if seen is None or seen.signature != signature:
            self._folders[folder] = _Seen(signature, now)
            return False
        return now - seen.since >= self.settle

    # ---- scan ------------------------------------------------------------------------------

    def scan(self, queued: set[str] | None = None) -> list[Group]:
        """Everything ready to be queued now. `queued` holds source paths already in the job queue."""
        queued = queued or set()
        now = self.clock()
        groups: list[Group] = []
        self.unsettled = 0
        present: set[Path] = set()
        if not self.inbox.is_dir():
            return groups

        for entry in sorted(self.inbox.iterdir(), key=natural_key):
            if is_ignored(entry):
                continue
            if entry.is_file():
                present.add(entry)
                groups += self._single([entry], None, now, queued)
            elif entry.is_dir():
                groups += self._subject_folder(entry, now, queued, present)
        for gone in set(self._files) - present:  # forget files that left the inbox
            del self._files[gone]
        return groups

    def _single(self, files: list[Path], subject: str | None, now: float, queued: set[str]) -> list[Group]:
        out = []
        for path in files:
            if str(path.resolve()) in queued:
                continue
            if not self._file_ready(path, now):
                self.unsettled += 1
                continue
            out.append(Group([path], subject, False, [self._kind(path)], self._arrival(path)))
        return out

    def _subject_folder(self, folder: Path, now: float, queued: set[str], present: set[Path]) -> list[Group]:
        groups: list[Group] = []
        loose = [p for p in sorted(folder.iterdir(), key=natural_key) if p.is_file() and not is_ignored(p)]
        for p in loose:
            present.add(p)
        groups += self._single(loose, folder.name, now, queued)
        for album in sorted((d for d in folder.iterdir() if d.is_dir() and not is_ignored(d)), key=natural_key):
            files = sorted((p for p in album.rglob("*") if p.is_file() and not is_ignored(p)), key=natural_key)
            if not files:
                continue
            present.update(files)
            images = [p for p in files if self._is_album_image(p)]
            others = [p for p in files if p not in images]
            groups += self._single(others, folder.name, now, queued)  # PDFs and odd files are separate documents
            if len(images) == 1:
                groups += self._single(images, folder.name, now, queued)
            elif len(images) > 1:
                groups += self._album(album, folder.name, images, now, queued)
        return groups

    def _album(self, folder: Path, subject: str, images: list[Path], now: float, queued: set[str]) -> list[Group]:
        if any(str(p.resolve()) in queued for p in images):
            return []
        ready = [self._file_ready(p, now) for p in images]  # evaluate everything, so every quiet period advances
        folder_quiet = self._folder_ready(folder, images, now)
        if not (all(ready) and folder_quiet):
            self.unsettled += 1
            return []
        return [Group(images, subject, True, ["photo-file"] * len(images), min(self._arrival(p) for p in images))]

    # ---- helpers ---------------------------------------------------------------------------

    @staticmethod
    def _head(path: Path) -> bytes:
        try:
            with path.open("rb") as handle:
                return handle.read(16)
        except OSError:
            return b""

    def _kind(self, path: Path) -> str:
        detected = sniff(self._head(path))
        return "pdf" if detected == "pdf" else ("photo-file" if detected in ALBUM_TYPES else "document")

    def _is_album_image(self, path: Path) -> bool:
        return sniff(self._head(path)) in ALBUM_TYPES

    def _arrival(self, path: Path) -> datetime:
        """When it arrived: the first time we saw the file in this state (a copied photo keeps its old mtime)."""
        seen = self._files.get(path)
        return datetime.fromtimestamp(seen.since if seen else self.clock())


class InboxWatcher:
    def __init__(self, service: IngestService, inbox: Path | None = None, scanner: InboxScanner | None = None,
                 rescan_seconds: float = RESCAN_SECONDS, use_watchdog: bool = True,
                 on_ticket: Callable[[Ticket, Group], None] | None = None):
        self.service = service
        self.inbox = inbox or service.inbox_dir
        self.scanner = scanner or InboxScanner(self.inbox)
        self.rescan = rescan_seconds
        self.use_watchdog = use_watchdog
        self.on_ticket = on_ticket
        self._wake = threading.Event()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._observer = None

    def scan_once(self) -> list[Ticket]:
        """Queue every ready group. A group whose subject folder name is unusable is moved to failed/ with the reason."""
        tickets: list[Ticket] = []
        for group in self.scanner.scan(self.service.jobs.pending_paths()):
            try:
                ticket = self.service.submit(group.files, subject=group.subject, source_kind=group.kinds,
                                             received_at=group.received_at, origin="inbox", is_album=group.is_album)
            except SubjectError as exc:
                self._quarantine(group, str(exc))
                continue
            except StudyBuddyError as exc:
                log.warning("could not queue %s: %s", group.files[0], exc)
                continue
            tickets.append(ticket)
            if self.on_ticket:
                self.on_ticket(ticket, group)
        return tickets

    def _quarantine(self, group: Group, error: str) -> None:
        dest = self.service.settings.data_dir / "failed" / f"inbox-{datetime.now():%Y%m%dT%H%M%S}-{group.files[0].stem[:20]}"
        moved: list[str] = []
        for path in group.files:
            try:
                moved.append(str(safe_move(path, dest)))
            except FilingError as exc:
                log.warning("%s", exc)
        dest.mkdir(parents=True, exist_ok=True)
        (dest / "reason.json").write_text(json.dumps({
            "error": f"{error} (the folder name {group.subject!r} cannot be used as a subject; rename the folder)",
            "kind": "permanent", "subject": None, "origin": "inbox", "files": moved,
            "original_files": [str(p) for p in group.files], "is_album": group.is_album,
            "source_kinds": group.kinds, "received_at": group.received_at.isoformat(timespec="seconds"),
            "cleanup_sources": True, "failed_at": datetime.now().isoformat(timespec="seconds"),
        }, indent=2), encoding="utf-8")

    # ---- thread ----------------------------------------------------------------------------

    def start(self) -> "InboxWatcher":
        self.inbox.mkdir(parents=True, exist_ok=True)
        if self.use_watchdog:
            self._start_observer()
        self._stop.clear()
        self._thread = threading.Thread(target=self._loop, name="inbox-watcher", daemon=True)
        self._thread.start()
        return self

    def stop(self) -> None:
        self._stop.set()
        self._wake.set()
        if self._observer is not None:
            try:
                self._observer.stop()
                self._observer.join(5)
            except Exception:  # pragma: no cover - shutting down anyway
                pass
        if self._thread:
            self._thread.join(10)

    def poke(self) -> None:
        self._wake.set()

    def _start_observer(self) -> None:
        try:
            from watchdog.events import FileSystemEventHandler
            from watchdog.observers import Observer

            watcher = self

            class Handler(FileSystemEventHandler):
                def on_any_event(self, event):  # created, modified, moved: all just mean "look again"
                    watcher.poke()

            self._observer = Observer()
            self._observer.schedule(Handler(), str(self.inbox), recursive=True)
            self._observer.start()
        except Exception as exc:  # e.g. a network drive: polling alone still works
            log.warning("file watching unavailable (%s); polling the inbox every %.0f s instead", exc, self.rescan)
            self._observer = None

    def _loop(self) -> None:
        while not self._stop.is_set():
            try:
                self.scan_once()
            except Exception:
                log.exception("inbox scan failed")
            # while a file is still settling, look again soon; otherwise the periodic rescan is enough
            delay = min(self.rescan, max(0.05, self.scanner.settle / 2)) if self.scanner.unsettled else self.rescan
            self._wake.wait(delay)
            self._wake.clear()
