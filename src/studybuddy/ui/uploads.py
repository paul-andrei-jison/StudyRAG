"""The logic behind the Upload tab: copy what was dropped to a safe place, then hand it to `IngestService.submit()`.

The Telegram bot uses the same `submit()`. Uploads are copied into `SB_DATA_DIR/uploads/` first: Gradio's own temporary
copy can disappear when the app restarts, while a queued or deferred job still needs its files.
"""
from __future__ import annotations

import logging
import shutil
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Sequence

from ..errors import PermanentIngestError, StudyBuddyError
from ..ingest.files import FileInfo, classify
from ..ingest.naming import sanitize_component
from ..ingest.service import IngestService, Ticket
from ..ingest.subjects import clean_subject_name
from ..ingest.watcher import ALBUM_TYPES, natural_key

log = logging.getLogger(__name__)

AUTO_DETECT = "Let Study Buddy decide"
EMPTY_DIR_GRACE_SECONDS = 60


@dataclass
class UploadOutcome:
    ok: bool
    message: str
    queued: int = 0                      # documents queued (an album counts once)
    skipped: list[str] = field(default_factory=list)
    tickets: list[Ticket] = field(default_factory=list)


def uploads_dir(service: IngestService) -> Path:
    return service.settings.data_dir / "uploads"


def sweep_empty_uploads(directory: Path, grace: float = EMPTY_DIR_GRACE_SECONDS, now: float | None = None) -> int:
    """Remove upload folders that processing has emptied (a folder younger than `grace` may still be filling up)."""
    removed = 0
    now = time.time() if now is None else now
    if not directory.is_dir():
        return 0
    for folder in directory.iterdir():
        try:
            if folder.is_dir() and not any(folder.iterdir()) and now - folder.stat().st_mtime >= grace:
                folder.rmdir()
                removed += 1
        except OSError:
            continue
    return removed


def _subject_choice(service: IngestService, choice: str | None) -> str | None:
    """None for auto-detect; otherwise the canonical subject (a new name is validated now and added when a document is filed)."""
    if not choice or not choice.strip() or choice.strip().casefold() == AUTO_DETECT.casefold():
        return None
    return service.subjects.find(choice) or clean_subject_name(choice)


def submit_uploads(service: IngestService, files: Sequence[str | Path] | None, subject: str | None = None,
                   album: bool = False) -> UploadOutcome:
    """Queue what the student dropped. Never waits for the documents to be read."""
    paths = [Path(f) for f in (files or [])]
    if not paths:
        return UploadOutcome(False, "Choose at least one photo or a PDF first.")
    try:
        canonical = _subject_choice(service, subject)
    except StudyBuddyError as exc:
        return UploadOutcome(False, str(exc))

    good: list[FileInfo] = []
    skipped: list[str] = []
    for path in paths:
        try:
            good.append(classify(path))          # judged by the file's bytes, like the worker does
        except PermanentIngestError as exc:       # its message already starts with the file name
            skipped.append(str(exc))
        except OSError as exc:
            skipped.append(f"{path.name}: cannot be read ({exc})")
    if album:
        if skipped:
            return UploadOutcome(False, "Nothing was added, because " + "; ".join(skipped), skipped=skipped)
        if len(good) < 2:
            return UploadOutcome(False, "Pages that belong together need at least two photos. Turn that switch off to add a single photo.")
        if any(info.kind not in ALBUM_TYPES for info in good):
            return UploadOutcome(False, "Pages that belong together must all be photos. Add the PDF on its own.")
        good.sort(key=lambda info: natural_key(Path(info.path.name)))   # same page order as the inbox: img2 before img10
    if not good:
        return UploadOutcome(False, "Nothing was added: " + "; ".join(skipped), skipped=skipped)

    sweep_empty_uploads(uploads_dir(service))
    groups = [good] if album else [[item] for item in good]
    tickets: list[Ticket] = []
    for group in groups:
        try:
            tickets.append(_queue_group(service, group, canonical, album))
        except StudyBuddyError as exc:
            skipped.append(str(exc))
        except OSError as exc:
            skipped.append(f"{group[0].path.name}: could not be copied ({exc})")
    if not tickets:
        return UploadOutcome(False, "Nothing was added: " + "; ".join(skipped), skipped=skipped)
    if album:
        message = f"Added 1 paper made of {len(good)} pages. It is being read now; follow it in the corner of the screen."
    elif len(tickets) == 1:
        message = "Added 1 paper. It is being read now; follow it in the corner of the screen."
    else:
        message = f"Added {len(tickets)} papers. They are being read now; follow them in the corner of the screen."
    if skipped:
        message += " Not added: " + "; ".join(skipped)
    return UploadOutcome(True, message, queued=len(tickets), skipped=skipped, tickets=tickets)


def _queue_group(service: IngestService, group: list[FileInfo], subject: str | None, album: bool) -> Ticket:
    folder = uploads_dir(service) / f"{datetime.now():%Y%m%d-%H%M%S}-{uuid.uuid4().hex[:6]}"
    folder.mkdir(parents=True, exist_ok=True)
    copies: list[Path] = []
    kinds: list[str] = []
    try:
        for number, info in enumerate(group, 1):
            stem = sanitize_component(info.path.stem, 60, fallback="upload")
            target = folder / f"{number:02d}-{stem}{info.ext}"      # the right extension for the real type
            shutil.copyfile(info.path, target)
            copies.append(target)
            kinds.append("pdf" if info.is_pdf else "photo-file")
        return service.submit(copies, subject=subject, source_kind=kinds, origin="ui", is_album=album, cleanup_sources=True)
    except BaseException:
        shutil.rmtree(folder, ignore_errors=True)   # nothing was queued: leave nothing behind
        raise
