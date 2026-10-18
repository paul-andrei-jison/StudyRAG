"""Back up the library and the index to a folder you choose, and check a backup later.

  python scripts/backup.py D:\\Backups              make D:\\Backups\\studybuddy-backup-YYYYMMDD-HHMMSS
  python scripts/backup.py --verify <backup folder>   re-read every file and compare it with the manifest

What is copied: the whole library (your photos and PDFs), and from the data folder the index (SQLite through its backup API, so the
copy is consistent even while the app runs), the vectors, the embedding record, the Telegram bot's pending state, the job queue,
uploads, the inbox and duplicates. Not copied: logs, samples, evaluation files, older backups, and `.env` (it holds the bot token:
keep your own copy of it somewhere private). Nothing is ever overwritten: every backup is a new folder, built under a temporary
name and renamed only after every file was read back and matched.
"""
from __future__ import annotations

import hashlib
import json
import shutil
import sqlite3
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

from .config import REPO_ROOT, Settings, _is_within
from .errors import StudyBuddyError
from .index.index import DB_NAME, INDEX_DIRNAME

MANIFEST = "MANIFEST.json"
RESTORE_NOTE = "RESTORE.txt"
PREFIX = "studybuddy-backup-"
EXCLUDED_DATA = {"logs", "samples", "eval", "backup", "backups", "relocations"}
EXCLUDED_NAMES = {"vectors.new", "vectors.old"}          # leftovers of an interrupted re-index
SQLITE_SIDE_FILES = (DB_NAME + "-wal", DB_NAME + "-shm", DB_NAME + "-journal")


class BackupError(StudyBuddyError):
    pass


@dataclass
class BackupResult:
    folder: Path
    files: int
    bytes: int
    skipped: list[str] = field(default_factory=list)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def check_destination(settings: Settings, destination: Path, repo_root: Path = REPO_ROOT) -> Path:
    destination = destination.expanduser().resolve()
    for label, protected in (("the library", settings.library_dir), ("the data folder", settings.data_dir), ("the project folder", repo_root.resolve())):
        if _is_within(destination, protected) or _is_within(protected, destination):
            raise BackupError(f"The backup folder ({destination}) must not be inside or contain {label} ({protected}). Choose a different folder, ideally on another drive.")
    return destination


def _wanted_data_files(data_dir: Path):
    for path in sorted(data_dir.rglob("*")):
        relative = path.relative_to(data_dir)
        if not path.is_file() or relative.parts[0] in EXCLUDED_DATA or any(p in EXCLUDED_NAMES for p in relative.parts):
            continue
        if path.name.endswith(".partial") or (relative.parts[:2] == (INDEX_DIRNAME, DB_NAME)) or path.name in SQLITE_SIDE_FILES:
            continue                                      # the database is copied separately, below
        yield path, relative


def _backup_database(source: Path, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    src = sqlite3.connect(f"file:{source.as_posix()}?mode=ro", uri=True)
    try:
        dst = sqlite3.connect(target)
        try:
            src.backup(dst)
        finally:
            dst.close()
    finally:
        src.close()


def make_backup(settings: Settings, destination: Path, *, now: Callable[[], float] = time.time, repo_root: Path = REPO_ROOT) -> BackupResult:
    destination = check_destination(settings, destination, repo_root)
    if not settings.library_dir.is_dir():
        raise BackupError(f"The library folder ({settings.library_dir}) does not exist, so there is nothing to back up.")
    stamp = time.strftime("%Y%m%d-%H%M%S", time.localtime(now()))
    final = destination / f"{PREFIX}{stamp}"
    work = destination / f"{PREFIX}{stamp}.partial"
    if final.exists() or work.exists():
        raise BackupError(f"{final.name} already exists; wait a second and run again. Nothing was changed.")
    destination.mkdir(parents=True, exist_ok=True)
    work.mkdir()
    entries: dict[str, dict] = {}
    skipped: list[str] = []
    try:
        pairs: list[tuple[Path, str]] = []
        for path in sorted(settings.library_dir.rglob("*")):
            if path.is_file() and not path.name.endswith(".partial"):
                pairs.append((path, "library/" + path.relative_to(settings.library_dir).as_posix()))
        for path, relative in _wanted_data_files(settings.data_dir):
            pairs.append((path, "data/" + relative.as_posix()))
        for source, name in pairs:
            target = work / name
            target.parent.mkdir(parents=True, exist_ok=True)
            try:
                shutil.copy2(source, target)
            except OSError as exc:                      # a file that vanished or is locked: say so, do not hide it
                skipped.append(f"{name}: {exc}")
                continue
        database = settings.data_dir / INDEX_DIRNAME / DB_NAME
        if database.exists():
            _backup_database(database, work / "data" / INDEX_DIRNAME / DB_NAME)
        total = 0
        for path in sorted(p for p in work.rglob("*") if p.is_file()):
            name = path.relative_to(work).as_posix()
            entries[name] = {"sha256": _sha256(path), "size": path.stat().st_size}
            total += entries[name]["size"]
        manifest = {"format": 1, "created": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(now())), "files": entries}
        (work / MANIFEST).write_text(json.dumps(manifest, indent=1), encoding="utf-8")
        (work / RESTORE_NOTE).write_text(_restore_note(), encoding="utf-8")
        problems = verify_backup(work)
        if problems:
            raise BackupError("The copy did not match the original, so it was thrown away: " + "; ".join(problems[:3]))
        work.rename(final)
    except BaseException:
        shutil.rmtree(work, ignore_errors=True)
        raise
    return BackupResult(final, len(entries), total, skipped)


def verify_backup(folder: Path) -> list[str]:
    """Problems found (empty list = every file is present and unchanged)."""
    manifest_path = folder / MANIFEST
    if not manifest_path.exists():
        return [f"{MANIFEST} is missing: this is not a Study Buddy backup"]
    try:
        files = json.loads(manifest_path.read_text(encoding="utf-8"))["files"]
    except (ValueError, KeyError) as exc:
        return [f"{MANIFEST} cannot be read ({exc})"]
    problems = []
    for name, expected in files.items():
        path = folder / name
        if not path.is_file():
            problems.append(f"missing: {name}")
        elif path.stat().st_size != expected["size"] or _sha256(path) != expected["sha256"]:
            problems.append(f"changed: {name}")
    known = set(files) | {MANIFEST, RESTORE_NOTE}
    for path in folder.rglob("*"):
        if path.is_file() and path.relative_to(folder).as_posix() not in known:
            problems.append(f"unexpected file: {path.relative_to(folder).as_posix()}")
    return problems


def _restore_note() -> str:
    return (
        "How to restore this backup (see docs/backup.md in the project for the full version)\n\n"
        "1. Close Study Buddy (and check no python window is still running it).\n"
        "2. Run:  python scripts/backup.py --verify <this folder>   - it must say the backup is intact.\n"
        "3. Keep the current folders: rename your library folder and your data folder (add -old to the names).\n"
        "4. Copy the 'library' folder here to the place named by SB_LIBRARY_DIR in .env, and the 'data' folder to SB_DATA_DIR.\n"
        "5. Run:  python -m studybuddy --check   - then start the app. If searches look wrong, run:\n"
        "   python scripts/index_admin.py reindex   (rebuilds the search data from the text already stored).\n"
        "The .env file (bot token) is NOT in the backup; keep your own private copy.\n"
    )
