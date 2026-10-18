"""Windows-safe, readable names for filed documents.

Target shape: 2026-10-17_1425_Science_photosynthesis-worksheet.jpg  (received time, subject, topic slug).
"""
from __future__ import annotations

import os
import re
import unicodedata
from datetime import datetime
from pathlib import Path

from ..errors import StudyBuddyError

MAX_SUBJECT_CHARS = 40
MAX_SLUG_CHARS = 50
MAX_PATH_CHARS = 240      # stay clear of the 260-character Windows limit
FALLBACK = "untitled"

# Device names Windows refuses as file names, with or without an extension.
WINDOWS_RESERVED = frozenset(
    {"con", "prn", "aux", "nul"} | {f"com{i}" for i in range(1, 10)} | {f"lpt{i}" for i in range(1, 10)}
)
_INVALID = re.compile(r'[<>:"/\\|?*\x00-\x1f]')
_SPACES = re.compile(r"\s+")


class NamingError(StudyBuddyError):
    """No valid library name can be made (for example the folder path alone is too long)."""


def is_windows_reserved(name: str) -> bool:
    return name.split(".", 1)[0].strip().lower() in WINDOWS_RESERVED


def sanitize_component(text: str, max_len: int, fallback: str = FALLBACK) -> str:
    """One path component: no invalid characters, no trailing dot/space, not a reserved device name."""
    cleaned = _SPACES.sub(" ", _INVALID.sub(" ", unicodedata.normalize("NFC", text or ""))).strip()
    cleaned = cleaned[:max_len].rstrip(" .")
    cleaned = cleaned.lstrip(" ")
    if not cleaned.strip(". "):
        return fallback
    if is_windows_reserved(cleaned):
        cleaned = f"_{cleaned}"
    return cleaned


def slugify(text: str, max_len: int = MAX_SLUG_CHARS) -> str:
    """Lower-case ASCII words joined by hyphens, cut at a word boundary where possible."""
    folded = unicodedata.normalize("NFKD", text or "").encode("ascii", "ignore").decode("ascii").lower()
    folded = folded.replace("'", "").replace("`", "")  # "Earth's" is "earths", not "earth-s" (curly ones fold away above)
    slug = re.sub(r"[^a-z0-9]+", "-", folded).strip("-")
    if len(slug) > max_len:
        cut = slug[:max_len]
        slug = cut[: cut.rfind("-")] if "-" in cut[max_len // 2:] else cut
        slug = slug.strip("-")
    return slug or FALLBACK


def subject_token(subject: str) -> str:
    """The subject as it appears inside a filename (spaces become hyphens)."""
    return sanitize_component(subject, MAX_SUBJECT_CHARS).replace(" ", "-")


def subject_folder(subject: str) -> str:
    return sanitize_component(subject, MAX_SUBJECT_CHARS)


def library_stem(received_at: datetime, subject: str, title_or_slug: str) -> str:
    """`2026-10-17_1425_Science_photosynthesis-worksheet` (no extension)."""
    return f"{received_at:%Y-%m-%d_%H%M}_{subject_token(subject)}_{slugify(title_or_slug)}"


def fit_name(directory: Path, stem: str, ext: str, max_total: int = MAX_PATH_CHARS) -> str:
    """`stem + ext`, shortened from the end of the stem if the full path would exceed `max_total`."""
    room = max_total - len(str(directory)) - 1 - len(ext) - 4  # 4 spare for a "_99" collision suffix
    if room < 12:
        raise NamingError(f"The library folder path is too long to file documents in: {directory}")
    if len(stem) > room:
        stem = stem[:room].rstrip("-_ .")
    return f"{stem}{ext}"


def _numbered(stem: str, ext: str, n: int) -> str:
    return f"{stem}{ext}" if n == 1 else f"{stem}_{n}{ext}"


def claim_unique_file(directory: Path, stem: str, ext: str) -> Path:
    """Atomically create an empty file with a name nobody else has (`name.jpg`, `name_2.jpg`, ...).

    Exclusive creation makes the name safe against another process; the caller replaces it with the real content.
    """
    directory.mkdir(parents=True, exist_ok=True)
    n = 1
    while True:
        path = directory / fit_name(directory, _numbered(stem, "", n), ext)
        try:
            fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        except FileExistsError:
            n += 1
            continue
        os.close(fd)
        return path


def claim_unique_dir(parent: Path, name: str) -> Path:
    """Atomically create a new folder (`name`, `name_2`, ...)."""
    parent.mkdir(parents=True, exist_ok=True)
    n = 1
    while True:
        path = parent / fit_name(parent, _numbered(name, "", n), "")
        try:
            path.mkdir()
        except FileExistsError:
            n += 1
            continue
        return path
