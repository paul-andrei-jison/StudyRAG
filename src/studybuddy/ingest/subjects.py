"""The subject list: folder names under the library. Auto-detect picks from it; typed subjects are added to it.

The repo's `subjects.json` is only the default template. If SB_SUBJECTS_FILE points inside the repo, the live,
editable list is `SB_DATA_DIR/subjects.json`, seeded from the template, so adding a subject never dirties the repo.
"""
from __future__ import annotations

import json
import os
import re
import threading
from pathlib import Path

from ..config import REPO_ROOT, Settings, _is_within
from ..errors import StudyBuddyError
from .naming import MAX_SUBJECT_CHARS, WINDOWS_RESERVED, sanitize_component

UNSORTED = "Unsorted"        # fallback subject: the library folder for documents whose subject is unknown
SPOOL_FOLDER = "_Unsorted"   # the bot's arrival spool (chunk 7); never a subject
RESERVED_SUBJECTS = frozenset({UNSORTED.casefold(), SPOOL_FOLDER.casefold(), "failed", "duplicates", "inbox", "queue"})
_ANSWER_PREFIX = re.compile(r"^\s*(?:the\s+)?(?:subject|category|answer)\s*(?:is|:|-)\s*", re.IGNORECASE)
_NOT_NAME = re.compile(r"[^\w\s&'-]", re.UNICODE)


class SubjectError(StudyBuddyError):
    """A subject name or the subject list file is not usable."""


def clean_subject_name(raw: str) -> str:
    """Validate and tidy a typed subject. Raises SubjectError rather than guessing."""
    name = sanitize_component(raw or "", 10_000, fallback="")
    if not name:
        raise SubjectError("A subject name cannot be empty.")
    if len(name) > MAX_SUBJECT_CHARS:
        raise SubjectError(f"Subject names are limited to {MAX_SUBJECT_CHARS} characters: {name[:20]!r}...")
    if name.casefold() in RESERVED_SUBJECTS or name.split(".", 1)[0].strip("_").casefold() in WINDOWS_RESERVED:
        raise SubjectError(f"{name!r} is reserved and cannot be used as a subject.")
    return name


class SubjectList:
    def __init__(self, path: Path, defaults: list[str] | None = None):
        self.path = path
        self._lock = threading.RLock()
        if not path.exists():
            self._write([clean_subject_name(n) for n in (defaults or [])])
        self._names = self._read()

    # ---- file ------------------------------------------------------------------------------

    def _read(self) -> list[str]:
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            raise SubjectError(f"{self.path} is not valid JSON ({exc}). Fix or delete it.") from exc
        if not isinstance(data, list) or not all(isinstance(n, str) and n.strip() for n in data):
            raise SubjectError(f"{self.path} must be a JSON list of subject names.")
        names: list[str] = []
        for raw in data:
            name = clean_subject_name(raw)
            if name.casefold() not in {n.casefold() for n in names}:
                names.append(name)
        return names

    def _write(self, names: list[str]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_name(self.path.name + ".tmp")
        tmp.write_text(json.dumps(names, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        os.replace(tmp, self.path)  # atomic: a crash never leaves a half-written list

    def reload(self) -> None:
        with self._lock:
            self._names = self._read()

    # ---- use -------------------------------------------------------------------------------

    def names(self) -> list[str]:
        with self._lock:
            return list(self._names)

    def choices(self) -> list[str]:
        """What auto-detect may answer: the list, then the fallback."""
        return self.names() + [UNSORTED]

    def find(self, name: str) -> str | None:
        """The canonical spelling of `name` (case-insensitive), or None. `Unsorted` is always known."""
        key = (name or "").strip().casefold()
        if key == UNSORTED.casefold():
            return UNSORTED
        with self._lock:
            return next((n for n in self._names if n.casefold() == key), None)

    def add(self, raw: str) -> str:
        """Add a typed subject (or return the existing spelling). Never used by auto-detect."""
        name = clean_subject_name(raw)
        with self._lock:
            self.reload()  # pick up edits made by hand since we last looked
            existing = self.find(name)
            if existing:
                return existing
            self._names.append(name)
            self._write(self._names)
            return name

    def resolve(self, raw: str, add_if_new: bool = False) -> str:
        """Canonical subject for `raw`; unknown names are added only when `add_if_new`."""
        known = self.find(raw)
        if known:
            return known
        if add_if_new:
            return self.add(raw)
        raise SubjectError(f"Unknown subject {raw!r}. Known: {', '.join(self.names())}")

    def match_model_answer(self, answer: str) -> str:
        """Map a model's free-text answer onto the list; anything that is not exactly one entry is Unsorted."""
        text = _ANSWER_PREFIX.sub("", (answer or "").strip().splitlines()[0] if (answer or "").strip() else "")
        text = _NOT_NAME.sub("", text).strip()
        return self.find(text) or UNSORTED

    @classmethod
    def from_settings(cls, settings: Settings, repo_root: Path = REPO_ROOT) -> "SubjectList":
        template = settings.subjects_file
        defaults: list[str] = []
        if template.exists():
            try:
                defaults = [str(n) for n in json.loads(template.read_text(encoding="utf-8"))]
            except (OSError, ValueError, TypeError) as exc:
                raise SubjectError(f"{template} is not a valid subject list ({exc}).") from exc
        if _is_within(template.resolve(), repo_root.resolve()):
            return cls(settings.data_dir / "subjects.json", defaults)
        return cls(template, defaults)
