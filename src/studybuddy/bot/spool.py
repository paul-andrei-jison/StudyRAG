"""The arrival spool: every file the bot receives is written to disk at once, before any question is asked.

`SB_LIBRARY_DIR/_Unsorted/` is the human-facing "arrived, not yet filed" folder. A file is written under a temporary name, flushed and
renamed into place, so a power cut or a sleeping computer never leaves a half file that looks complete. The ingest service later copies the
file into its subject folder, verifies the copy by hash and only then removes the spool file; until then the original is simply here.
"""
from __future__ import annotations

import os
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Callable

from ..ingest.files import EXTENSION, sha256_file, sniff
from ..ingest.naming import claim_unique_file, sanitize_component
from ..ingest.subjects import SPOOL_FOLDER

MAX_NAME_CHARS = 60
_PARTIAL = ".{}.partial"


@dataclass(frozen=True)
class SpoolFile:
    path: Path
    sha256: str
    size: int
    kind: str | None            # what the bytes really are (jpeg, pdf, ...), None if unknown


class Spool:
    def __init__(self, library_dir: Path, clock: Callable[[], datetime] = datetime.now):
        self.dir = Path(library_dir) / SPOOL_FOLDER
        self.clock = clock

    def write(self, data: bytes, original_name: str | None = None, fallback: str = "telegram") -> SpoolFile:
        """Store `data`; the extension comes from the bytes, not from the name the sender gave."""
        kind = sniff(bytes(data[:16]))
        ext = EXTENSION.get(kind, ".bin")
        stem = sanitize_component(Path(original_name or "").stem, MAX_NAME_CHARS, fallback=fallback)
        final = claim_unique_file(self.dir, f"{self.clock():%Y-%m-%d_%H%M%S}_{stem}", ext)
        temp = final.with_name(_PARTIAL.format(final.name))
        try:
            with temp.open("wb") as out:
                out.write(data)
                out.flush()
                os.fsync(out.fileno())
            os.replace(temp, final)
        except BaseException:
            temp.unlink(missing_ok=True)
            final.unlink(missing_ok=True)
            raise
        return SpoolFile(final, sha256_file(final), len(data), None if kind == "unknown" else kind)

    def sweep_partials(self) -> int:
        """At start-up, before anything is written: remove half-written files (and the empty name claimed for one) a crash left. Returns how many."""
        if not self.dir.is_dir():
            return 0
        stale = [p for p in self.dir.iterdir() if p.is_file() and (re.fullmatch(r"\..+\.partial", p.name) or p.stat().st_size == 0)]
        for path in stale:
            path.unlink(missing_ok=True)
        return len(stale)
