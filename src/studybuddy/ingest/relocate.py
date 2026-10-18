"""Move a filed document to another subject folder: the file(s), the index and every stored path, all or nothing.

Run it on the ingest worker (`IngestService.run_exclusive` / `try_exclusive`), so it never overlaps a document being read.

  1. Check that every file exists inside the library; otherwise refuse and change nothing.
  2. Write a small journal (what will be created) to SB_DATA_DIR/relocations/.
  3. Copy each file into the new subject folder, verifying the copy by SHA-256 against the source before it is renamed into place.
  4. Rewrite the subject and every stored path in one SQLite transaction (then the vector store); if that fails the rows are put back.
  5. Only after that: re-hash the new copies, delete the old files, remove the old folder when it is empty.

A failure in steps 1-4 deletes exactly what this call created, so the document is as it was. A crash leaves the journal;
`recover()` rolls back (database still on the old paths) or finishes the clean-up (database already on the new paths).
"""
from __future__ import annotations

import json
import logging
import os
import re
import shutil
from dataclasses import dataclass, field
from pathlib import Path

from ..errors import StudyBuddyError
from ..index.index import Index
from .files import sha256_file
from .filing import CopyFn, Filer, FilingError, _copy_bytes
from .naming import claim_unique_dir, claim_unique_file, subject_folder, subject_token

log = logging.getLogger(__name__)

JOURNAL_DIRNAME = "relocations"
_DATED_STEM = re.compile(r"^(?P<when>\d{4}-\d{2}-\d{2}_\d{4})_(?P<subject>.+?)_(?P<rest>.+)$")


class RelocationError(StudyBuddyError):
    """The document was not moved; the message says why in plain words. Nothing was changed."""


@dataclass
class Relocation:
    doc_id: int
    old_subject: str
    new_subject: str
    new_path: str                       # the new file, or the new album folder
    moved: bool = True                  # False: it was already filed under that subject
    warnings: list[str] = field(default_factory=list)


def _inside(path: Path, root: Path) -> bool:
    return path == root or root in path.parents


class Relocator:
    def __init__(self, index: Index, library_dir: Path, journal_dir: Path, copy_fn: CopyFn = _copy_bytes):
        self.index = index
        self.library_dir = Path(library_dir).resolve()
        self.journal_dir = Path(journal_dir)
        self.filer = Filer(self.library_dir, copy_fn)

    # ---- the move ---------------------------------------------------------------------------

    def relocate(self, doc_id: int, new_subject: str) -> Relocation:
        doc = self.index.store.get_document(doc_id)
        if doc is None:
            raise RelocationError("That document no longer exists.")
        if (doc.subject or "").casefold() == new_subject.casefold():
            return Relocation(doc_id, doc.subject, new_subject, doc.file_path, moved=False)

        old_root = Path(doc.file_path)
        album = old_root.is_dir()
        files = self._files_of(doc, old_root, album)
        for path in [old_root, *files]:
            self._check_inside(path)
        new_dir = self.library_dir / subject_folder(new_subject)
        new_stem = self._new_stem(old_root.stem if not album else old_root.name, doc.subject, new_subject)

        created: list[Path] = []
        mapping: dict[str, str] = {}
        warnings: list[str] = []
        try:
            digests = {f: sha256_file(f) for f in files}
            if not album and digests[old_root] != doc.content_hash:
                warnings.append("the file in the library is not byte-for-byte the one that was read; it was moved as it is")
            if album:
                folder = claim_unique_dir(new_dir, new_stem)
                created.append(folder)
                mapping[str(old_root)] = str(folder)
                for f in files:
                    name = f.name
                    if f.suffix.lower() == ".pdf" and f.stem == old_root.name:
                        name = f"{folder.name}.pdf"          # the album PDF is named after its folder
                    target = claim_unique_file(folder, Path(name).stem, f.suffix)
                    self._journal(doc, new_subject, mapping, created + [target], state="copying")
                    self.filer._copy_verified(f, target, digests[f])
                    mapping[str(f)] = str(target)
            else:
                target = claim_unique_file(new_dir, new_stem, old_root.suffix)
                created.append(target)
                self._journal(doc, new_subject, {str(old_root): str(target)}, created, state="copying")
                self.filer._copy_verified(old_root, target, digests[old_root])
                mapping[str(old_root)] = str(target)
            self._journal(doc, new_subject, mapping, created, state="copied")
            self.index.relocate_document(doc_id, new_subject, mapping)
        except BaseException as exc:
            Filer.rollback(created)
            self._drop_journal(doc_id)
            if isinstance(exc, (RelocationError, KeyboardInterrupt, SystemExit)):
                raise
            raise RelocationError(f"The document was not moved and nothing was changed ({self._plain(exc)}).") from exc

        warnings += self._remove_old(files, old_root, album, mapping, digests)
        self._drop_journal(doc_id)
        return Relocation(doc_id, doc.subject, new_subject, mapping[str(old_root)], True, warnings)

    # ---- pieces -----------------------------------------------------------------------------

    def _files_of(self, doc, old_root: Path, album: bool) -> list[Path]:
        if not album:
            if not old_root.is_file():
                raise RelocationError(f"The file is missing from the library ({old_root.name}), so the document was not moved.")
            return [old_root]
        pages = [Path(p["image_path"]) for p in self.index.store.pages(doc.id)]
        missing = [p.name for p in pages if not p.is_file()]
        if missing:
            raise RelocationError(f"Album page(s) are missing from the library ({', '.join(missing)}), so the document was not moved.")
        extras = sorted(p for p in old_root.iterdir() if p.is_file() and p not in pages and p.suffix.lower() == ".pdf"
                        and not p.name.startswith("."))
        return pages + extras

    def _check_inside(self, path: Path) -> None:
        try:
            real = path.resolve(strict=False)
        except (OSError, ValueError, RuntimeError) as exc:
            raise RelocationError(f"A stored path cannot be read ({exc}), so the document was not moved.") from exc
        if not _inside(real, self.library_dir):
            raise RelocationError("The file is outside the library folder, so it was not moved.")

    @staticmethod
    def _new_stem(old_stem: str, old_subject: str, new_subject: str) -> str:
        """`2026-10-17_1425_Math_title` -> `2026-10-17_1425_Science_title`; a name that does not follow the pattern is kept."""
        match = _DATED_STEM.match(old_stem)
        if match and match.group("subject") == subject_token(old_subject):
            return f"{match.group('when')}_{subject_token(new_subject)}_{match.group('rest')}"
        return old_stem

    @staticmethod
    def _plain(exc: BaseException) -> str:
        return str(exc) if isinstance(exc, StudyBuddyError) else f"{type(exc).__name__}: {exc}"

    def _remove_old(self, files: list[Path], old_root: Path, album: bool, mapping: dict[str, str], digests: dict) -> list[str]:
        """After the index points at the new copies: verify them once more, then delete the old ones. Never raises."""
        warnings: list[str] = []
        for f in files:
            new = Path(mapping[str(f)]) if str(f) in mapping else None
            try:
                if new is None or not new.is_file() or sha256_file(new) != digests[f]:
                    warnings.append(f"{f.name}: the old copy was kept because the new one could not be verified")
                    continue
                f.unlink(missing_ok=True)
            except OSError as exc:
                warnings.append(f"{f.name}: the old copy could not be removed ({exc})")
        parents = {old_root} if album else set()
        parents |= {f.parent for f in files}
        for folder in sorted(parents, key=lambda p: len(p.parts), reverse=True):
            if folder.resolve() == self.library_dir:
                continue
            try:
                folder.rmdir()                       # only succeeds when empty: never a folder with other documents
            except OSError:
                if folder == old_root and album:
                    warnings.append(f"the old album folder still holds other files and was kept: {folder}")
        return warnings

    # ---- journal and crash recovery ---------------------------------------------------------

    def _journal_path(self, doc_id: int) -> Path:
        return self.journal_dir / f"{doc_id}.json"

    def _journal(self, doc, new_subject: str, mapping: dict[str, str], created: list[Path], state: str) -> None:
        self.journal_dir.mkdir(parents=True, exist_ok=True)
        data = {"doc_id": doc.id, "old_subject": doc.subject, "new_subject": new_subject, "old_path": doc.file_path,
                "mapping": mapping, "created": [str(p) for p in created], "state": state}
        tmp = self._journal_path(doc.id).with_suffix(".json.tmp")
        tmp.write_text(json.dumps(data, indent=2), encoding="utf-8")
        os.replace(tmp, self._journal_path(doc.id))

    def _drop_journal(self, doc_id: int) -> None:
        self._journal_path(doc_id).unlink(missing_ok=True)

    def recover(self) -> list[str]:
        """Settle moves a crash interrupted. Returns one plain sentence per journal found."""
        notes: list[str] = []
        for path in sorted(self.journal_dir.glob("*.json")) if self.journal_dir.exists() else []:
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
                doc = self.index.store.get_document(int(data["doc_id"]))
                created = [Path(p) for p in data["created"]]
                new_paths = set(data["mapping"].values())
                if doc is not None and doc.file_path in new_paths and data["state"] == "copied":
                    old = [Path(o) for o in data["mapping"] if Path(o).exists() and Path(o) not in created]
                    for f in old:
                        if f.is_file() and Path(data["mapping"][str(f)]).is_file():
                            f.unlink()
                    for folder in {f.parent for f in old} | {Path(data["old_path"])}:
                        try:
                            folder.rmdir()
                        except OSError:
                            pass
                    notes.append(f"document {data['doc_id']}: finished moving it to {data['new_subject']}")
                else:
                    Filer.rollback(created)
                    notes.append(f"document {data['doc_id']}: an interrupted move to {data['new_subject']} was undone")
            except (OSError, ValueError, KeyError, FilingError) as exc:
                log.warning("could not settle %s: %s", path, exc)
                notes.append(f"{path.name}: could not be settled ({exc}); check the library folder")
                continue
            path.unlink(missing_ok=True)
        return notes
