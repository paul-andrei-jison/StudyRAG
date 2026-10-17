"""File the original(s) into SB_LIBRARY_DIR/<Subject>/ as verified byte-copies.

Every copy is written under a temporary name, flushed to disk, checked by SHA-256 against the hash taken when the
job started, and only then renamed into place. Originals are never modified; a source is removed only after its
filed copy has been re-hashed. Everything this module creates is returned so a failed job can undo exactly that.
"""
from __future__ import annotations

import os
import shutil
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Callable, Sequence

from PIL import Image, ImageOps

from ..errors import StudyBuddyError
from .files import FileInfo, sha256_file
from .naming import NamingError, claim_unique_dir, claim_unique_file, fit_name, library_stem, subject_folder

PDF_MAX_EDGE = 2500   # the combined album PDF is a convenience copy; the page photos keep full resolution
PDF_QUALITY = 90

CopyFn = Callable[[Path, Path], None]


class FilingError(StudyBuddyError):
    """Filing failed. `transient` means a retry may work (changed source, flaky disk)."""

    def __init__(self, message: str, transient: bool = True):
        super().__init__(message)
        self.transient = transient


@dataclass
class FiledDocument:
    final_path: Path                 # the filed file, or the album folder
    page_files: list[Path]           # filed copy of each source, in order
    pdf_path: Path | None = None     # combined PDF of an album
    created: list[Path] = field(default_factory=list)   # everything this call created (files and folders)


def _copy_bytes(src: Path, dst: Path) -> None:
    with src.open("rb") as fin, dst.open("wb") as fout:
        shutil.copyfileobj(fin, fout, 1024 * 1024)
        fout.flush()
        os.fsync(fout.fileno())


def _rmdir_if_empty(folder: Path) -> None:
    """Remove a subject folder this job created and left empty (never one that holds other documents)."""
    try:
        folder.rmdir()
    except OSError:
        pass


def _temp_name(final: Path) -> Path:
    return final.with_name(f".{final.name}.partial")


class Filer:
    def __init__(self, library_dir: Path, copy_fn: CopyFn = _copy_bytes):
        self.library_dir = library_dir
        self._copy = copy_fn

    # ---- one verified copy -----------------------------------------------------------------

    def _copy_verified(self, src: Path, final: Path, expected_hash: str) -> None:
        """Copy `src` onto the already-claimed `final` path; raises FilingError (leaving no partial file)."""
        temp = _temp_name(final)
        try:
            self._copy(src, temp)
            got = sha256_file(temp)
            if got != expected_hash:
                raise FilingError(
                    f"{src.name}: the copy does not match the original (the file changed or the disk misbehaved); "
                    f"it will be tried again", transient=True)
            os.replace(temp, final)
        except FilingError:
            raise
        except OSError as exc:
            raise FilingError(f"{src.name}: could not be copied into the library ({exc})", transient=True) from exc
        finally:
            temp.unlink(missing_ok=True)

    # ---- documents -------------------------------------------------------------------------

    def file_document(self, sources: Sequence[Path], infos: Sequence[FileInfo], hashes: Sequence[str], subject: str,
                      title: str, received_at: datetime, is_album: bool) -> FiledDocument:
        subject_dir = self.library_dir / subject_folder(subject)
        stem = library_stem(received_at, subject, title)
        if is_album and len(sources) > 1:
            return self._file_album(sources, infos, hashes, subject_dir, stem)
        return self._file_single(sources[0], infos[0], hashes[0], subject_dir, stem)

    def _file_single(self, src: Path, info: FileInfo, digest: str, subject_dir: Path, stem: str) -> FiledDocument:
        final = claim_unique_file(subject_dir, stem, info.ext)
        try:
            self._copy_verified(src, final, digest)
        except BaseException:
            final.unlink(missing_ok=True)
            _rmdir_if_empty(final.parent)
            raise
        return FiledDocument(final, [final], None, [final])

    def _file_album(self, sources: Sequence[Path], infos: Sequence[FileInfo], hashes: Sequence[str],
                    subject_dir: Path, stem: str) -> FiledDocument:
        folder = claim_unique_dir(subject_dir, stem)
        pages: list[Path] = []
        try:
            for number, (src, info, digest) in enumerate(zip(sources, infos, hashes), start=1):
                final = claim_unique_file(folder, f"page-{number:02d}", info.ext)
                self._copy_verified(src, final, digest)
                pages.append(final)
            pdf = claim_unique_file(folder, self._pdf_stem(folder), ".pdf")
            self._build_pdf(pages, pdf)
        except BaseException:
            shutil.rmtree(folder, ignore_errors=True)  # we just created this folder: it holds only our files
            _rmdir_if_empty(folder.parent)
            raise
        return FiledDocument(folder, pages, pdf, [folder])

    @staticmethod
    def _pdf_stem(folder: Path) -> str:
        """The PDF is named after its folder; if a very long library path would cut that name short, call it `album`."""
        try:
            fitted = fit_name(folder, folder.name, ".pdf")
        except NamingError:
            return "album"
        return folder.name if fitted == f"{folder.name}.pdf" else "album"

    def _build_pdf(self, pages: Sequence[Path], target: Path) -> None:
        temp = _temp_name(target)
        images: list[Image.Image] = []
        try:
            for page in pages:
                with Image.open(page) as img:
                    img = ImageOps.exif_transpose(img).convert("RGB")
                    longest = max(img.size)
                    if longest > PDF_MAX_EDGE:
                        ratio = PDF_MAX_EDGE / longest
                        img = img.resize((round(img.width * ratio), round(img.height * ratio)), Image.LANCZOS)
                    images.append(img.copy())
            images[0].save(temp, format="PDF", save_all=True, append_images=images[1:], quality=PDF_QUALITY)
            os.replace(temp, target)
        except Exception as exc:
            raise FilingError(f"could not build the combined PDF ({type(exc).__name__}: {exc})", transient=False) from exc
        finally:
            temp.unlink(missing_ok=True)

    # ---- undo / cleanup --------------------------------------------------------------------

    @staticmethod
    def rollback(created: Sequence[Path]) -> None:
        """Delete exactly what a failed job created; anything else in the library is left alone."""
        for path in created:
            if path.is_dir():
                shutil.rmtree(path, ignore_errors=True)
            else:
                path.unlink(missing_ok=True)
            _rmdir_if_empty(path.parent)

    @staticmethod
    def remove_sources(sources: Sequence[Path], filed: Sequence[Path], hashes: Sequence[str]) -> list[str]:
        """Remove each source, but only after its filed copy is re-hashed and matches. Returns problems (never raises)."""
        problems: list[str] = []
        for src, copy, digest in zip(sources, filed, hashes):
            try:
                if not copy.is_file() or sha256_file(copy) != digest:
                    problems.append(f"{src.name}: kept in place because its filed copy could not be verified")
                    continue
                src.unlink(missing_ok=True)
            except OSError as exc:
                problems.append(f"{src.name}: could not be removed from the inbox ({exc})")
        return problems


def safe_move(src: Path, dest_dir: Path, name: str | None = None) -> Path:
    """Move a file into `dest_dir` (copy, verify by hash, then delete the source). Used for duplicates and failed/.

    Raises FilingError and leaves the source untouched if the copy cannot be verified."""
    dest_dir.mkdir(parents=True, exist_ok=True)
    stem, ext = os.path.splitext(name or src.name)
    target = claim_unique_file(dest_dir, stem, ext)
    try:
        digest = sha256_file(src)
        _copy_bytes(src, target)
        if sha256_file(target) != digest:
            raise FilingError(f"{src.name}: the moved copy does not match the original", transient=True)
        src.unlink()
    except FilingError:
        target.unlink(missing_ok=True)
        raise
    except OSError as exc:
        target.unlink(missing_ok=True)
        raise FilingError(f"{src.name}: could not be moved ({exc})", transient=True) from exc
    return target
