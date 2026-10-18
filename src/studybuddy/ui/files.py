"""Opening page images for the UI, and only from inside the library folder.

The index stores the path of each filed original. A handler never hands such a path to Gradio (which would serve the file
by URL): it asks the guard for a decoded, size-limited image instead, so a doctored path cannot read anything else.
"""
from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageOps

from ..errors import ReaderError, StudyBuddyError
from ..ingest.files import heif_supported, sniff
from ..reader.pdf import page_count, render_pdf_pages

DISPLAY_EDGE = 1800
PDF_DPI = 130
SHOWN_KINDS = {"jpeg", "png", "webp", "bmp", "tiff", "heic", "pdf"}


class PathNotAllowedError(StudyBuddyError):
    """The requested file is outside the library, or is not a page image."""


class PageUnavailableError(StudyBuddyError):
    """The file is in the library but cannot be shown (moved, deleted, damaged, page out of range)."""


class LibraryGuard:
    def __init__(self, library_dir: Path):
        self.root = Path(library_dir).resolve()

    def resolve(self, raw: str | Path) -> Path:
        """The real path of `raw` if it lies inside the library; raises otherwise. Symlinks and `..` are followed first."""
        text = str(raw)
        if not text.strip() or "\x00" in text:
            raise PathNotAllowedError("No file was given.")
        candidate = Path(text)
        if not candidate.is_absolute():
            candidate = self.root / candidate
        try:
            real = candidate.resolve(strict=False)
        except (OSError, ValueError, RuntimeError) as exc:
            raise PathNotAllowedError(f"That path cannot be opened ({exc}).") from exc
        if real != self.root and self.root not in real.parents:
            raise PathNotAllowedError("That file is outside the library folder, so it will not be opened.")
        if any(":" in part for part in real.relative_to(self.root).parts):
            raise PathNotAllowedError("That path names an alternate data stream and will not be opened.")
        return real

    def render_page(self, raw: str | Path, page_number: int = 1, max_edge: int = DISPLAY_EDGE) -> Image.Image:
        """The page as an RGB image no larger than `max_edge` (a PDF page is rendered; a photo is turned upright)."""
        path = self.resolve(raw)
        try:
            with path.open("rb") as f:
                kind = sniff(f.read(16))
        except FileNotFoundError as exc:
            raise PageUnavailableError(f"The file is no longer in the library: {path.name}") from exc
        except OSError as exc:
            raise PageUnavailableError(f"{path.name} cannot be read ({exc}).") from exc
        if kind not in SHOWN_KINDS:
            raise PathNotAllowedError(f"{path.name} is not a photo or PDF, so it will not be opened.")
        try:
            if kind == "pdf":
                image = self._pdf_page(path, page_number)
            else:
                if kind == "heic" and not heif_supported():
                    raise PageUnavailableError(f"{path.name} is a HEIC photo and pillow-heif is not installed.")
                with Image.open(path) as opened:
                    image = ImageOps.exif_transpose(opened).convert("RGB")
        except StudyBuddyError:
            raise
        except Exception as exc:  # damaged image, decompression bomb, ...
            raise PageUnavailableError(f"{path.name} cannot be shown ({type(exc).__name__}: {exc})") from exc
        if max(image.size) > max_edge:
            image.thumbnail((max_edge, max_edge))
        return image

    @staticmethod
    def _pdf_page(path: Path, page_number: int) -> Image.Image:
        try:
            total = page_count(path)
            if not 1 <= page_number <= total:
                raise PageUnavailableError(f"{path.name} has {total} page(s); page {page_number} does not exist.")
            return next(image for _, image in render_pdf_pages(path, dpi=PDF_DPI, only={page_number}))
        except ReaderError as exc:
            raise PageUnavailableError(str(exc)) from exc
