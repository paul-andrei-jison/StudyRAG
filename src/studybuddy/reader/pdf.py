"""PDF handling with pypdfium2: text layer detection and page rendering for scanned PDFs."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import pypdfium2 as pdfium
from PIL import Image

from ..errors import ReaderError

MIN_TEXT_CHARS = 25     # fewer characters than this on a page: treat it as an image-only page
MAX_RENDER_EDGE = 4096  # cap oversized pages (posters) so rendering stays bounded


@dataclass(frozen=True)
class PdfPageInfo:
    page_number: int          # 1-based
    text: str                 # text layer content, "" when none
    has_text_layer: bool


def _open(path: Path | str) -> pdfium.PdfDocument:
    try:
        return pdfium.PdfDocument(str(path))
    except Exception as exc:  # pypdfium2 raises PdfiumError for bad/encrypted files
        raise ReaderError(f"Cannot open PDF {Path(path).name}: {exc}") from exc


def page_count(path: Path | str) -> int:
    pdf = _open(path)
    try:
        return len(pdf)
    finally:
        pdf.close()


def inspect_pdf(path: Path | str, min_chars: int = MIN_TEXT_CHARS) -> list[PdfPageInfo]:
    """One record per page, in order. Pages with a real text layer carry that text."""
    pdf = _open(path)
    infos: list[PdfPageInfo] = []
    try:
        for i in range(len(pdf)):
            page = pdf[i]
            textpage = page.get_textpage()
            try:
                text = textpage.get_text_bounded().replace("\r\n", "\n").strip()
            finally:
                textpage.close()
                page.close()
            usable = len(text) >= min_chars
            infos.append(PdfPageInfo(i + 1, text if usable else "", usable))
    finally:
        pdf.close()
    return infos


def render_pdf_pages(path: Path | str, dpi: int = 200) -> Iterator[tuple[int, Image.Image]]:
    """Yield (page_number, RGB image) for EVERY page; a render failure raises, never skips."""
    pdf = _open(path)
    try:
        total = len(pdf)
        for i in range(total):
            page = pdf[i]
            try:
                width, height = page.get_size()
                scale = min(dpi / 72.0, MAX_RENDER_EDGE / max(width, height))
                image = page.render(scale=scale).to_pil().convert("RGB")
            except Exception as exc:
                raise ReaderError(f"Failed to render page {i + 1} of {total}: {exc}") from exc
            finally:
                page.close()
            yield i + 1, image
    finally:
        pdf.close()
