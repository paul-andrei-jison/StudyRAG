"""Read a document's pages: every input page yields exactly one PageRecord."""
from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

from PIL import Image

from ..errors import ReaderError
from .agreement import agreement
from .images import clean_image
from .ocr import OcrEngine
from .quality import PageRecord, score_page
from .transcribe import Reader

log = logging.getLogger(__name__)


@dataclass
class PageInput:
    page_number: int
    source_path: str                          # the original file, recorded on the PageRecord
    image: bytes | Path | str | Image.Image | None = None   # what to read; None for text-layer pages
    text_layer: str | None = None             # PDF text layer: used directly, READER skipped
    source_kind: str = "image"                # "image" | "pdf-scan" | "pdf-text"


def read_page(page: PageInput, reader: Reader, ocr: OcrEngine | None, max_edge: int) -> PageRecord:
    if page.text_layer is not None:
        quality, flags, ratio, cpm = score_page(page.text_layer, (1000, 1000), None)
        flags = [f for f in flags if f != "too_short"]  # no image: the density heuristic does not apply
        return PageRecord(page.page_number, page.source_path, "pdf-text", text=page.text_layer,
                          quality=quality, flags=flags, illegible_ratio=ratio)

    record = PageRecord(page.page_number, page.source_path, page.source_kind)
    try:
        if page.image is None:
            raise ReaderError("page has neither an image nor a text layer")
        cleaned = clean_image(page.image, max_edge=max_edge)
        record.image_size, record.steps = cleaned.original_size, cleaned.steps
        record.skew_deg, record.page_found = cleaned.skew_deg, cleaned.page_found
        transcript = reader.transcribe(cleaned.image)
        record.text, record.sent_size = transcript.text, transcript.image_size

        if ocr is not None:
            try:
                record.ocr_agreement = agreement(transcript.text, ocr.read(cleaned.image)).score
            except Exception as exc:  # the cross-check must never lose a page
                log.warning("OCR cross-check failed on page %s: %s", page.page_number, exc)
                record.flags.append("ocr_failed")

        quality, flags, ratio, cpm = score_page(record.text, cleaned.original_size,
                                                record.ocr_agreement, cleaned.glare_fraction)
        record.quality, record.illegible_ratio, record.chars_per_megapixel = quality, ratio, cpm
        record.flags.extend(flags)
    except Exception as exc:  # record the failure; do not drop the page
        record.error = f"{type(exc).__name__}: {exc}"
        record.quality = 0.0
        record.flags = ["error"]
    return record


def read_pages(pages: Sequence[PageInput], reader: Reader, ocr: OcrEngine | None = None,
               max_edge: int = 2048) -> list[PageRecord]:
    """Pages in must equal pages out; anything else raises instead of returning a partial result."""
    records = [read_page(p, reader, ocr, max_edge) for p in pages]
    in_numbers = [p.page_number for p in pages]
    out_numbers = [r.page_number for r in records]
    if len(set(in_numbers)) != len(in_numbers):
        raise ReaderError(f"Duplicate page numbers in input: {in_numbers}")
    if out_numbers != in_numbers:
        raise ReaderError(f"Page mismatch: {len(pages)} pages in, records for {out_numbers}")
    return records
