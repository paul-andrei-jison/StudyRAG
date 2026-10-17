"""Per-page record and quality score. Thresholds are heuristics: tune them with the bake-off."""
from __future__ import annotations

import re
from dataclasses import dataclass, field

ILLEGIBLE_RE = re.compile(r"\[illegible\]", re.IGNORECASE)
WORD_RE = re.compile(r"\S+")

MANY_ILLEGIBLE = 0.10      # share of words that are [illegible]
MIN_CHARS_PER_MP = 20.0    # fewer transcribed characters per megapixel than this: suspiciously short
LOW_AGREEMENT = 0.60       # READER vs classic OCR
HIGH_GLARE = 0.15          # share of near-white pixels in the source photo
ATTENTION_QUALITY = 0.60

ATTENTION_FLAGS = {"empty", "error", "many_illegible", "low_agreement"}


@dataclass
class PageRecord:
    page_number: int                      # 1-based, same numbering as the input
    source_path: str                      # original file this page came from (never a copy)
    source_kind: str                      # "image" | "pdf-scan" | "pdf-text"
    text: str = ""
    quality: float = 0.0                  # 0..1
    flags: list[str] = field(default_factory=list)
    error: str | None = None
    illegible_ratio: float = 0.0
    chars_per_megapixel: float | None = None
    ocr_agreement: float | None = None    # None when no OCR engine ran or it failed
    image_size: tuple[int, int] | None = None   # original size after EXIF rotation
    sent_size: tuple[int, int] | None = None    # size actually sent to the READER
    steps: list[str] = field(default_factory=list)
    skew_deg: float = 0.0
    page_found: bool = False

    @property
    def needs_attention(self) -> bool:
        return self.quality < ATTENTION_QUALITY or bool(ATTENTION_FLAGS & set(self.flags))


def illegible_ratio(text: str) -> float:
    words = WORD_RE.findall(text)
    return len(ILLEGIBLE_RE.findall(text)) / len(words) if words else 0.0


def score_page(text: str, image_size: tuple[int, int], ocr_agreement: float | None,
               glare_fraction: float = 0.0) -> tuple[float, list[str], float, float]:
    """Return (quality, flags, illegible_ratio, chars_per_megapixel)."""
    flags: list[str] = []
    stripped = text.strip()
    megapixels = max(image_size[0] * image_size[1] / 1_000_000, 1e-6)
    cpm = len(stripped) / megapixels
    ratio = illegible_ratio(stripped)

    if not stripped:
        return 0.0, ["empty"], 0.0, 0.0

    quality = 1.0
    if ratio > MANY_ILLEGIBLE:
        flags.append("many_illegible")
    quality *= 1.0 - min(1.0, 3.0 * ratio)
    if cpm < MIN_CHARS_PER_MP:
        flags.append("too_short")
        quality *= 0.5
    if ocr_agreement is not None:
        if ocr_agreement < LOW_AGREEMENT:
            flags.append("low_agreement")
        quality *= ocr_agreement
    if glare_fraction > HIGH_GLARE:
        flags.append("glare")
        quality *= 0.8
    return round(quality, 3), flags, ratio, cpm
