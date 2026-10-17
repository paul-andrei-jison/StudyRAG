"""Detect sideways / upside-down pages when the photo carries no EXIF orientation.

Telegram strips EXIF from compressed photos, so a page held sideways arrives unrotated. The cheap OCR
detector can tell: text lines of a sideways page are tall boxes, and text of an upside-down page
recognises far worse than the same page the right way up.

Why not just score four rotations: PP-OCR's recogniser silently turns tall line crops by 90 degrees,
so the page rotated one way scores as well as the upright page. The box shape settles the axis first.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from PIL import Image

PROBE_EDGE = 960        # long edge used for the probe: fast, and text is still legible
MIN_LINES = 3           # fewer detected lines than this: not enough evidence, leave the image alone
SIDEWAYS_ASPECT = 2.0   # a line box this much taller than wide counts as sideways text
FLIP_MARGIN = 1.25      # the flipped page must read this much better before we flip


@dataclass(frozen=True)
class Line:
    width: float
    height: float
    text: str
    confidence: float


class LineProbe(Protocol):
    def probe(self, image: Image.Image) -> list[Line]:
        """Detect and recognise text lines with NO built-in orientation correction."""


def _score(lines: list[Line]) -> float:
    return sum(len(line.text.strip()) * line.confidence for line in lines)


def _small(image: Image.Image) -> Image.Image:
    scale = PROBE_EDGE / max(image.size)
    if scale >= 1:
        return image
    return image.resize((max(1, round(image.width * scale)), max(1, round(image.height * scale))),
                        Image.LANCZOS)


def detect_rotation(image: Image.Image, probe: LineProbe) -> int:
    """Counter-clockwise rotation in degrees (0, 90, 180 or 270) that makes the page upright.

    Returns 0 whenever the evidence is weak, so a blank or photo-only page is never rotated by guesswork.
    """
    small = _small(image.convert("RGB"))
    lines = probe.probe(small)
    if len(lines) < MIN_LINES:
        return 0

    tall = sum(1 for ln in lines if ln.height > SIDEWAYS_ASPECT * ln.width)
    wide = sum(1 for ln in lines if ln.width > SIDEWAYS_ASPECT * ln.height)
    if tall > wide:
        base, base_lines, base_deg = small.transpose(Image.Transpose.ROTATE_90), None, 90
    else:
        base, base_lines, base_deg = small, lines, 0
    if base_lines is None:
        base_lines = probe.probe(base)

    flipped_lines = probe.probe(base.transpose(Image.Transpose.ROTATE_180))
    if _score(flipped_lines) > FLIP_MARGIN * _score(base_lines):
        return (base_deg + 180) % 360
    return base_deg


def rotate_ccw(image: Image.Image, degrees: int) -> Image.Image:
    """Exact (lossless) rotation by a multiple of 90 degrees, counter-clockwise."""
    ops = {90: Image.Transpose.ROTATE_90, 180: Image.Transpose.ROTATE_180, 270: Image.Transpose.ROTATE_270}
    return image.transpose(ops[degrees]) if degrees else image


def probe_of(ocr: object) -> LineProbe | None:
    """The OCR engine's probe, if it offers one (fakes and other engines may not)."""
    return ocr if callable(getattr(ocr, "probe", None)) else None  # type: ignore[return-value]
