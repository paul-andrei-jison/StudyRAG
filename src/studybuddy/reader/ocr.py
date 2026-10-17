"""Classic OCR engine used only as an independent cross-check on READER output."""
from __future__ import annotations

from typing import Protocol

import numpy as np
from PIL import Image


class OcrEngine(Protocol):
    name: str

    def read(self, image: Image.Image) -> str:
        """Return the page text, one detected line per output line."""


class RapidOcrEngine:
    """RapidOCR (PP-OCR models on ONNX Runtime). Loaded lazily: it is slow to import."""

    name = "rapidocr"

    def __init__(self) -> None:
        self._engine = None

    def read(self, image: Image.Image) -> str:
        if self._engine is None:
            from rapidocr import RapidOCR  # lazy: keeps imports and tests light

            self._engine = RapidOCR()
        result = self._engine(np.array(image.convert("RGB")))
        return "\n".join(result.txts) if result.txts else ""
