"""Agreement between the READER transcript and the classic OCR text (never rewrites either)."""
from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass

from rapidfuzz import fuzz

_FIGURE = re.compile(r"\[figure:[^\]]*\]", re.IGNORECASE)
_ILLEGIBLE = re.compile(r"\[illegible\]", re.IGNORECASE)
_TABLE_RULE = re.compile(r"^\s*\|?\s*:?-{3,}:?\s*(\|\s*:?-{3,}:?\s*)*\|?\s*$", re.MULTILINE)
_WORD = re.compile(r"\w+", re.UNICODE)


@dataclass(frozen=True)
class Agreement:
    word_f1: float   # overlap of the two word multisets: ignores reading order, so columns can differ
    char_sim: float  # character similarity of the sorted words
    score: float     # mean of the two, 0..1


def normalise(text: str) -> list[str]:
    """Lowercased words with markup that only one engine can produce removed."""
    text = _FIGURE.sub(" ", text)
    text = _ILLEGIBLE.sub(" ", text)
    text = _TABLE_RULE.sub(" ", text)
    text = text.replace("$", " ").replace("\\", " ").replace("|", " ")
    return [w.lower() for w in _WORD.findall(text)]


def agreement(reader_text: str, ocr_text: str) -> Agreement:
    a, b = normalise(reader_text), normalise(ocr_text)
    if not a and not b:
        return Agreement(1.0, 1.0, 1.0)
    if not a or not b:
        return Agreement(0.0, 0.0, 0.0)
    common = sum((Counter(a) & Counter(b)).values())
    precision, recall = common / len(a), common / len(b)
    f1 = 0.0 if common == 0 else 2 * precision * recall / (precision + recall)
    char = fuzz.ratio(" ".join(sorted(a)), " ".join(sorted(b))) / 100.0
    return Agreement(f1, char, (f1 + char) / 2)
