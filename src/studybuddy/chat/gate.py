"""Do the retrieved sources answer the question? Decided from the raw scores, before any model is asked.

Rank-fusion scores only order results, so the decision uses the raw numbers of the best-matching source:
the vector cosine (nomic: relevant chunks ~0.70-0.80, unrelated ~0.50-0.55) and the share of the question's
content words that actually occur in that source. Two thresholds, kept simple on purpose:

    answerable  if  top_cosine >= HIGH
                or  top_cosine >= LOW  and  keyword coverage of that source >= COVERAGE

The numbers are calibrated on real questions (see docs/chat.md and `scripts/calibrate_gate.py`), not guessed.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Sequence

from ..index.search import Hit, query_terms

_WORD = re.compile(r"\w+", re.UNICODE)
_SUFFIXES = ("ing", "ed", "es", "ly", "s")


@dataclass(frozen=True)
class GateConfig:
    high_cosine: float = 0.70      # at or above: answerable on the vector score alone
    low_cosine: float = 0.60       # at or above (and below high): needs the keywords to agree
    min_coverage: float = 0.50     # share of the question's content words found in the best source


@dataclass(frozen=True)
class Signals:
    top_cosine: float | None       # best raw cosine among the sources (None: no vector evidence at all)
    coverage: float                # content-word coverage of THAT source (best-cosine one)
    max_coverage: float            # best coverage of any source
    sources: int


@dataclass(frozen=True)
class Assessment:
    answerable: bool
    reason: str
    signals: Signals


def forms(word: str) -> frozenset[str]:
    """The word and its plausible stems (plates/plate, moving/move, gases/gas all share a form)."""
    word = word.lower()
    out = {word}
    for suffix in _SUFFIXES:
        if len(word) > len(suffix) + 2 and word.endswith(suffix):
            base = word[: -len(suffix)]
            out.add(base)
            if suffix in ("ing", "ed"):
                out.add(base + "e")  # moving -> mov / move
    return frozenset(out)


def coverage(question: str, text: str) -> float:
    """Share of the question's content words (stopwords removed, plurals and endings ignored) that appear in `text`."""
    terms = query_terms(question)
    if not terms:
        return 0.0
    seen: set[str] = set()
    for word in _WORD.findall(text.lower()):
        seen |= forms(word)
    return sum(1 for term in terms if forms(term) & seen) / len(terms)


def signals_for(hits: Sequence[Hit], question: str) -> Signals:
    if not hits:
        return Signals(None, 0.0, 0.0, 0)
    scored = [h for h in hits if h.vector_score is not None]
    best = max(scored, key=lambda h: h.vector_score) if scored else None
    coverages = [coverage(question, h.text) for h in hits]
    return Signals(best.vector_score if best else None,
                   coverage(question, best.text) if best else (max(coverages) if coverages else 0.0),
                   max(coverages), len(hits))


def decide(signals: Signals, config: GateConfig = GateConfig()) -> tuple[bool, str]:
    if signals.sources == 0:
        return False, "nothing was found in your papers for this question"
    if signals.top_cosine is None:  # keyword-only evidence: stricter, since there is no semantic check
        ok = signals.max_coverage >= max(config.min_coverage, 0.75)
        return ok, ("the keywords match" if ok else "no semantic match and too few keywords matched")
    if signals.top_cosine >= config.high_cosine:
        return True, f"strong match (cosine {signals.top_cosine:.2f})"
    if signals.top_cosine >= config.low_cosine and signals.coverage >= config.min_coverage:
        return True, f"moderate match (cosine {signals.top_cosine:.2f}) and the question's words appear in it"
    return False, (f"the closest page is only a weak match (cosine {signals.top_cosine:.2f}, "
                   f"{signals.coverage:.0%} of the question's words)")


def assess(hits: Sequence[Hit], question: str, config: GateConfig = GateConfig()) -> Assessment:
    signals = signals_for(hits, question)
    answerable, reason = decide(signals, config)
    return Assessment(answerable, reason, signals)
