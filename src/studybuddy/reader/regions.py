"""Score printed text and handwritten text separately.

The reference .txt has no markup, so handwriting is described in an optional sidecar file next to it,
`<stem>.hw`, one entry per line (blank lines and lines starting with # are ignored):

    Juan A. Cruz            literal text: every occurrence in the reference is handwritten
    re:^[A-D](?= \\d+\\.)             regex (MULTILINE): every match is handwritten (answer letters here)

An entry that matches nothing raises: a silently dead entry would make the printed score look worse than it is.
Insertions (extra text the model invented) are counted against the printed text.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import jiwer

from .textnorm import normalise_text

HW_SUFFIX = ".hw"


@dataclass(frozen=True)
class SplitRates:
    printed_cer: float
    printed_wer: float
    hw_cer: float | None      # None when the page has no handwritten region
    hw_wer: float | None
    hw_words: int
    printed_words: int


def load_regions(reference: str, sidecar: Path | None) -> list[tuple[int, int]]:
    """Character spans of `reference` that are handwritten, per the sidecar ([] when there is none)."""
    if sidecar is None or not sidecar.is_file():
        return []
    spans: list[tuple[int, int]] = []
    for number, raw in enumerate(sidecar.read_text(encoding="utf-8").splitlines(), 1):
        entry = raw.rstrip("\r\n")
        if not entry.strip() or entry.lstrip().startswith("#"):
            continue
        if entry.startswith("re:"):
            try:
                found = [(m.start(), m.end()) for m in re.finditer(entry[3:], reference, re.MULTILINE)]
            except re.error as exc:
                raise ValueError(f"{sidecar.name} line {number}: bad regex ({exc})") from exc
        else:
            found = [(m.start(), m.end()) for m in re.finditer(re.escape(entry), reference)]
        found = [(s, e) for s, e in found if e > s]
        if not found:
            raise ValueError(f"{sidecar.name} line {number} matches nothing in the reference: {entry!r}")
        spans.extend(found)
    return spans


def normalise_with_mask(text: str, spans: list[tuple[int, int]]) -> tuple[str, list[bool]]:
    """normalise_text() that also carries a per-character handwritten flag through the whitespace collapse."""
    flags = [False] * len(text)
    for start, end in spans:
        for i in range(start, min(end, len(text))):
            flags[i] = True
    out: list[str] = []
    mask: list[bool] = []
    i, n = 0, len(text)
    while i < n:
        if text[i].isspace():
            j = i
            while j < n and text[j].isspace():
                j += 1
            if out and j < n:  # interior run: one space, handwritten only if both sides are
                out.append(" ")
                mask.append(flags[i - 1] and flags[j])
            i = j
        else:
            out.append(text[i])
            mask.append(flags[i])
            i += 1
    normal = normalise_text("".join(out))
    assert len(normal) == len(mask), "normalisation must be 1:1 on already-collapsed text"
    return normal, mask


def _errors_by_tag(alignment, tags: list[bool]) -> tuple[list[int], list[int], int]:
    """Per-tag error counts over reference units + insertions: (errors[printed,hw], totals[printed,hw], inserts)."""
    errors, inserts = [0, 0], 0
    for chunk in alignment:
        if chunk.type in ("substitute", "delete"):
            for idx in range(chunk.ref_start_idx, chunk.ref_end_idx):
                errors[1 if tags[idx] else 0] += 1
        elif chunk.type == "insert":
            inserts += chunk.hyp_end_idx - chunk.hyp_start_idx
    hw_total = sum(tags)
    return errors, [len(tags) - hw_total, hw_total], inserts


def _rates(errors: list[int], totals: list[int], inserts: int) -> tuple[float, float | None]:
    printed = (errors[0] + inserts) / totals[0] if totals[0] else 0.0
    hw = errors[1] / totals[1] if totals[1] else None
    return printed, hw


def split_error_rates(reference: str, hypothesis: str, spans: list[tuple[int, int]]) -> SplitRates:
    ref, char_tags = normalise_with_mask(reference, spans)
    hyp = normalise_text(hypothesis)
    if not ref:
        raise ValueError("reference text is empty")

    words = list(re.finditer(r"\S+", ref))
    word_tags = [any(char_tags[m.start():m.end()]) for m in words]

    if not hyp:  # everything is missing
        cer_err, cer_tot, cer_ins = _errors_by_tag([_Chunk("delete", 0, len(ref))], char_tags)
        wer_err, wer_tot, wer_ins = _errors_by_tag([_Chunk("delete", 0, len(words))], word_tags)
    else:
        cer_err, cer_tot, cer_ins = _errors_by_tag(jiwer.process_characters(ref, hyp).alignments[0], char_tags)
        wer_err, wer_tot, wer_ins = _errors_by_tag(jiwer.process_words(ref, hyp).alignments[0], word_tags)
    p_cer, h_cer = _rates(cer_err, cer_tot, cer_ins)
    p_wer, h_wer = _rates(wer_err, wer_tot, wer_ins)
    return SplitRates(p_cer, p_wer, h_cer, h_wer, hw_words=wer_tot[1], printed_words=wer_tot[0])


@dataclass(frozen=True)
class _Chunk:
    type: str
    ref_start_idx: int
    ref_end_idx: int
    hyp_start_idx: int = 0
    hyp_end_idx: int = 0
