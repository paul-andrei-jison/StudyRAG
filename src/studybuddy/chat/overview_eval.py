"""Measure overview questions end to end on your own pages (files in SB_SAMPLES_DIR, outside the repo).

`questions.json` may hold items `{"question": ..., "kind": "overview", "expect_docs": ["page-stem", ...]}`: whole-document questions
that must be answered from those pages. `nearmiss.json` holds `{"question": ..., "must": "normal" | "refuse"}`: questions that contain
document words but are not overview requests ("normal": must stay on the page-search path) or that must be refused. This module only
loads and judges; `scripts/eval_overview.py` runs the real pipeline.
"""
from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence

from .answers import ANSWERED, NOT_IN_PAPERS_STATUS, OVERVIEW, SEARCH, ChatAnswer

OVERVIEW_GROUP, ANSWERABLE_GROUP, UNANSWERABLE_GROUP, NEARMISS_GROUP = "overview", "answerable", "unanswerable", "near-miss"


@dataclass(frozen=True)
class OverviewCase:
    question: str
    expect_docs: tuple[str, ...]


@dataclass(frozen=True)
class NearMiss:
    question: str
    must: str            # normal | refuse


@dataclass
class Outcome:
    group: str
    question: str
    ok: bool
    why: str = ""
    status: str = ""
    mode: str = ""
    seconds: float = 0.0
    extra: dict = field(default_factory=dict)


def load_overview_cases(samples_dir: Path) -> list[OverviewCase]:
    items = json.loads((samples_dir / "questions.json").read_text(encoding="utf-8"))
    cases = []
    for item in items:
        if item.get("kind") == "overview":
            docs = item.get("expect_docs") or []
            cases.append(OverviewCase(item["question"], tuple(str(d) for d in (docs if isinstance(docs, list) else [docs]))))
    return cases


def load_near_misses(samples_dir: Path) -> list[NearMiss]:
    path = samples_dir / "nearmiss.json"
    if not path.is_file():
        return []
    out = []
    for item in json.loads(path.read_text(encoding="utf-8")):
        must = item.get("must", "normal")
        if must not in ("normal", "refuse"):
            raise ValueError(f"nearmiss.json: must is 'normal' or 'refuse', got {must!r}")
        out.append(NearMiss(item["question"], must))
    return out


def judge_overview(case: OverviewCase, answer: ChatAnswer, source_stems: Sequence[str], seconds: float) -> Outcome:
    problems = []
    if answer.status != ANSWERED:
        problems.append(f"status {answer.status}")
    if answer.mode != OVERVIEW:
        problems.append(f"mode {answer.mode}")
    if not answer.cited:
        problems.append("no valid citation")
    if answer.invalid_citations:
        problems.append(f"invalid citations {answer.invalid_citations}")
    missing = [d for d in case.expect_docs if d not in set(source_stems)]
    if missing:
        problems.append(f"missing documents {missing}")
    return Outcome(OVERVIEW_GROUP, case.question, not problems, "; ".join(problems), answer.status, answer.mode, seconds,
                   {"cited": answer.cited, "documents": sorted(set(source_stems))})


def judge_answerable(question: str, answer: ChatAnswer, seconds: float) -> Outcome:
    problems = []
    if answer.status != ANSWERED:
        problems.append(f"refused or failed ({answer.status})")
    if answer.mode != SEARCH:
        problems.append("wrongly treated as an overview")
    return Outcome(ANSWERABLE_GROUP, question, not problems, "; ".join(problems), answer.status, answer.mode, seconds)


def judge_unanswerable(question: str, answer: ChatAnswer, seconds: float) -> Outcome:
    problems = []
    if answer.status != NOT_IN_PAPERS_STATUS:
        problems.append(f"a false answer ({answer.status})")
    if answer.mode != SEARCH:
        problems.append("wrongly treated as an overview")
    return Outcome(UNANSWERABLE_GROUP, question, not problems, "; ".join(problems), answer.status, answer.mode, seconds)


def judge_near_miss(case: NearMiss, answer: ChatAnswer, seconds: float) -> Outcome:
    problems = []
    if answer.mode != SEARCH:
        problems.append("wrongly detected as an overview")
    if case.must == "refuse" and answer.status != NOT_IN_PAPERS_STATUS:
        problems.append(f"should have been refused, got {answer.status}")
    return Outcome(NEARMISS_GROUP, case.question, not problems, "; ".join(problems), answer.status, answer.mode, seconds,
                   {"must": case.must})


def tally(outcomes: Sequence[Outcome]) -> dict[str, tuple[int, int]]:
    """group -> (passed, total)."""
    counts: dict[str, list[int]] = defaultdict(lambda: [0, 0])
    for o in outcomes:
        counts[o.group][1] += 1
        counts[o.group][0] += o.ok
    return {g: (p, t) for g, (p, t) in counts.items()}


def wrongly_detected(outcomes: Sequence[Outcome]) -> int:
    """Questions that are NOT overview requests but were handled as one (near misses, answerable and unanswerable ones)."""
    return sum(1 for o in outcomes if o.group != OVERVIEW_GROUP and o.mode == OVERVIEW)


def format_report(outcomes: Sequence[Outcome]) -> str:
    lines = []
    for o in outcomes:
        lines.append(f"{'ok   ' if o.ok else 'FAIL '} [{o.group:12}] {o.seconds:5.1f}s  {o.status:13} {o.mode:8} {o.question[:64]}"
                     + (f"   <-- {o.why}" if o.why else ""))
    lines.append("")
    for group, (passed, total) in tally(outcomes).items():
        lines.append(f"{group:13} {passed}/{total}")
    lines.append(f"wrongly detected as overview (not an overview question): {wrongly_detected(outcomes)}")
    return "\n".join(lines)
