"""Calibrate the answerability gate on labelled questions and report false-refusal and false-answer rates.

Cases come from files OUTSIDE the repo (SB_SAMPLES_DIR): `questions.json` (answerable, from the student's pages) and
`unanswerable.json` ([{"question": "...", "kind": "far" | "near"}]: topics the pages do not cover, "near" ones share
words with the pages). Only the chosen thresholds and the rates are ever committed.
"""
from __future__ import annotations

import itertools
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Sequence

from ..index.search import Hit
from .gate import GateConfig, Signals, decide, signals_for
from .select import spread

ANSWERABLE, FAR, NEAR = "answerable", "far", "near"
POOL = 20


@dataclass(frozen=True)
class Case:
    question: str
    kind: str   # answerable | far | near

    @property
    def answerable(self) -> bool:
        return self.kind == ANSWERABLE


@dataclass(frozen=True)
class Scored:
    case: Case
    signals: Signals


@dataclass(frozen=True)
class Rates:
    n_answerable: int
    n_unanswerable: int
    false_refusals: int     # answerable questions the gate refused
    false_answers: int      # unanswerable questions the gate let through
    far_false_answers: int
    near_false_answers: int

    @property
    def false_refusal_rate(self) -> float:
        return self.false_refusals / self.n_answerable if self.n_answerable else 0.0

    @property
    def false_answer_rate(self) -> float:
        return self.false_answers / self.n_unanswerable if self.n_unanswerable else 0.0


def load_cases(samples_dir: Path) -> list[Case]:
    items = json.loads((samples_dir / "questions.json").read_text(encoding="utf-8"))
    # items with "kind": "overview" ask about whole documents; they are measured by scripts/eval_overview.py, not by the gate
    cases = [Case(q["question"], ANSWERABLE) for q in items if q.get("kind") != "overview"]
    path = samples_dir / "unanswerable.json"
    if path.is_file():
        for item in json.loads(path.read_text(encoding="utf-8")):
            kind = item.get("kind", FAR)
            if kind not in (FAR, NEAR):
                raise ValueError(f"unanswerable.json: kind must be 'far' or 'near', got {kind!r}")
            cases.append(Case(item["question"], kind))
    return cases


def collect(search: Callable[[str], Sequence[Hit]], cases: Sequence[Case]) -> list[Scored]:
    """Run retrieval (as chat does: pool, then spread) and record the gate's signals for every case."""
    return [Scored(c, signals_for(spread(search(c.question)), c.question)) for c in cases]


def evaluate(scored: Sequence[Scored], config: GateConfig) -> Rates:
    refused_ok = answered_bad = far_bad = near_bad = n_a = n_u = 0
    for s in scored:
        answerable, _ = decide(s.signals, config)
        if s.case.answerable:
            n_a += 1
            refused_ok += 0 if answerable else 1
        else:
            n_u += 1
            if answerable:
                answered_bad += 1
                far_bad += s.case.kind == FAR
                near_bad += s.case.kind == NEAR
    return Rates(n_a, n_u, refused_ok, answered_bad, far_bad, near_bad)


def candidate_values(scored: Sequence[Scored]) -> tuple[list[float], list[float]]:
    cosines = sorted({round(s.signals.top_cosine, 3) for s in scored if s.signals.top_cosine is not None})
    covs = sorted({0.34, 0.5, 0.67, 0.75, 1.0} | {round(s.signals.coverage, 2) for s in scored})
    return cosines, covs


def sweep(scored: Sequence[Scored]) -> list[tuple[GateConfig, Rates]]:
    cosines, covs = candidate_values(scored)
    # thresholds midway between observed scores, so a chosen value is not sitting exactly on a data point
    mids = [round((a + b) / 2, 3) for a, b in zip(cosines, cosines[1:])] + [round(cosines[0] - 0.01, 3), round(cosines[-1] + 0.01, 3)]
    out = []
    for high, low in itertools.product(sorted(set(mids)), repeat=2):
        if low > high:
            continue
        for cov in covs:
            config = GateConfig(high, low, cov)
            out.append((config, evaluate(scored, config)))
    return out


def cost(r: Rates) -> tuple:
    """What to minimise: far-off questions answered (worst), then near-misses answered, then answerable ones refused."""
    return (r.far_false_answers, r.near_false_answers, r.false_refusals)


def choose(scored: Sequence[Scored]) -> tuple[GateConfig, Rates]:
    """The least-bad thresholds: the middle of all configurations that tie for the lowest cost, so the pick is not a knife-edge."""
    results = sweep(scored)
    best_cost = min(cost(r) for _, r in results)
    best = [(c, r) for c, r in results if cost(r) == best_cost]
    best.sort(key=lambda cr: (cr[0].high_cosine, cr[0].low_cosine, cr[0].min_coverage))
    config, rates = best[len(best) // 2]
    return config, rates


def format_table(scored: Sequence[Scored], config: GateConfig) -> str:
    lines = [f"{'kind':10} {'cosine':>7} {'cover':>6} {'gate':>8}  question"]
    for s in sorted(scored, key=lambda s: (s.case.kind, -(s.signals.top_cosine or 0))):
        ok, _ = decide(s.signals, config)
        cos = "-" if s.signals.top_cosine is None else f"{s.signals.top_cosine:.3f}"
        wrong = ok != s.case.answerable
        lines.append(f"{s.case.kind:10} {cos:>7} {s.signals.coverage:6.2f} {'ANSWER' if ok else 'REFUSE':>8}{' <-- WRONG' if wrong else ''}  {s.case.question[:70]}")
    return "\n".join(lines)


def format_rates(r: Rates, config: GateConfig) -> str:
    return (f"thresholds: high {config.high_cosine}, low {config.low_cosine}, coverage {config.min_coverage}\n"
            f"false refusals: {r.false_refusals}/{r.n_answerable} = {r.false_refusal_rate:.0%}   "
            f"false answers: {r.false_answers}/{r.n_unanswerable} = {r.false_answer_rate:.0%} "
            f"(far {r.far_false_answers}, near {r.near_false_answers})")
