"""Do the nomic task prefixes help? Compare vector retrieval with and without them on a question set.

Vector-only on purpose: hybrid search would hide the effect. Chunks come from proofread reference texts in
SB_SAMPLES_DIR (outside the repo), questions from SB_SAMPLES_DIR/questions.json:

    [{"question": "What covers 71% of Earth?", "expect": ["page-stem"], "expect_text": "water"}, ...]

A chunk counts as correct when its document stem is in `expect` (a string or a list) and, if given,
`expect_text` occurs in it (case-insensitive). Nothing from the samples is stored in the repo.
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

Embed = Callable[[list[str]], list[list[float]]]

VARIANTS = ("none", "both", "doc-only", "query-only")


@dataclass(frozen=True)
class Question:
    question: str
    expect: tuple[str, ...]
    expect_text: str = ""


@dataclass(frozen=True)
class ChunkText:
    stem: str
    text: str


@dataclass
class QuestionResult:
    question: str
    rank: int | None        # 1-based rank of the first correct chunk, None if no chunk is correct
    margin: float | None    # best correct cosine minus best wrong cosine (positive = right answer on top)
    top_cosine: float


@dataclass
class VariantResult:
    name: str
    doc_prefix: str
    query_prefix: str
    results: list[QuestionResult] = field(default_factory=list)

    def hit_at(self, k: int) -> float:
        return sum(1 for r in self.results if r.rank is not None and r.rank <= k) / len(self.results)

    @property
    def mrr(self) -> float:
        return sum(1.0 / r.rank for r in self.results if r.rank) / len(self.results)

    @property
    def mean_margin(self) -> float:
        margins = [r.margin for r in self.results if r.margin is not None]
        return sum(margins) / len(margins) if margins else 0.0


def load_questions(path: Path) -> list[Question]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, list) or not data:
        raise ValueError(f"{path.name} must be a non-empty JSON list")
    questions = []
    for i, item in enumerate(data, 1):
        if isinstance(item, dict) and item.get("kind") == "overview":
            continue      # whole-document questions have no single answering chunk
        if not isinstance(item, dict) or not item.get("question") or not item.get("expect"):
            raise ValueError(f"{path.name} entry {i} needs 'question' and 'expect'")
        expect = item["expect"] if isinstance(item["expect"], list) else [item["expect"]]
        questions.append(Question(item["question"], tuple(str(e) for e in expect), str(item.get("expect_text", ""))))
    return questions


def is_correct(chunk: ChunkText, question: Question) -> bool:
    if chunk.stem not in question.expect:
        return False
    return not question.expect_text or question.expect_text.casefold() in chunk.text.casefold()


def cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na, nb = math.sqrt(sum(x * x for x in a)), math.sqrt(sum(y * y for y in b))
    return dot / (na * nb) if na and nb else 0.0


def prefixes_for(variant: str, doc_prefix: str, query_prefix: str) -> tuple[str, str]:
    return {"none": ("", ""), "both": (doc_prefix, query_prefix), "doc-only": (doc_prefix, ""),
            "query-only": ("", query_prefix)}[variant]


def compare(chunks: list[ChunkText], questions: list[Question], embed: Embed, doc_prefix: str,
            query_prefix: str, variants: tuple[str, ...] = VARIANTS) -> list[VariantResult]:
    """Score every variant. `embed` is the RAW embedding call (no prefixes added by it)."""
    for q in questions:
        if not any(is_correct(c, q) for c in chunks):
            raise ValueError(f"No chunk can answer: {q.question!r} (check expect / expect_text)")
    out: list[VariantResult] = []
    doc_cache: dict[str, list[list[float]]] = {}
    query_cache: dict[str, list[list[float]]] = {}
    for name in variants:
        d_prefix, q_prefix = prefixes_for(name, doc_prefix, query_prefix)
        if d_prefix not in doc_cache:
            doc_cache[d_prefix] = embed([d_prefix + c.text for c in chunks])
        if q_prefix not in query_cache:
            query_cache[q_prefix] = embed([q_prefix + q.question for q in questions])
        result = VariantResult(name, d_prefix, q_prefix)
        for q, qvec in zip(questions, query_cache[q_prefix]):
            sims = [cosine(qvec, dvec) for dvec in doc_cache[d_prefix]]
            order = sorted(range(len(chunks)), key=lambda i: -sims[i])
            rank = next((pos for pos, i in enumerate(order, 1) if is_correct(chunks[i], q)), None)
            right = [sims[i] for i in range(len(chunks)) if is_correct(chunks[i], q)]
            wrong = [sims[i] for i in range(len(chunks)) if not is_correct(chunks[i], q)]
            margin = (max(right) - max(wrong)) if right and wrong else None
            result.results.append(QuestionResult(q.question, rank, margin, sims[order[0]]))
        out.append(result)
    return out


def format_results(results: list[VariantResult], n_chunks: int) -> str:
    n_q = len(results[0].results)
    lines = [f"{n_q} questions over {n_chunks} chunks (vector search only; small sample, treat as indicative)",
             f"{'variant':12} {'hit@1':>6} {'hit@3':>6} {'MRR':>6} {'margin':>7}"]
    for r in results:
        lines.append(f"{r.name:12} {r.hit_at(1):6.2f} {r.hit_at(3):6.2f} {r.mrr:6.2f} {r.mean_margin:7.3f}")
    lines.append("")
    lines.append("Per question, rank of the first correct chunk (. = none):")
    lines.append(f"{'#':>3} " + " ".join(f"{r.name:>10}" for r in results))
    for i in range(n_q):
        cells = " ".join(f"{(str(r.results[i].rank) if r.results[i].rank else '.'):>10}" for r in results)
        lines.append(f"{i + 1:>3} {cells}")
    return "\n".join(lines)
