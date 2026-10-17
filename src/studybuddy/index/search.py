"""Hybrid retrieval: vector search and FTS5 keyword search merged by reciprocal rank fusion.

RRF scores are relative (they only order candidates), so every Hit also carries the raw cosine similarity and
bm25 score: the chat layer needs those to decide that the sources do not contain an answer.
"""
from __future__ import annotations

import re
import sqlite3
from dataclasses import dataclass

from ..errors import IndexMismatchError, IndexStoreError
from .index import REINDEX_HINT, Index
from .store import subject_key

MODES = ("hybrid", "vector", "keyword")
POOL = 50
RRF_K = 60
SNIPPET_CHARS = 240

STOPWORDS = frozenset("""
a an the is are was were be been being am of to in on at for and or but nor so by with from as into than then
that this these those it its it's i you he she we they me my your our their do does did done have has had
what which who whom whose how why when where whether can could would should will shall may might must
about above after again all any both each few more most other some such no not only own same too very just
""".split())

_WORD = re.compile(r"\w+", re.UNICODE)


@dataclass
class Hit:
    chunk_id: int
    doc_id: int
    score: float                 # fused RRF score (hybrid), cosine (vector) or bm25 (keyword)
    text: str
    snippet: str
    subject: str
    title: str
    file_path: str               # the document in the library
    page_number: int
    image_path: str              # original image of the cited page (a PDF path for PDF pages)
    page_quality: float | None
    page_flags: list[str]
    vector_rank: int | None = None     # 1-based rank in the vector list, None if absent
    keyword_rank: int | None = None
    vector_score: float | None = None  # raw cosine similarity
    keyword_score: float | None = None  # raw bm25 (bigger is better)


def query_terms(query: str) -> list[str]:
    """Lowercased unique words; stopwords dropped unless nothing else is left."""
    words = [w for w in _WORD.findall(query.lower()) if len(w) > 1 or w.isdigit()]
    seen: set[str] = set()
    unique = [w for w in words if not (w in seen or seen.add(w))]
    content = [w for w in unique if w not in STOPWORDS]
    return content or unique


def build_match(query: str) -> str | None:
    """A safe FTS5 query: each term quoted (so operators like AND, NEAR, * or : are plain words), OR-joined."""
    terms = query_terms(query)
    if not terms:
        return None
    return " OR ".join('"' + t.replace('"', '""') + '"' for t in terms)


def rrf(rankings: list[list[int]], k: int = RRF_K, weights: list[float] | None = None) -> dict[int, float]:
    """Reciprocal rank fusion: each list contributes weight / (k + rank) per item (rank is 1-based)."""
    weights = weights or [1.0] * len(rankings)
    scores: dict[int, float] = {}
    for ranking, weight in zip(rankings, weights):
        for rank, item in enumerate(ranking, start=1):
            scores[item] = scores.get(item, 0.0) + weight / (k + rank)
    return scores


def make_snippet(text: str, terms: list[str], width: int = SNIPPET_CHARS) -> str:
    flat = re.sub(r"\s+", " ", text).strip()
    if len(flat) <= width:
        return flat
    lower = flat.lower()
    positions = [p for p in (lower.find(t) for t in terms) if p >= 0]
    start = max(0, min(positions) - width // 4) if positions else 0
    end = min(len(flat), start + width)
    start = max(0, end - width)
    return ("... " if start else "") + flat[start:end].strip() + (" ..." if end < len(flat) else "")


def search(index: Index, query: str, k: int = 8, subject: str | None = None, mode: str = "hybrid",
           pool: int = POOL, rrf_k: int = RRF_K, vector_weight: float = 1.0,
           keyword_weight: float = 1.0) -> list[Hit]:
    """Top-k chunks for `query`, optionally restricted to one subject (case-insensitive)."""
    if mode not in MODES:
        raise ValueError(f"mode must be one of {MODES}")
    if not query.strip() or k <= 0:
        return []

    vector_hits: list[tuple[int, float]] = []
    keyword_hits: list[tuple[int, float]] = []
    if mode in ("hybrid", "vector") and index.vectors.count() > 0:  # nothing stored: skip the embedding call
        qvec = index.embedder.embed_queries([query])[0]
        try:
            vector_hits = index.vectors.search(qvec, pool, subject_key(subject) if subject else None)
        except IndexStoreError as exc:
            raise IndexMismatchError(f"{exc}. The query embedding does not match the stored vectors. {REINDEX_HINT}") from exc
    if mode in ("hybrid", "keyword"):
        match = build_match(query)
        if match:
            try:
                keyword_hits = index.store.keyword_search(match, pool, subject)
            except sqlite3.OperationalError:
                keyword_hits = []

    v_rank = {cid: r for r, (cid, _) in enumerate(vector_hits, start=1)}
    k_rank = {cid: r for r, (cid, _) in enumerate(keyword_hits, start=1)}
    v_score, k_score = dict(vector_hits), dict(keyword_hits)

    if mode == "vector":
        fused = {cid: s for cid, s in vector_hits}
    elif mode == "keyword":
        fused = {cid: s for cid, s in keyword_hits}
    else:
        fused = rrf([[c for c, _ in vector_hits], [c for c, _ in keyword_hits]], rrf_k,
                    [vector_weight, keyword_weight])

    inf = float("inf")
    ordered = sorted(fused, key=lambda c: (-fused[c], v_rank.get(c, inf), k_rank.get(c, inf), c))
    rows = index.store.chunks_by_id(ordered)
    terms = query_terms(query)

    hits: list[Hit] = []
    for cid in ordered:
        row = rows.get(cid)
        if row is None:  # an orphan vector (see Index.check): skip it rather than cite nothing
            continue
        hits.append(Hit(
            chunk_id=cid, doc_id=row.doc_id, score=fused[cid], text=row.text, snippet=make_snippet(row.text, terms),
            subject=row.subject, title=row.title, file_path=row.file_path, page_number=row.page_number,
            image_path=row.image_path, page_quality=row.page_quality, page_flags=row.page_flags,
            vector_rank=v_rank.get(cid), keyword_rank=k_rank.get(cid),
            vector_score=v_score.get(cid), keyword_score=k_score.get(cid),
        ))
        if len(hits) == k:
            break
    return hits
