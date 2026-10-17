"""Query expansion (optional, off by default): search with the question and two rephrasings, then fuse the results.

Measured on real pages before being switched on (see docs/chat.md): it only stays enabled if it helps retrieval
without raising false answers. Raw scores in the fused hits always refer to the ORIGINAL question: a chunk found only
through a rephrasing has no cosine for the original, so it cannot inflate the answerability gate.
"""
from __future__ import annotations

import dataclasses
import json
import re
from typing import Sequence

from ..errors import LLMError, TransientLLMError
from ..index.index import Index
from ..index.search import RRF_K, Hit, rrf
from ..ingest.metadata import Chats

MAX_PHRASINGS = 2
MAX_PHRASING_CHARS = 200
_JSON_LIST = re.compile(r"\[.*\]", re.DOTALL)

SYSTEM_PROMPT = (
    "You help search a student's notes. Given a question, write alternative phrasings of the SAME question that use "
    "different words (synonyms, other terms a handout might use). Do not answer it and do not add new topics. "
    'Reply with ONLY a JSON list of strings, for example ["...", "..."].'
)


def parse_phrasings(raw: str, question: str, limit: int = MAX_PHRASINGS) -> list[str]:
    """Up to `limit` distinct phrasings from a model answer; anything unusable gives []."""
    match = _JSON_LIST.search(raw or "")
    if not match:
        return []
    try:
        items = json.loads(match.group(0))
    except ValueError:
        return []
    seen, out = {question.strip().casefold()}, []
    for item in items if isinstance(items, list) else []:
        if not isinstance(item, str):
            continue
        text = " ".join(item.split())
        if text and len(text) <= MAX_PHRASING_CHARS and text.casefold() not in seen:
            seen.add(text.casefold())
            out.append(text)
    return out[:limit]


def expand_query(chat: Chats, question: str, limit: int = MAX_PHRASINGS) -> list[str]:
    """Alternative phrasings of `question`. A transient model failure propagates; other failures give no expansion."""
    messages = [{"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": f"Question: {question}\nWrite {limit} alternative phrasings."}]
    try:
        return parse_phrasings(chat.chat(messages, temperature=0.0, max_tokens=150), question, limit)
    except TransientLLMError:
        raise
    except LLMError:
        return []


def fuse_searches(index: Index, queries: Sequence[str], k: int, subject: str | None = None) -> list[Hit]:
    """Hybrid-search each query and fuse the rankings (reciprocal rank fusion across queries).

    queries[0] is the original question: hits it found keep their raw scores; hits found only by a rephrasing get
    vector_score=None so the gate never treats a rephrasing's similarity as evidence about the real question.
    """
    rankings: list[list[int]] = []
    by_id: dict[int, Hit] = {}
    for position, query in enumerate(queries):
        hits = index.search(query, k=k, subject=subject, mode="hybrid")
        rankings.append([h.chunk_id for h in hits])
        for h in hits:
            if position == 0:
                by_id[h.chunk_id] = h
            elif h.chunk_id not in by_id:
                by_id[h.chunk_id] = dataclasses.replace(h, vector_score=None, keyword_score=None)
    fused = rrf(rankings, RRF_K)
    ordered = sorted(fused, key=lambda cid: (-fused[cid], cid))
    return [dataclasses.replace(by_id[cid], score=fused[cid]) for cid in ordered[:k]]
