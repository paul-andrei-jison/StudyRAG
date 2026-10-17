"""Choose which retrieved chunks become sources: best first, but spread out instead of five chunks from one page."""
from __future__ import annotations

from collections import Counter
from typing import Sequence

from ..index.search import Hit

SOURCE_COUNT = 5
PER_PAGE = 2
PER_DOCUMENT = 3


def spread(hits: Sequence[Hit], n: int = SOURCE_COUNT, per_page: int = PER_PAGE, per_doc: int = PER_DOCUMENT) -> list[Hit]:
    """Up to `n` hits in score order with at most `per_page` per page and `per_doc` per document.

    If those limits leave fewer than `n` (the question really is about one page), the best of the skipped hits
    fill the remaining places, so spreading never costs an answer.
    """
    if n <= 0:
        return []
    chosen: list[Hit] = []
    by_page: Counter = Counter()
    by_doc: Counter = Counter()
    skipped: list[Hit] = []
    for hit in hits:
        page_key = (hit.doc_id, hit.page_number)
        if len(chosen) < n and by_page[page_key] < per_page and by_doc[hit.doc_id] < per_doc:
            chosen.append(hit)
            by_page[page_key] += 1
            by_doc[hit.doc_id] += 1
        else:
            skipped.append(hit)
    for hit in skipped:
        if len(chosen) >= n:
            break
        chosen.append(hit)
    rank = {id(h): i for i, h in enumerate(hits)}
    return sorted(chosen, key=lambda h: rank[id(h)])  # keep the retrieval order: best first
