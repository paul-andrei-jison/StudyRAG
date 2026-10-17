"""Command-line front end for searching the index: `scripts/search.py`."""
from __future__ import annotations

import argparse
import sys
from typing import TextIO

from ..config import Settings
from ..errors import StudyBuddyError
from .cli import _setup
from .index import Embedder, Index
from .search import MODES


def search_main(argv: list[str], settings: Settings | None = None, embedder: Embedder | None = None,
                out: TextIO | None = None) -> int:
    out = out or sys.stdout
    parser = argparse.ArgumentParser(prog="search.py", description="Search the index from the command line.")
    parser.add_argument("query")
    parser.add_argument("--subject", help="only this subject (case-insensitive)")
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--mode", choices=MODES, default="hybrid")
    parser.add_argument("--text", action="store_true", help="print each chunk's full text, not a snippet")
    args = parser.parse_args(argv)

    try:
        settings, embedder, client = _setup(settings, embedder)
    except StudyBuddyError as exc:
        print(f"[FAIL] {exc}", file=out)
        return 1
    try:
        index = Index.open(settings, embedder)
        try:
            hits = index.search(args.query, k=args.k, subject=args.subject, mode=args.mode)
        finally:
            index.close()
    except StudyBuddyError as exc:
        print(f"[FAIL] {exc}", file=out)
        return 1
    finally:
        if client is not None:
            client.close()

    if not hits:
        print("No results.", file=out)
        return 0
    for rank, h in enumerate(hits, start=1):
        vec = f"vec #{h.vector_rank} cos {h.vector_score:.3f}" if h.vector_rank else "vec -"
        kw = f"kw #{h.keyword_rank} bm25 {h.keyword_score:.2f}" if h.keyword_rank else "kw -"
        print(f"{rank}. [{h.subject or 'no subject'}] {h.title or '(untitled)'} - {h.file_path}, page {h.page_number}"
              f"   score {h.score:.4f} ({vec} | {kw})", file=out)
        flags = f" flags: {', '.join(h.page_flags)}" if h.page_flags else ""
        quality = f"{h.page_quality:.2f}" if h.page_quality is not None else "n/a"
        print(f"   image: {h.image_path}   page quality {quality}{flags}", file=out)
        body = h.text if args.text else h.snippet
        for line in body.splitlines() or [""]:
            print(f"   {line}", file=out)
    return 0
