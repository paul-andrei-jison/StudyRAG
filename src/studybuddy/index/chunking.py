"""Paragraph-aware chunking of one page of text.

Chunks never cross a page, so every citation has an exact page. The embedding model silently truncates
over-long input (verified: a 2,500-word input is accepted at a 2,048-token context), so MAX_CHARS is a hard cap.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

TARGET_CHARS = 1200   # preferred chunk size
MAX_CHARS = 1800      # hard cap (about 450 tokens)
OVERLAP_CHARS = 150   # tail of the previous chunk repeated at the start of the next
MIN_TAIL_CHARS = 200  # a trailing chunk shorter than this merges into the previous one

_SENTENCE_END = re.compile(r"(?<=[.!?])\s+")


@dataclass(frozen=True)
class Chunk:
    page_number: int
    chunk_index: int   # 0-based position within the page
    text: str


def _blocks(text: str) -> list[str]:
    """Paragraphs split on blank lines (a Markdown table has none, so it stays whole).

    A lone `[figure: ...]` line is joined to the paragraph after it."""
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text.replace("\r\n", "\n").replace("\r", "\n"))]
    paragraphs = [p for p in paragraphs if p]
    merged: list[str] = []
    carry = ""
    for p in paragraphs:
        if "\n" not in p and p.lower().startswith("[figure:"):
            carry = f"{carry}\n{p}" if carry else p
            continue
        merged.append(f"{carry}\n{p}" if carry else p)
        carry = ""
    if carry:
        merged.append(carry)
    return merged


def _group(pieces: list[str], sep: str, limit: int) -> list[str]:
    """Join consecutive pieces up to `limit`; a single piece over the limit is returned as is."""
    out: list[str] = []
    cur = ""
    for piece in pieces:
        if cur and len(cur) + len(sep) + len(piece) > limit:
            out.append(cur)
            cur = piece
        else:
            cur = f"{cur}{sep}{piece}" if cur else piece
    if cur:
        out.append(cur)
    return out


def _split_long(block: str, limit: int) -> list[str]:
    """Split a block longer than `limit`: by lines (tables), then sentences, then words."""
    if len(block) <= limit:
        return [block]
    result: list[str] = []
    for line_group in _group(block.split("\n"), "\n", limit):
        if len(line_group) <= limit:
            result.append(line_group)
            continue
        for sentence_group in _group(_SENTENCE_END.split(line_group), " ", limit):
            if len(sentence_group) <= limit:
                result.append(sentence_group)
            else:
                for group in _group(sentence_group.split(), " ", limit):
                    # a single token longer than the limit (e.g. a pasted base64 blob): cut it by characters
                    result.extend(group[i:i + limit] for i in range(0, len(group), limit))
    return result


def _overlap_tail(previous: str, overlap: int) -> str:
    """The last ~`overlap` characters of `previous`, starting at a word boundary."""
    if overlap <= 0 or len(previous) <= overlap:
        return previous if overlap > 0 else ""
    tail = previous[-overlap:]
    if not previous[-overlap - 1].isspace():  # we cut a word in half: drop the partial word
        parts = tail.split(None, 1)
        tail = parts[1] if len(parts) == 2 else ""
    return tail.strip()


def chunk_page(text: str, page_number: int, target: int = TARGET_CHARS, max_chars: int = MAX_CHARS,
               overlap: int = OVERLAP_CHARS) -> list[Chunk]:
    """Split one page into chunks of about `target` characters, never more than `max_chars`."""
    if target > max_chars:
        raise ValueError("target must not exceed max_chars")
    pieces: list[str] = []
    for block in _blocks(text):
        pieces.extend(_split_long(block, target))
    if not pieces:
        return []

    bodies: list[str] = []
    cur = ""
    for piece in pieces:
        if cur and len(cur) + 2 + len(piece) > target:
            bodies.append(cur)
            cur = piece
        else:
            cur = f"{cur}\n\n{piece}" if cur else piece
    bodies.append(cur)

    if len(bodies) > 1 and len(bodies[-1]) < MIN_TAIL_CHARS and len(bodies[-2]) + 2 + len(bodies[-1]) <= max_chars:
        bodies[-2:] = [f"{bodies[-2]}\n\n{bodies[-1]}"]

    chunks: list[Chunk] = []
    for i, body in enumerate(bodies):
        text_out = body
        if i > 0:
            tail = _overlap_tail(bodies[i - 1], overlap)
            room = max_chars - len(body) - 1
            if tail and room > 0:
                if len(tail) > room:  # keep the most recent words that still fit
                    tail = _overlap_tail(tail, room)
                text_out = f"{tail}\n{body}" if tail else body
        chunks.append(Chunk(page_number, i, text_out))
    return chunks
