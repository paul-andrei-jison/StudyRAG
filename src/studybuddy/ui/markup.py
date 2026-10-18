"""Small pure functions that build the few pieces of HTML and the visible wording the screens need (everything is escaped here).

Keeping them out of `app.py` means they are tested without a browser, and the words a student reads live in one place.
"""
from __future__ import annotations

import html
from datetime import datetime
from pathlib import Path
from typing import Sequence

from ..chat.answers import Source

TITLE_LIMIT = 48


def esc(text: object) -> str:
    return html.escape(str(text or ""), quote=True)


def short_title(title: str, limit: int = TITLE_LIMIT) -> str:
    """Cut at a word boundary, with no ellipsis (a button label must not end in dots)."""
    title = " ".join((title or "").split())
    if len(title) <= limit:
        return title
    cut = title[:limit]
    return (cut.rsplit(" ", 1)[0] if " " in cut else cut).rstrip(",;:-")


# ---- Ask ----------------------------------------------------------------------------------------------------------------

def greeting(now: datetime | None = None) -> str:
    hour = (now or datetime.now()).hour
    hello = "Good morning" if hour < 12 else "Good afternoon" if hour < 18 else "Good evening"
    return (f"<div class='sb-hello'><h1>{hello}.</h1><p>What are you studying?</p></div>")


def suggest_questions(rows: Sequence) -> list[str]:
    """Up to three example questions built from the student's own papers (`rows`: newest first, with .title and .subject)."""
    titled = [r for r in rows if (r.title or "").strip()]
    if not titled:
        return []
    out: list[str] = []
    newest = short_title(titled[0].title)
    out.append(f"What is {newest} about?")
    counts: dict[str, int] = {}
    for r in titled:
        if r.subject:
            counts[r.subject] = counts.get(r.subject, 0) + 1
    if counts:
        subject = max(counts, key=lambda name: (counts[name], name))
        out.append(f"Summarize my {subject} papers" if counts[subject] > 1 else f"What are the main ideas in my {subject} paper?")
    other = titled[1] if len(titled) > 1 else titled[0]
    out.append(f"What are the key terms in {short_title(other.title)}?")
    seen: set[str] = set()
    return [q for q in out if not (q in seen or seen.add(q))]


# ---- the paper slip ------------------------------------------------------------------------------------------------------

def slip_caption(source: Source | None, problem: str | None = None) -> str:
    """The file name and page under the page image, and a plain warning when the page was hard to read or cannot be shown."""
    if source is None:
        return f"<div class='sb-slip-caption'><p>{esc(problem)}</p></div>"
    name = Path(source.file_path).name
    parts = [f"<p class='sb-slip-file' title='{esc(source.file_path)}'><strong>{esc(name)}</strong>, page {source.page_number}</p>"]
    meta = " · ".join(x for x in (source.subject, source.title) if x)
    if meta:
        parts.append(f"<p class='sb-muted'>{esc(meta)}</p>")
    if source.cited:
        parts.append("<p class='sb-muted'>Used in the answer.</p>")
    if source.hard_to_read:
        why = "; ".join(source.problems)
        parts.append(f"<p class='sb-note' role='note'><span aria-hidden='true'>&#9888;</span> This page was hard to read"
                     f"{': ' + esc(why) if why else ''}. Check the photo.</p>")
    if problem:
        parts.append(f"<p class='sb-note' role='note'>{esc(problem)}</p>")
    return "<div class='sb-slip-caption'>" + "".join(parts) + "</div>"
