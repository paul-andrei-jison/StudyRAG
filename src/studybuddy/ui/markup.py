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


# ---- Library ------------------------------------------------------------------------------------------------------------

def _date(added: str, with_year: bool = False) -> str:
    try:
        when = datetime.strptime(added[:16], "%Y-%m-%d %H:%M")
    except ValueError:
        return added
    return f"{when.day} {when:%b} {when.year}" if with_year else f"{when.day} {when:%b}"


def _pages(count: int) -> str:
    return f"{count} page{'s' if count != 1 else ''}"


def papers_html(rows: Sequence, selected: int | None = None) -> str:
    """The list of papers: one real button per paper (keyboard and screen-reader friendly); a small mark if a page was hard to read."""
    items = []
    for row in rows:
        title = row.title or "Untitled paper"
        meta = " · ".join(x for x in (row.subject, _date(row.added), _pages(row.pages)) if x)
        warn = ""
        if row.flagged:
            many = f"{row.flagged} pages" if row.flagged != 1 else "1 page"
            warn = f"<span class='sb-row-warn'><span aria-hidden='true'>&#9888;</span> {many} hard to read</span>"
        current = " aria-current='true'" if selected == row.id else ""
        items.append(f"<li><button type='button' class='sb-row' data-id='{int(row.id)}'{current}>"
                     f"<span class='sb-row-title'>{esc(title)}</span><span class='sb-row-meta'>{esc(meta)}</span>{warn}</button></li>")
    return f"<ul class='sb-papers'>{''.join(items)}</ul>" if items else ""


def library_empty(searching: bool) -> str:
    if searching:
        return "<p class='sb-empty'>No paper matches that. Try fewer words, or pick All subjects.</p>"
    return ("<div class='sb-empty'><h2>No papers yet</h2><p>Add a photo or a PDF on the Add screen, "
            "or send a photo to your Telegram bot. It shows up here when it has been read.</p></div>")


def paper_head_html(detail) -> str:
    """Title, where and when, the summary, and a plain explanation for every page that was hard to read."""
    row = detail.row
    meta = " · ".join(x for x in (row.subject, f"Added {_date(row.added, True)}", _pages(row.pages)) if x)
    parts = [f"<h2>{esc(row.title or 'Untitled paper')}</h2>", f"<p class='sb-muted'>{esc(meta)}</p>"]
    if row.summary:
        parts.append(f"<p class='sb-summary'>{esc(row.summary)}</p>")
    for page in detail.pages:
        if page.flagged:
            why = "; ".join(page.reasons) or "the reading quality was low"
            parts.append(f"<p class='sb-note' role='note'><span aria-hidden='true'>&#9888;</span> Page {page.page_number} was hard to read: "
                         f"{esc(why)}. Check it against the photo.</p>")
    return "<div class='sb-paper-head'>" + "".join(parts) + "</div>"
