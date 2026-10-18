"""Which document(s) an overview question is about, and asking the student when that is not clear.

Order of the rules, inside the subject filter if there is one: a document the question names; the one document that is
left; "this / latest / just uploaded" and a document added in the last hour; a few documents (answered one by one);
otherwise ask, listing the most recent ones. A topic that matches no paper means the question is not an overview of
anything the student has, so nothing is resolved and the normal path (and its refusal) takes over.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Sequence

from ..index.store import DocumentRow
from .gate import coverage
from .overview import Detection

RECENT_MINUTES = 60          # "the latest upload" counts only if it arrived in this window
FEW_DOCUMENTS = 3            # up to this many documents are each summarised instead of asking
LIST_MAX = 8                 # candidates printed when asking which paper
NAME_COVERAGE = 0.6          # share of a title's words that must appear in the question for the title to count as named
TOPIC_COVERAGE = 0.5         # share of the topic words that must appear in a title, subject or stored summary

DOCUMENTS, ASK, NO_MATCH = "documents", "ask", "no_match"
_WORDS = re.compile(r"\w+", re.UNICODE)


@dataclass
class Resolution:
    kind: str                                   # documents | ask | no_match
    docs: list[DocumentRow] = field(default_factory=list)
    why: str = ""                               # named | only | latest | few
    candidates: list[DocumentRow] = field(default_factory=list)   # for ask: everything in scope
    shown: list[DocumentRow] = field(default_factory=list)        # for ask: the numbered list, newest first
    text: str = ""                              # for ask: the message


def _added(doc: DocumentRow) -> datetime:
    try:
        when = datetime.fromisoformat(doc.created_at)
    except ValueError:
        return datetime.min.replace(tzinfo=timezone.utc)
    return when if when.tzinfo else when.replace(tzinfo=timezone.utc)


def newest_first(docs: Sequence[DocumentRow]) -> list[DocumentRow]:
    return sorted(docs, key=lambda d: (_added(d), d.id), reverse=True)


def _named(question: str, scope: Sequence[DocumentRow]) -> DocumentRow | None:
    """The one document whose title the question (mostly) spells out; a tie or a weak match names nothing."""
    scored = sorted(((coverage(d.title, question), d) for d in scope if d.title.strip()), key=lambda x: -x[0])
    if scored and scored[0][0] >= NAME_COVERAGE and (len(scored) == 1 or scored[1][0] < scored[0][0]):
        return scored[0][1]
    return None


def _on_topic(topic: Sequence[str], doc: DocumentRow) -> bool:
    return coverage(" ".join(topic), f"{doc.title} {doc.subject} {doc.summary}") >= TOPIC_COVERAGE


def resolve_targets(detection: Detection, question: str, docs: Sequence[DocumentRow], subject: str | None = None,
                    now: datetime | None = None) -> Resolution:
    now = now or datetime.now(timezone.utc)
    wanted = (subject or "").strip().casefold()
    scope = [d for d in docs if not wanted or (d.subject or "").casefold() == wanted]
    if not scope:
        return Resolution(NO_MATCH)
    named = _named(question, scope)
    if named is not None:
        return Resolution(DOCUMENTS, [named], "named")
    if detection.topic_terms:
        on_topic = [d for d in scope if _on_topic(detection.topic_terms, d)]
        if not on_topic:
            return Resolution(NO_MATCH)          # a topic none of the papers is about: let the normal path refuse it
        scope = on_topic
    if len(scope) == 1:
        return Resolution(DOCUMENTS, scope, "only")
    newest = newest_first(scope)[0]
    if detection.recent_hint and now - _added(newest) <= timedelta(minutes=RECENT_MINUTES):
        return Resolution(DOCUMENTS, [newest], "latest")
    if len(scope) <= FEW_DOCUMENTS:
        return Resolution(DOCUMENTS, sorted(scope, key=lambda d: d.id), "few")
    return ask_which(scope)


def _line(number: int, doc: DocumentRow) -> str:
    return f"{number}. {doc.subject or 'no subject'} | {doc.title or '(untitled)'} | {doc.created_at[:10]}"


def ask_which(candidates: Sequence[DocumentRow], header: str = "Which paper do you mean?") -> Resolution:
    shown = newest_first(candidates)[:LIST_MAX]
    lines = [header + " Reply with its number or part of its title.", ""]
    lines += [_line(i, d) for i, d in enumerate(shown, 1)]
    hidden = len(candidates) - len(shown)
    if hidden > 0:
        lines += ["", f"{hidden} more are not shown: pick a subject to narrow it, or type part of a title."]
    return Resolution(ASK, candidates=list(candidates), shown=shown, text="\n".join(lines))


# ---- the reply to "which paper?" ----------------------------------------------------------------

@dataclass
class Choice:
    kind: str                                   # chosen | ambiguous | invalid | none
    doc: DocumentRow | None = None
    ask: Resolution | None = None               # ambiguous / invalid: what to show and remember next


def match_choice(reply: str, pending: Resolution) -> Choice:
    """A number from the shown list, or a typed title or part of one (every word of the reply must be in the title)."""
    text = " ".join((reply or "").split())
    if not text:
        return Choice("none")
    if text.rstrip(".").isdigit():
        number = int(text.rstrip("."))
        if 1 <= number <= len(pending.shown):
            return Choice("chosen", pending.shown[number - 1])
        again = Resolution(ASK, candidates=pending.candidates, shown=pending.shown,
                           text=f"There is no number {number} in the list.\n\n{pending.text}")
        return Choice("invalid", ask=again)
    if not [w for w in _WORDS.findall(text.lower()) if len(w) > 1]:
        return Choice("none")
    exact = [d for d in pending.candidates if d.title.strip().casefold() == text.casefold()]
    if len(exact) == 1:
        return Choice("chosen", exact[0])
    matches = [d for d in pending.candidates if coverage(text, d.title) == 1.0]
    if len(matches) == 1:
        return Choice("chosen", matches[0])
    if len(matches) > 1:
        return Choice("ambiguous", ask=ask_which(matches, header="Several papers match."))
    return Choice("none")
