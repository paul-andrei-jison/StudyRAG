"""The logic behind the Chat tab: streaming with a visible "waking the model" state, and turning answers into display text.

`ChatSession.ask_stream` blocks until the follow-up rewrite, query expansion and retrieval calls return, and with a cold
model that is the slow part. So it runs on a helper thread and this module's generator keeps the screen alive meanwhile.
"""
from __future__ import annotations

import logging
import queue
import threading
import time
from dataclasses import dataclass
from typing import Iterator

from PIL import Image

from ..chat.answers import ERROR, NEEDS_CHOICE, NOT_IN_PAPERS_STATUS, ChatAnswer, Progress, Source
from ..chat.session import ChatSession
from ..errors import ChatContextError, IndexMismatchError, StudyBuddyError, TransientLLMError
from .files import LibraryGuard
from .markup import slip_caption

log = logging.getLogger(__name__)

SEARCH_AFTER_SECONDS = 1.0     # nothing yet: say we are working
WAKE_AFTER_SECONDS = 8.0       # still nothing: a warm question starts answering in about 5 s, so this is the model loading
POLL_SECONDS = 0.25
SEARCHING, WAKING, WORKING, STREAMING, DONE, FAILED = "searching", "waking", "working", "streaming", "done", "error"
_END = object()


@dataclass
class TurnUpdate:
    text: str                       # what the assistant message shows right now
    phase: str                      # searching | waking | working | streaming | done | error
    answer: ChatAnswer | None = None   # only on the last update of a successful turn
    elapsed: float = 0.0


def waking_text(elapsed: float) -> str:
    return f"_Waking up the model. This can take a minute ({int(elapsed)} s)._"


def searching_text(elapsed: float) -> str:
    return f"_Looking through your papers ({int(elapsed)} s)._"


def friendly_error(exc: BaseException) -> str:
    """What happened and what to do next, in one or two plain sentences; never a traceback."""
    if isinstance(exc, TransientLLMError):
        return ("I can't reach the language model. Open LM Studio, load the models and start its server, then ask again.")
    if isinstance(exc, IndexMismatchError):
        return ("Your papers were indexed with a different embedding model, so questions can't be answered yet. "
                "Run `python scripts/index_admin.py reindex`, or set the embedding model in `.env` back to the old one.")
    if isinstance(exc, (ChatContextError, StudyBuddyError, ValueError)):
        return str(exc)
    log.error("unexpected chat failure", exc_info=exc)
    return "Something went wrong while answering. Ask again; if it keeps happening, look at the log file (its path is shown when the app starts)."


def stream_turn(session: ChatSession, question: str, subject: str | None, *, wake_after: float = WAKE_AFTER_SECONDS,
                search_after: float = SEARCH_AFTER_SECONDS, poll: float = POLL_SECONDS, clock=time.monotonic) -> Iterator[TurnUpdate]:
    """Ask one question; yield display updates until the last (phase `done` or `error`).

    The subject filter is applied to the session first (an unknown subject is reported, not guessed)."""
    started = clock()
    try:
        session.set_subject(subject)
    except StudyBuddyError as exc:
        yield TurnUpdate(friendly_error(exc), FAILED)
        return

    items: "queue.Queue[object]" = queue.Queue()

    def work() -> None:
        try:
            for item in session.ask_stream(question):
                items.put(item)
        except BaseException as exc:  # reported to the screen below
            items.put(exc)
        finally:
            items.put(_END)

    threading.Thread(target=work, name="chat-turn", daemon=True).start()
    shown = ""
    working = False                 # condensing pages: slow on purpose, not a sleeping model
    while True:
        try:
            item = items.get(timeout=poll)
        except queue.Empty:
            elapsed = clock() - started
            if working:
                continue
            if not shown and elapsed >= wake_after:
                yield TurnUpdate(waking_text(elapsed), WAKING, elapsed=elapsed)
            elif not shown and elapsed >= search_after:
                yield TurnUpdate(searching_text(elapsed), SEARCHING, elapsed=elapsed)
            continue
        elapsed = clock() - started
        if item is _END:
            return
        if isinstance(item, BaseException):
            yield TurnUpdate(friendly_error(item), FAILED, elapsed=elapsed)
            continue                      # the _END marker follows
        if isinstance(item, Progress):
            working = True
            yield TurnUpdate(f"_{item.text}_", WORKING, elapsed=elapsed)
            continue
        if isinstance(item, ChatAnswer):
            yield TurnUpdate(render_answer(item), DONE, item, elapsed)
            continue
        shown += item
        yield TurnUpdate(shown, STREAMING, elapsed=elapsed)


# ---- display text ----------------------------------------------------------------------------

def _body_and_notes(answer: ChatAnswer) -> tuple[str, list[str]]:
    text = answer.text
    if answer.notes:
        joined = "\n".join(answer.notes)
        if text.endswith(joined):
            text = text[: -len(joined)]
    return text.strip(), list(answer.notes)


def render_answer(answer: ChatAnswer) -> str:
    """The final message: the answer, then what the program (not the model) wants the student to know."""
    if answer.status == NEEDS_CHOICE:
        return "Which paper do you mean?" if answer.choices else answer.text
    if answer.status == NOT_IN_PAPERS_STATUS:
        return "I couldn't find that in your papers."
    if answer.status == ERROR:
        return answer.text
    body, notes = _body_and_notes(answer)
    lines = [body]
    for note in notes:
        lines += ["", f"> **Check this:** {note.removeprefix('Note: ')}"]
    if answer.truncated:
        lines += ["", "> **This answer was cut short.** Ask again to get the rest."]
    for warning in answer.warnings:
        lines += ["", f"_{warning}_"]
    return "\n".join(lines)


CHIP_TITLE_CHARS = 30


def _short(text: str, limit: int) -> str:
    text = " ".join((text or "").split())
    if len(text) <= limit:
        return text
    return text[:limit].rsplit(" ", 1)[0].rstrip(",;:-") if " " in text[:limit] else text[:limit]


def chip_label(source: Source) -> str:
    """Short text for a source chip: its number, the paper and the page. Plain text only (a Radio label is not markdown)."""
    return f"{source.number}  {_short(source.title or 'Untitled', CHIP_TITLE_CHARS)}, p. {source.page_number}"


def source_choices(answer: ChatAnswer) -> list[tuple[str, int]]:
    """(chip text, source number), cited sources first. For a refusal these are the closest pages, not cited ones."""
    ordered = sorted(answer.sources, key=lambda s: (not s.cited, s.number))
    return [(chip_label(s), s.number) for s in ordered]


def sources_caption(answer: ChatAnswer) -> str:
    """The small heading above the chips; empty when there is nothing to show."""
    if not answer.sources:
        return ""
    return "Closest pages, probably not what you asked about" if answer.status == NOT_IN_PAPERS_STATUS else "Sources"


def source_details(answer: ChatAnswer, number: int, guard: LibraryGuard) -> tuple[Image.Image | None, str, str]:
    """(page image or None, caption HTML for the slip, text read from the page) for a chosen source."""
    source = next((s for s in answer.sources if s.number == number), None)
    if source is None:
        return None, slip_caption(None, "That source is not in the last answer."), ""
    image, problem = None, None
    try:
        image = guard.render_page(source.image_path, source.page_number)
    except StudyBuddyError as exc:
        problem = f"The page image is not available: {exc}"
    return image, slip_caption(source, problem), source.text
