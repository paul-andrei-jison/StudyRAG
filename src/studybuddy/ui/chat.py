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
    return f"_Waking the model, this can take a minute... ({int(elapsed)} s)_"


def searching_text(elapsed: float) -> str:
    return f"_Searching your papers... ({int(elapsed)} s)_"


def friendly_error(exc: BaseException) -> str:
    """One plain sentence for the student; never a traceback."""
    if isinstance(exc, TransientLLMError):
        return (f"I can't reach the language model right now. Check that LM Studio is running with the models loaded, "
                f"then ask again. ({exc})")
    if isinstance(exc, IndexMismatchError):
        return (f"The search index was built with a different embedding model, so questions can't be answered until it is "
                f"rebuilt: run `python scripts/index_admin.py reindex`. ({exc})")
    if isinstance(exc, ChatContextError):
        return str(exc)
    if isinstance(exc, StudyBuddyError):
        return str(exc)
    if isinstance(exc, ValueError):
        return str(exc)
    log.error("unexpected chat failure", exc_info=exc)
    return f"Something went wrong ({type(exc).__name__}: {exc}). Please ask again."


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
        return answer.text
    if answer.status == NOT_IN_PAPERS_STATUS:
        lines = ["**Not in your papers.**", "", answer.text]
        if answer.refusal_reason:
            lines += ["", f"_Why: {answer.refusal_reason}._"]
        if answer.sources:
            lines += ["", "The pages below are the closest ones, but they are **probably not relevant**."]
        return "\n".join(lines)
    if answer.status == ERROR:
        return f"**No answer this time.** {answer.text}"
    body, notes = _body_and_notes(answer)
    lines = [body]
    for note in notes:
        lines += ["", f"> **Check this:** {note.removeprefix('Note: ')}"]
    if answer.truncated:
        lines += ["", "> **Incomplete:** this answer was cut short."]
    for warning in answer.warnings:
        lines += ["", f"_Heads up: {warning}._"]
    return "\n".join(lines)


def source_label(source: Source) -> str:
    marks = []
    if source.cited:
        marks.append("cited")
    if source.hard_to_read:
        marks.append("hard to read")
    suffix = f"  ({', '.join(marks)})" if marks else ""
    return f"[{source.number}] {source.subject or 'no subject'} | {source.title or '(untitled)'} | page {source.page_number}{suffix}"


def source_choices(answer: ChatAnswer) -> list[tuple[str, int]]:
    """(label, source number) for the sources list, cited sources first. A refusal lists the closest pages as such."""
    ordered = sorted(answer.sources, key=lambda s: (not s.cited, s.number))
    return [(source_label(s), s.number) for s in ordered]


def sources_caption(answer: ChatAnswer) -> str:
    if not answer.sources:
        return "No sources."
    if answer.status == NOT_IN_PAPERS_STATUS:
        return "Closest pages (probably not relevant) - click one to see it"
    return "Sources - click one to see the page"


def source_details(answer: ChatAnswer, number: int, guard: LibraryGuard) -> tuple[Image.Image | None, str, str]:
    """(page image or None, file/page description, text read from the page) for a clicked source."""
    source = next((s for s in answer.sources if s.number == number), None)
    if source is None:
        return None, "That source is not in the last answer.", ""
    lines = [f"File: {source.file_path}", f"Page: {source.page_number}", f"Subject: {source.subject or '-'}  |  Title: {source.title or '-'}"]
    if source.cited:
        lines.append("This page was cited in the answer.")
    if source.hard_to_read:
        why = "; ".join(source.problems)
        lines.append(f"Hard to read{': ' + why if why else ''}. Check the photo.")
    image = None
    try:
        image = guard.render_page(source.image_path, source.page_number)
    except StudyBuddyError as exc:
        lines.append(f"Page image not shown: {exc}")
    return image, "\n".join(lines), source.text
