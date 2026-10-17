"""Turn a follow-up ("and what about its causes?") into a standalone search query using recent history."""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Sequence

from ..errors import LLMError, TransientLLMError
from ..ingest.metadata import Chats

HISTORY_TURNS = 3         # how many recent turns the rewrite may look at
ANSWER_CHARS = 300        # each earlier answer is shortened to this
MAX_QUERY_CHARS = 300
_PREFIX = re.compile(r"^\s*(?:rewritten|standalone)?\s*(?:question|query)\s*[:\-]\s*", re.IGNORECASE)
_CITATION = re.compile(r"\s*[\[【][\d\s,;\-–]+[\]】]")

SYSTEM_PROMPT = (
    "You rewrite a student's follow-up question so that it can be understood on its own, without the conversation. "
    "Use the earlier turns only to replace references such as 'it', 'that', 'those', 'the second one' or a missing topic. "
    "Reply with ONLY the rewritten question on a single line. If the question is already clear on its own, repeat it "
    "unchanged. Never answer the question and never add new information."
)


@dataclass(frozen=True)
class Turn:
    question: str
    answer: str


@dataclass(frozen=True)
class Rewrite:
    query: str          # what to search for
    rewritten: bool     # False: the question was used as it was
    note: str = ""      # why a rewrite was not used, if relevant


def _short(text: str, limit: int) -> str:
    text = _CITATION.sub("", " ".join(text.split()))
    return text if len(text) <= limit else text[:limit].rstrip() + "..."


def build_prompt(history: Sequence[Turn], question: str) -> list[dict]:
    recent = list(history)[-HISTORY_TURNS:]
    lines = []
    for turn in recent:
        lines.append(f"Student: {_short(turn.question, 300)}")
        lines.append(f"Assistant: {_short(turn.answer, ANSWER_CHARS)}")
    conversation = "\n".join(lines)
    return [{"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"Earlier conversation:\n{conversation}\n\nFollow-up question: {question}\n\nRewritten question:"}]


def clean_rewrite(raw: str, question: str) -> str | None:
    """The rewritten question from a model answer, or None if it cannot be trusted."""
    lines = [ln.strip() for ln in (raw or "").splitlines() if ln.strip()]
    if not lines:
        return None
    text = _PREFIX.sub("", lines[0]).strip().strip("\"'`*").strip()
    if not text or len(text) > MAX_QUERY_CHARS or len(text) > 3 * len(question) + 160:
        return None  # empty, or far too long to be a question: the model probably started answering
    if re.match(r"^(i'm sorry|i am sorry|sorry|i cannot|i can't|as an ai)", text, re.IGNORECASE):
        return None
    return text


def rewrite_query(chat: Chats, history: Sequence[Turn], question: str) -> Rewrite:
    """With no history the question is used as it is and no model is called. A transient model failure propagates."""
    if not history:
        return Rewrite(question, False)
    try:
        raw = chat.chat(build_prompt(history, question), temperature=0.0, max_tokens=100)
    except TransientLLMError:
        raise
    except LLMError as exc:
        return Rewrite(question, False, f"could not rewrite the question ({exc}); used it as typed")
    text = clean_rewrite(raw, question)
    if text is None:
        return Rewrite(question, False, "the rewrite was unusable; the question was used as typed")
    if text.casefold() == question.strip().casefold():
        return Rewrite(question, False)
    return Rewrite(text, True)
