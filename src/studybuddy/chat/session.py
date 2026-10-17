"""A chat session: questions are answered only from the student's own papers, with checked citations.

Per question: rewrite a follow-up into a standalone query, retrieve, spread the sources, decide from the raw scores
whether the papers answer it at all (if not, say so without calling the model), fit the prompt in the context
budget, stream the answer, validate the citations, and add the notes the model cannot be trusted to add.
"""
from __future__ import annotations

import re
import time
from dataclasses import dataclass
from typing import Callable, Iterator, Protocol, Sequence

from ..config import Settings
from ..errors import ChatContextError, ContextOverflowError, TransientLLMError
from ..index.index import Index
from ..index.search import Hit
from ..ingest.subjects import SubjectError, SubjectList
from ..llm import StreamEnd
from .answers import ANSWERED, ERROR, NOT_IN_PAPERS_STATUS, ChatAnswer, Source
from .budget import TokenEstimator, minimum_context, plan_budget, resolve_context_tokens, too_small
from .citations import CitationFilter
from .expand import expand_query, fuse_searches
from .gate import Assessment, GateConfig, assess, coverage
from .prompt import NOT_IN_PAPERS, SYSTEM_PROMPT, build_messages, format_source
from .rewrite import Turn, rewrite_query
from .select import PER_DOCUMENT, PER_PAGE, SOURCE_COUNT, spread

REFUSAL_TEXT = "Your papers don't seem to cover this question, so I won't guess."
CLOSEST_SOURCES = 3
LIKELY_SOURCE_COVERAGE = 0.6   # share of the answer's content words a source must contain to be named as the likely one
HANDWRITING_QUESTION = re.compile(
    r"\b(name|names|date|dates|score|scores|section|teacher|grade|answer key|which letter|what letter|my answers?|i answered|who wrote)\b",
    re.IGNORECASE)
HANDWRITING_NOTE = "Handwritten names, dates, scores and answer letters may be read wrongly: check them against the photo."


class ChatClient(Protocol):
    def chat(self, messages: list[dict], temperature: float = 0.0, max_tokens: int | None = None) -> str: ...
    def chat_stream(self, messages: list[dict], temperature: float = 0.0, max_tokens: int | None = None) -> Iterator: ...
    def context_length(self, model: str | None = None) -> int | None: ...


@dataclass
class ChatConfig:
    max_answer_tokens: int = 700
    context_tokens: int | None = None        # None: ask LM Studio (or SB_CHAT_CONTEXT_TOKENS)
    source_count: int = SOURCE_COUNT
    pool: int = 20
    per_page: int = PER_PAGE
    per_document: int = PER_DOCUMENT
    history_turns: int = 6
    expand_queries: bool = False

    @classmethod
    def from_settings(cls, settings: Settings, **overrides) -> "ChatConfig":
        options = dict(max_answer_tokens=settings.chat_max_answer_tokens, context_tokens=settings.chat_context_tokens,
                       expand_queries=settings.chat_expand_queries)
        return cls(**{**options, **overrides})


class _RefusalWatch:
    """Holds back the first few characters of an answer, because a reply that is exactly NOT_IN_PAPERS must not be shown."""

    def __init__(self) -> None:
        self.text = ""
        self.released = False

    def feed(self, piece: str) -> tuple[str, bool]:
        """Returns (text safe to show now, is_refusal)."""
        self.text += piece
        if self.released:
            return piece, False
        head = self.text.lstrip()
        if NOT_IN_PAPERS.startswith(head) and len(head) < len(NOT_IN_PAPERS):
            return "", False                       # could still become NOT_IN_PAPERS: wait
        if head.startswith(NOT_IN_PAPERS):
            return "", True
        self.released = True
        return self.text, False


class ChatSession:
    def __init__(self, index: Index, client: ChatClient, settings: Settings, subjects: SubjectList | None = None,
                 config: ChatConfig | None = None, gate: GateConfig | None = None,
                 clock: Callable[[], float] = time.monotonic):
        self.index, self.client, self.settings = index, client, settings
        self.subjects = subjects
        self.config = config or ChatConfig.from_settings(settings)
        self.gate = gate or GateConfig()
        self.clock = clock
        self.estimator = TokenEstimator()
        self.history: list[Turn] = []
        self.last_answer: ChatAnswer | None = None
        self._subject: str | None = None
        self._context: int | None = self.config.context_tokens

    # ---- state -----------------------------------------------------------------------------

    @property
    def subject(self) -> str | None:
        return self._subject

    def set_subject(self, name: str | None) -> str | None:
        """Restrict answers to one subject (case-insensitive, must be in the subject list); None clears it."""
        if not name:
            self._subject = None
        elif self.subjects is None:
            self._subject = name.strip()
        else:
            known = self.subjects.find(name)
            if known is None:
                raise SubjectError(f"Unknown subject {name!r}. Known: {', '.join(self.subjects.choices())}")
            self._subject = known
        return self._subject

    def new(self) -> None:
        """Forget the conversation (the subject filter stays)."""
        self.history.clear()
        self.last_answer = None

    def context_tokens(self) -> int:
        if self._context is None:
            self._context = resolve_context_tokens(self.client, self.settings)
        return self._context

    # ---- asking ----------------------------------------------------------------------------

    def ask(self, question: str, on_token: Callable[[str], None] | None = None) -> ChatAnswer:
        final: ChatAnswer | None = None
        for item in self.ask_stream(question):
            if isinstance(item, ChatAnswer):
                final = item
            elif on_token:
                on_token(item)
        assert final is not None
        return final

    def ask_stream(self, question: str) -> Iterator[str | ChatAnswer]:
        """Yield the answer text piece by piece, then the final ChatAnswer."""
        question = " ".join(question.split())
        if not question:
            raise ValueError("The question is empty.")
        started = self.clock()
        reserve = self.config.max_answer_tokens
        context = self.context_tokens()
        system_tokens = self.estimator.estimate(SYSTEM_PROMPT)
        question_tokens = self.estimator.estimate(question)
        needed = minimum_context(system_tokens, question_tokens, reserve, self.estimator)
        if context < needed:
            raise too_small(context, needed, reserve)

        rewrite = rewrite_query(self.client, self.history[-self.config.history_turns:], question)
        hits = self._retrieve(rewrite.query)
        chosen = spread(hits, self.config.source_count, self.config.per_page, self.config.per_document)
        assessment = assess(chosen, rewrite.query, self.gate)
        answer = ChatAnswer(text="", status=ANSWERED, query=rewrite.query, rewritten=rewrite.rewritten,
                            subject_filter=self._subject, context_tokens=context)
        if rewrite.note:
            answer.warnings.append(rewrite.note)

        if not assessment.answerable:
            yield from self._refuse(question, answer, chosen, assessment.reason, started)
            return

        sources, history = self._fit(chosen, question, system_tokens, question_tokens, context, reserve, answer)
        answer.sources = sources
        messages = build_messages(question, sources, history)
        yield from self._stream_answer(question, messages, answer, started)

    # ---- steps -----------------------------------------------------------------------------

    def _retrieve(self, query: str) -> list[Hit]:
        if self.config.expand_queries:
            phrasings = expand_query(self.client, query)
            if phrasings:
                return fuse_searches(self.index, [query, *phrasings], self.config.pool, self._subject)
        return self.index.search(query, k=self.config.pool, subject=self._subject, mode="hybrid")

    def _fit(self, chosen: Sequence[Hit], question: str, system_tokens: int, question_tokens: int, context: int,
             reserve: int, answer: ChatAnswer) -> tuple[list[Source], list[Turn]]:
        sources = [Source.from_hit(i, h) for i, h in enumerate(chosen, 1)]
        history = self.history[-self.config.history_turns:]
        history_cost = [self.estimator.estimate(t.question) + self.estimator.estimate(t.answer) for t in history]
        source_cost = [self.estimator.estimate(format_source(s)) for s in sources]
        plan = plan_budget(context, reserve, system_tokens, question_tokens, history_cost, source_cost, self.estimator)
        if plan.first_source_chars is not None:
            sources[0].text = sources[0].text[: max(200, plan.first_source_chars - 200)].rstrip() + " ..."
            answer.warnings.append("the best source was shortened to fit the model's context")
        if plan.sources_kept < len(sources):
            answer.warnings.append(f"{len(sources) - plan.sources_kept} lower-ranked source(s) left out to fit the model's context")
        if plan.history_kept < len(history):
            answer.warnings.append(f"{len(history) - plan.history_kept} older turn(s) left out of the conversation to fit the context")
        return sources[: plan.sources_kept], history[len(history) - plan.history_kept:] if plan.history_kept else []

    def _refuse(self, question: str, answer: ChatAnswer, chosen: Sequence[Hit], reason: str, started: float) -> Iterator[ChatAnswer | str]:
        answer.status = NOT_IN_PAPERS_STATUS
        answer.refusal_reason = reason
        answer.sources = [Source.from_hit(i, h) for i, h in enumerate(list(chosen)[:CLOSEST_SOURCES], 1)]
        text = REFUSAL_TEXT
        if self._subject:
            text += f" (I only looked in {self._subject}.)"
        answer.text = text
        answer.total_seconds = round(self.clock() - started, 2)
        self._remember(question, text, answer)
        yield text
        yield answer

    def _stream_answer(self, question: str, messages: list[dict], answer: ChatAnswer, started: float) -> Iterator[str | ChatAnswer]:
        citations = CitationFilter(len(answer.sources))
        watch = _RefusalWatch()
        shown: list[str] = []
        usage: dict | None = None
        finish: str | None = None
        refused = False
        prompt_chars = sum(len(m["content"]) for m in messages)
        try:
            for item in self.client.chat_stream(messages, temperature=0.0, max_tokens=self.config.max_answer_tokens):
                if isinstance(item, StreamEnd):
                    usage, finish = item.usage, item.finish_reason
                    continue
                if answer.first_token_seconds is None:
                    answer.first_token_seconds = round(self.clock() - started, 2)
                text, refusal = watch.feed(item)
                if refusal:
                    refused = True
                    continue
                if text and not refused:
                    cleaned = citations.feed(text)
                    if cleaned:
                        shown.append(cleaned)
                        yield cleaned
        except ContextOverflowError as exc:
            raise ChatContextError(f"The model's context overflowed even though the prompt was budgeted: {exc}. "
                                   f"Raise the context length in LM Studio or lower SB_CHAT_MAX_ANSWER_TOKENS.") from exc
        except TransientLLMError as exc:
            if not "".join(shown).strip():
                raise
            answer.truncated = True
            answer.warnings.append(f"the answer is incomplete: {exc}")
        if not refused and not watch.released and watch.text:  # a short reply that stayed a possible NOT_IN_PAPERS prefix
            cleaned = citations.feed(watch.text)
            if cleaned:
                shown.append(cleaned)
                yield cleaned
        tail = "" if refused else citations.flush()
        if tail:
            shown.append(tail)
            yield tail

        if usage:
            answer.prompt_tokens, answer.completion_tokens = usage.get("prompt_tokens"), usage.get("completion_tokens")
            self.estimator.observe(prompt_chars, answer.prompt_tokens)
        if refused:
            answer.total_seconds = round(self.clock() - started, 2)
            answer.status, answer.refusal_reason = NOT_IN_PAPERS_STATUS, "the model found no answer in the sources"
            answer.text = REFUSAL_TEXT + (f" (I only looked in {self._subject}.)" if self._subject else "")
            yield answer.text
            self._remember(question, answer.text, answer)
            yield answer
            return

        body = "".join(shown).strip()
        if not body:
            answer.status = ERROR
            answer.text = "I couldn't produce an answer this time. Please ask again."
            answer.warnings.append("the model returned an empty answer")
            answer.total_seconds = round(self.clock() - started, 2)
            yield answer.text
            self.last_answer = answer  # not added to the history: there is nothing to follow up on
            yield answer
            return
        if finish == "length":
            answer.truncated = True
            answer.warnings.append("the answer was cut off at the length limit")
            body += " ... (the answer was cut off)"
        answer.cited, answer.invalid_citations = citations.cited, citations.invalid
        for source in answer.sources:
            source.cited = source.number in answer.cited
        answer.uncited = not answer.cited
        if answer.invalid_citations:
            answer.warnings.append(f"removed citation(s) that point at no source: {answer.invalid_citations}")
        notes = self._notes(question, answer, body)
        answer.notes = notes
        answer.text = body + ("\n\n" + "\n".join(notes) if notes else "")
        if notes:
            yield "\n\n" + "\n".join(notes)
        answer.total_seconds = round(self.clock() - started, 2)
        self._remember(question, body, answer)
        yield answer

    def _notes(self, question: str, answer: ChatAnswer, body: str = "") -> list[str]:
        notes: list[str] = []
        cited = answer.cited_sources
        for source in cited:
            if source.hard_to_read:
                why = f" ({'; '.join(source.problems)})" if source.problems else ""
                notes.append(f"Note: page {source.page_number} of '{source.title or source.file_path}' was hard to read, check the photo{why}.")
        if answer.uncited:
            likely = self._likely_source(body, answer.sources)
            if likely is not None:
                answer.likely_source = likely.number
                notes.append(f"Note: no source was cited; the wording matches SOURCE {likely.number} "
                             f"('{likely.title or likely.file_path}', page {likely.page_number}). Check it against that page.")
            else:
                notes.append("Note: no source was cited for this answer, so check it against your papers.")
        if cited and HANDWRITING_QUESTION.search(question):  # the question is about the kind of field that is handwritten
            notes.append(HANDWRITING_NOTE)
        return notes

    @staticmethod
    def _likely_source(body: str, sources: Sequence[Source]) -> Source | None:
        """For an answer that cited nothing: the one source whose text contains most of the answer's content words.

        Deterministic and shown only as a hint ("the wording matches"), never as a citation."""
        if not body.strip() or not sources:
            return None
        scored = sorted(((coverage(body, s.text), -s.number, s) for s in sources), key=lambda x: (x[0], x[1]), reverse=True)
        best = scored[0]
        runner_up = scored[1][0] if len(scored) > 1 else 0.0
        return best[2] if best[0] >= LIKELY_SOURCE_COVERAGE and best[0] > runner_up else None

    def _remember(self, question: str, text: str, answer: ChatAnswer) -> None:
        self.history.append(Turn(question, text))
        del self.history[: -self.config.history_turns]
        self.last_answer = answer
