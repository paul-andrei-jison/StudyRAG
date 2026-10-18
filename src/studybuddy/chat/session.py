"""A chat session: questions are answered only from the student's own papers, with checked citations.

Per question: rewrite a follow-up into a standalone query, retrieve, spread the sources, decide from the raw scores
whether the papers answer it at all (if not, say so without calling the model), fit the prompt in the context
budget, stream the answer, validate the citations, and add the notes the model cannot be trusted to add.
"""
from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass
from typing import Callable, Iterator, Protocol, Sequence

from ..config import Settings
from ..errors import ChatContextError, ContextOverflowError, TransientLLMError
from ..index.index import Index
from ..index.search import Hit
from ..index.store import DocumentRow
from ..ingest.results import IGNORED_FLAGS
from ..ingest.subjects import SubjectError, SubjectList
from ..llm import StreamEnd
from .answers import ANSWERED, ERROR, NEEDS_CHOICE, NOT_IN_PAPERS_STATUS, OVERVIEW, ChatAnswer, Progress, Source
from .budget import TokenEstimator, minimum_context, plan_budget, resolve_context_tokens, too_small
from .condense import CONDENSED, SUMMARY, condense_page, plan_document
from .citations import CitationFilter
from .expand import expand_query, fuse_searches
from .gate import GateConfig, assess, coverage
from .overview import detect
from .prompt import NOT_IN_PAPERS, OVERVIEW_SYSTEM_PROMPT, SYSTEM_PROMPT, build_messages, format_source
from .rewrite import Turn, rewrite_query
from .select import PER_DOCUMENT, PER_PAGE, SOURCE_COUNT, spread
from .targets import ASK, DOCUMENTS, Resolution, match_choice, resolve_targets

REFUSAL_TEXT = "Your papers don't seem to cover this question, so I won't guess."
CLOSEST_SOURCES = 3
LIKELY_SOURCE_COVERAGE = 0.6   # share of the answer's content words a source must contain to be named as the likely one
HANDWRITING_QUESTION = re.compile(
    r"\b(name|names|date|dates|score|scores|section|teacher|grade|answer key|which letter|what letter|my answers?|i answered|who wrote)\b",
    re.IGNORECASE)
MIN_DOC_TOKENS = 200        # each document in an overview needs at least this much room in the context
HEADING_RESERVE = 60
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
        self._pending: tuple[str, Resolution] | None = None   # an overview question waiting for the answer to "which paper?"

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
        self._pending = None

    def context_tokens(self) -> int:
        if self._context is None:
            try:
                self._context = resolve_context_tokens(self.client, self.settings)
            except ChatContextError:
                # LM Studio only reports a LOADED model's context. If it was unloaded (idle for long), a tiny request
                # makes it load the model; if LM Studio is off that raises TransientLLMError, which is the right message.
                self.client.chat([{"role": "user", "content": "Reply with OK."}], temperature=0.0, max_tokens=4)
                self._context = resolve_context_tokens(self.client, self.settings)
        return self._context

    # ---- asking ----------------------------------------------------------------------------

    def ask(self, question: str, on_token: Callable[[str], None] | None = None,
            on_progress: Callable[[str], None] | None = None) -> ChatAnswer:
        final: ChatAnswer | None = None
        for item in self.ask_stream(question):
            if isinstance(item, ChatAnswer):
                final = item
            elif isinstance(item, Progress):
                if on_progress:
                    on_progress(item.text)
            elif on_token:
                on_token(item)
        assert final is not None
        return final

    def ask_stream(self, question: str) -> Iterator[str | Progress | ChatAnswer]:
        """Yield the answer text piece by piece (and Progress notices while pages are condensed), then the final ChatAnswer."""
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

        target = None
        asked_question = question
        if self._pending is not None:
            asked_question, pending = self._pending
            choice = match_choice(question, pending)
            if choice.kind == "chosen":
                self._pending = None
                target = Resolution(DOCUMENTS, [choice.doc], "chosen")
            elif choice.ask is not None:     # still not clear which paper: ask again (same list, or the narrower one)
                self._pending = (asked_question, choice.ask)
                yield from self._ask_which(question, choice.ask, started, context)
                return
            else:
                self._pending = None         # not an answer to the question: carry on as an ordinary question
                asked_question = question

        rewrite = rewrite_query(self.client, self.history[-self.config.history_turns:], question)
        if target is None:
            found = detect(question)
            if not found.is_overview and rewrite.rewritten:
                found = detect(rewrite.query)
            if found.is_overview:
                resolved = resolve_targets(found, f"{question} {rewrite.query}", self.index.documents(), self._subject)
                if resolved.kind == ASK:
                    self._pending = (question, resolved)
                    yield from self._ask_which(question, resolved, started, context)
                    return
                if resolved.kind == DOCUMENTS:
                    target = resolved
        if target is not None:
            yield from self._overview(asked_question, target, context, reserve, started)
            return
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

    # ---- overview questions: answered from whole documents ---------------------------------------

    def _ask_which(self, question: str, asking: Resolution, started: float, context: int) -> Iterator[str | ChatAnswer]:
        answer = ChatAnswer(text=asking.text, status=NEEDS_CHOICE, query=question, subject_filter=self._subject,
                            context_tokens=context, mode=OVERVIEW)
        answer.total_seconds = round(self.clock() - started, 2)
        self.last_answer = answer
        yield asking.text
        yield answer

    def _overview(self, question: str, target: Resolution, context: int, reserve: int,
                  started: float) -> Iterator[str | Progress | ChatAnswer]:
        """Answer from the pages of the chosen documents in order; no gate (the question is about whole documents, not a fact)."""
        answer = ChatAnswer(text="", status=ANSWERED, query=question, subject_filter=self._subject, context_tokens=context,
                            mode=OVERVIEW)
        system_tokens = self.estimator.estimate(OVERVIEW_SYSTEM_PROMPT)
        available = context - reserve - system_tokens - self.estimator.estimate(question)
        if available < len(target.docs) * MIN_DOC_TOKENS:
            raise too_small(context, reserve + system_tokens + len(target.docs) * MIN_DOC_TOKENS, reserve)
        share = available // len(target.docs)
        sources: list[Source] = []
        for doc in target.docs:
            pages = self.index.store.pages(doc.id)
            if not pages:
                continue
            drafts = [self._page_source(doc, p) for p in pages]
            plan = plan_document(share, [self.estimator.estimate(format_source(d)) for d in drafts], self.estimator)
            if plan.tier == CONDENSED:
                answer.overview_notes.append(
                    f"'{doc.title}' has {len(drafts)} pages, so I summarized each page first, then combined them; fine details may be missing.")
                fallbacks = 0
                for number, draft in enumerate(drafts, 1):
                    yield Progress(f"Summarizing page {number} of {len(drafts)} of '{doc.title}' first...")
                    result = condense_page(self.client, draft.text, plan.page_chars, self.estimator)
                    draft.text = result.text
                    answer.model_calls_condensing += result.called
                    fallbacks += result.fell_back
                if fallbacks:
                    answer.overview_notes.append(f"{fallbacks} page(s) of '{doc.title}' could not be summarized and were only shortened.")
            elif plan.tier == SUMMARY:
                text = (doc.summary or "").strip() or drafts[0].text
                limit = self.estimator.chars_for(share - HEADING_RESERVE)
                drafts = drafts[:1]
                drafts[0].text = text if len(text) <= limit else text[:limit].rstrip() + " ..."
                answer.overview_notes.append(
                    f"'{doc.title}' is too long to read at once ({doc.page_count} pages), so this uses its stored summary only; "
                    f"details are missing. Ask about a specific part for more.")
            sources.extend(drafts)
        if not sources:
            yield from self._refuse(question, answer, [], "the chosen papers have no readable pages", started)
            return
        for number, source in enumerate(sources, 1):
            source.number = number
        titles = "; ".join(f"'{d.title or d.file_path}' ({d.page_count} page{'s' if d.page_count != 1 else ''})" for d in target.docs)
        answer.overview_notes.insert(0, f"This is an overview of {titles}, not an answer found on one page.")
        answer.sources = sources
        messages = build_messages(question, sources, [], system=OVERVIEW_SYSTEM_PROMPT)
        yield from self._stream_answer(question, messages, answer, started)

    @staticmethod
    def _page_source(doc: DocumentRow, page) -> Source:
        try:
            flags = [f for f in json.loads(page["flags"] or "[]") if f not in IGNORED_FLAGS]
        except ValueError:
            flags = []
        return Source(0, doc.subject, doc.title, doc.file_path, page["page_number"], page["image_path"], page["text"] or "",
                      page["quality"], flags)

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
        notes: list[str] = list(answer.overview_notes)
        cited = answer.sources if answer.mode == OVERVIEW else answer.cited_sources   # a summary draws on every page it was given
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
        if answer.cited and HANDWRITING_QUESTION.search(question):  # the question is about the kind of field that is handwritten
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
