"""Token budget for a chat prompt, so it never overflows the model's loaded context.

There is no tokenizer endpoint, so tokens are estimated from characters, starting conservatively (3 characters per
token) and tightened from the `prompt_tokens` the server reports. The plan keeps the system prompt and the question,
gives history at most a quarter of what is left (dropping the OLDEST turns first), and gives the rest to sources
(dropping the LOWEST-ranked first). If not even one source fits, it fails clearly instead of sending a cut prompt.
"""
from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass
from typing import Protocol, Sequence

from ..config import Settings
from ..errors import ChatContextError

START_CHARS_PER_TOKEN = 3.0
MIN_CHARS_PER_TOKEN, MAX_CHARS_PER_TOKEN = 2.0, 5.0
MESSAGE_OVERHEAD_TOKENS = 8      # role markers and separators per message
MIN_SOURCE_CHARS = 400           # a single source may be shortened to this before we give up
HISTORY_FRACTION = 0.25


class ContextSource(Protocol):
    def context_length(self, model: str | None = None) -> int | None: ...


class TokenEstimator:
    def __init__(self) -> None:
        self._observed: deque[float] = deque(maxlen=5)

    @property
    def chars_per_token(self) -> float:
        """3.0 until the server has reported real counts, then the most pessimistic of the last few (within 2-5)."""
        if not self._observed:
            return START_CHARS_PER_TOKEN
        return max(MIN_CHARS_PER_TOKEN, min(MAX_CHARS_PER_TOKEN, min(self._observed)))

    def estimate(self, text: str) -> int:
        return math.ceil(len(text) / self.chars_per_token) + MESSAGE_OVERHEAD_TOKENS

    def observe(self, prompt_chars: int, prompt_tokens: int | None) -> None:
        if prompt_tokens and prompt_tokens > 0 and prompt_chars > 0:
            self._observed.append(prompt_chars / prompt_tokens)

    def chars_for(self, tokens: int) -> int:
        return max(0, int((tokens - MESSAGE_OVERHEAD_TOKENS) * self.chars_per_token))


@dataclass(frozen=True)
class BudgetPlan:
    history_kept: int            # how many of the most recent turns fit
    sources_kept: int            # how many of the top-ranked sources fit
    first_source_chars: int | None  # set when the top source had to be shortened to fit
    available_tokens: int        # what was left for history + sources
    used_tokens: int             # estimated total prompt tokens including the answer reserve


def minimum_context(system_tokens: int, question_tokens: int, answer_reserve: int, estimator: TokenEstimator) -> int:
    """The smallest context in which chat can work: prompt basics, a shortened source and the answer."""
    return system_tokens + question_tokens + answer_reserve + estimator.estimate("x" * MIN_SOURCE_CHARS)


def too_small(context_tokens: int, needed: int, answer_reserve: int) -> ChatContextError:
    return ChatContextError(
        f"The loaded context is {context_tokens} tokens but chat needs at least {needed} (the instructions, your question, "
        f"room for one source and a {answer_reserve}-token answer). Raise the context length of the chat model in LM Studio, "
        f"or lower SB_CHAT_MAX_ANSWER_TOKENS.")


def resolve_context_tokens(client: ContextSource, settings: Settings) -> int:
    """SB_CHAT_CONTEXT_TOKENS if set, otherwise what LM Studio reports for the chat model."""
    if settings.chat_context_tokens:
        return settings.chat_context_tokens
    reported = client.context_length(settings.chat_model)
    if not reported:
        raise ChatContextError(
            f"Could not find out the context size of {settings.chat_model!r} from LM Studio. Load the model there, "
            f"or set SB_CHAT_CONTEXT_TOKENS in .env.")
    return reported


def plan_budget(context_tokens: int, answer_reserve: int, system_tokens: int, question_tokens: int,
                history_tokens: Sequence[int], source_tokens: Sequence[int], estimator: TokenEstimator,
                history_fraction: float = HISTORY_FRACTION) -> BudgetPlan:
    """Decide how much history (newest first) and how many sources (best first) fit.

    `history_tokens` are oldest to newest; `source_tokens` best to worst. Raises ChatContextError if not even a
    shortened top source fits."""
    available = context_tokens - answer_reserve - system_tokens - question_tokens
    minimum = estimator.estimate("x" * MIN_SOURCE_CHARS)
    if available < minimum or not source_tokens:
        if not source_tokens:
            raise ValueError("plan_budget needs at least one source")
        raise too_small(context_tokens, system_tokens + question_tokens + answer_reserve + minimum, answer_reserve)

    # history first, capped at a quarter of what is available; the newest turns win
    allowed, used_history, kept_history = int(available * history_fraction), 0, 0
    for cost in reversed(history_tokens):
        if used_history + cost > allowed:
            break
        used_history += cost
        kept_history += 1

    remaining = available - used_history
    if remaining < minimum:  # history must never crowd out the one source we cannot answer without
        used_history, kept_history, remaining = 0, 0, available
    kept_sources, used_sources, shortened = 0, 0, None
    for index, cost in enumerate(source_tokens):
        if used_sources + cost <= remaining:
            used_sources += cost
            kept_sources += 1
        elif index == 0:  # the best source alone does not fit whole: shorten it, never drop it
            shortened = estimator.chars_for(remaining)
            used_sources += remaining
            kept_sources = 1
            break
        else:
            break  # the lower-ranked ones are dropped
    used = context_tokens - available + used_history + used_sources
    return BudgetPlan(kept_history, kept_sources, shortened, available, used)
