import dataclasses

import pytest

from studybuddy.chat.budget import (
    MESSAGE_OVERHEAD_TOKENS, MIN_SOURCE_CHARS, TokenEstimator, minimum_context, plan_budget, resolve_context_tokens,
)
from studybuddy.config import load_settings
from studybuddy.errors import ChatContextError, ConfigError


def test_estimator_is_conservative_until_it_has_seen_real_counts():
    e = TokenEstimator()
    assert e.chars_per_token == 3.0
    assert e.estimate("x" * 300) == 100 + MESSAGE_OVERHEAD_TOKENS
    assert e.estimate("") == MESSAGE_OVERHEAD_TOKENS


def test_estimator_tightens_to_the_servers_numbers_within_bounds():
    e = TokenEstimator()
    e.observe(4000, 1000)           # 4 characters per token observed
    assert e.chars_per_token == 4.0
    e.observe(2400, 1000)           # a worse prompt (2.4): the more pessimistic one wins
    assert e.chars_per_token == 2.4
    e.observe(100, 1000)            # absurdly low: clamped
    assert e.chars_per_token == 2.0
    e2 = TokenEstimator()
    e2.observe(100_000, 1000)       # absurdly high: clamped
    assert e2.chars_per_token == 5.0


def test_estimator_ignores_unusable_observations():
    e = TokenEstimator()
    for chars, tokens in ((0, 10), (100, 0), (100, None), (100, -5)):
        e.observe(chars, tokens)
    assert e.chars_per_token == 3.0


def test_estimator_forgets_old_observations():
    e = TokenEstimator()
    e.observe(2200, 1000)
    for _ in range(5):
        e.observe(4000, 1000)
    assert e.chars_per_token == 4.0  # the bad prompt is no longer among the last five


def test_chars_for_inverts_estimate():
    e = TokenEstimator()
    assert e.estimate("x" * e.chars_for(500)) <= 500
    assert e.chars_for(5) == 0


def plan(context=13_721, answer=700, system=300, question=40, history=(), sources=(450,) * 5, e=None):
    return plan_budget(context, answer, system, question, list(history), list(sources), e or TokenEstimator())


def test_everything_fits_in_a_big_context():
    p = plan(history=(200, 200, 200))
    assert (p.history_kept, p.sources_kept, p.first_source_chars) == (3, 5, None)
    assert p.used_tokens < 13_721 and p.available_tokens == 13_721 - 700 - 300 - 40


def test_history_is_capped_at_a_quarter_and_the_oldest_turns_go_first():
    # available = 4000 - 700 - 300 - 40 = 2960; a quarter = 740
    p = plan(context=4000, history=(400, 300, 300, 300), sources=(450,) * 3)
    assert p.history_kept == 2  # the newest two (300 + 300 = 600); adding a third would pass 740
    p2 = plan(context=4000, history=(900,), sources=(450,) * 3)
    assert p2.history_kept == 0  # one turn bigger than the cap is dropped entirely


def test_lowest_ranked_sources_are_dropped_first():
    p = plan(context=2400, answer=700, system=300, question=40, sources=(500, 480, 460, 440, 420))
    # available = 1360 -> 500 + 480 = 980 fits, adding 460 would not
    assert p.sources_kept == 2 and p.first_source_chars is None


def test_the_top_source_is_shortened_not_dropped_when_it_alone_is_too_big():
    p = plan(context=1500, answer=500, system=300, question=40, sources=(900, 400))
    assert p.sources_kept == 1 and p.first_source_chars is not None and p.first_source_chars >= MIN_SOURCE_CHARS


def test_history_never_squeezes_out_the_one_source_we_need():
    e = TokenEstimator()
    minimum = e.estimate("x" * MIN_SOURCE_CHARS)
    context = 500 + 300 + 40 + minimum + 20  # available barely above the minimum: any history would starve the source
    p = plan(context=context, answer=500, system=300, question=40, history=(30,), sources=(900,))
    assert p.history_kept == 0 and p.sources_kept == 1 and p.first_source_chars >= MIN_SOURCE_CHARS


def test_a_context_that_is_too_small_fails_clearly_with_numbers():
    with pytest.raises(ChatContextError) as info:
        plan(context=1000, answer=700, system=300, question=40)
    message = str(info.value)
    assert "1000 tokens" in message and "at least" in message and "LM Studio" in message and "SB_CHAT_MAX_ANSWER_TOKENS" in message


def test_the_minimum_context_is_exactly_the_smallest_that_works():
    e = TokenEstimator()
    needed = minimum_context(300, 40, 700, e)
    assert plan(context=needed, sources=(900,)).sources_kept == 1
    with pytest.raises(ChatContextError):
        plan(context=needed - 1, sources=(900,))


def test_a_tiny_or_huge_question_is_handled():
    with pytest.raises(ChatContextError):
        plan(context=4000, question=3500)
    assert plan(question=0).sources_kept == 5


def test_a_better_estimator_lets_more_fit():
    tight, loose = TokenEstimator(), TokenEstimator()
    loose.observe(5000, 1000)
    chars = 3000
    context = 2000
    assert tight.estimate("x" * chars) > loose.estimate("x" * chars)
    a = plan_budget(context, 400, 200, 40, [], [tight.estimate("x" * chars)] * 3, tight)
    b = plan_budget(context, 400, 200, 40, [], [loose.estimate("x" * chars)] * 3, loose)
    assert b.sources_kept >= a.sources_kept


def test_plan_needs_a_source():
    with pytest.raises(ValueError):
        plan(sources=())


# ---- resolving the context size ------------------------------------------------------------

class Client:
    def __init__(self, value):
        self.value, self.asked = value, []

    def context_length(self, model=None):
        self.asked.append(model)
        return self.value


def test_context_comes_from_lm_studio(settings):
    c = Client(13_721)
    assert resolve_context_tokens(c, settings) == 13_721 and c.asked == ["chat-m"]


def test_the_override_wins_and_needs_no_lookup(settings):
    c = Client(13_721)
    assert resolve_context_tokens(c, dataclasses.replace(settings, chat_context_tokens=8192)) == 8192 and c.asked == []


def test_unknown_context_is_a_clear_error(settings):
    with pytest.raises(ChatContextError, match="SB_CHAT_CONTEXT_TOKENS"):
        resolve_context_tokens(Client(None), settings)


def test_chat_settings_come_from_the_environment(tmp_path):
    env = {"SB_READER_MODEL": "r", "SB_CHAT_MODEL": "c", "SB_EMBED_MODEL": "e",
           "SB_LIBRARY_DIR": str(tmp_path / "l"), "SB_DATA_DIR": str(tmp_path / "d")}
    defaults = load_settings(env, repo_root=tmp_path / "repo")
    assert defaults.chat_context_tokens is None and defaults.chat_max_answer_tokens == 700
    custom = load_settings({**env, "SB_CHAT_CONTEXT_TOKENS": "8192", "SB_CHAT_MAX_ANSWER_TOKENS": "500"}, repo_root=tmp_path / "repo")
    assert (custom.chat_context_tokens, custom.chat_max_answer_tokens) == (8192, 500)
    blank = load_settings({**env, "SB_CHAT_CONTEXT_TOKENS": "", "SB_CHAT_MAX_ANSWER_TOKENS": ""}, repo_root=tmp_path / "repo")
    assert blank.chat_context_tokens is None and blank.chat_max_answer_tokens == 700


@pytest.mark.parametrize("name,value", [("SB_CHAT_CONTEXT_TOKENS", "abc"), ("SB_CHAT_CONTEXT_TOKENS", "100"),
                                        ("SB_CHAT_MAX_ANSWER_TOKENS", "x"), ("SB_CHAT_MAX_ANSWER_TOKENS", "10")])
def test_bad_chat_settings_are_rejected(tmp_path, name, value):
    env = {"SB_READER_MODEL": "r", "SB_CHAT_MODEL": "c", "SB_EMBED_MODEL": "e",
           "SB_LIBRARY_DIR": str(tmp_path / "l"), "SB_DATA_DIR": str(tmp_path / "d"), name: value}
    with pytest.raises(ConfigError, match=name):
        load_settings(env, repo_root=tmp_path / "repo")
