import pytest

from studybuddy.chat.rewrite import MAX_QUERY_CHARS, Turn, build_prompt, clean_rewrite, rewrite_query
from studybuddy.errors import LLMError, TransientLLMError

HISTORY = [Turn("What generates Earth's magnetic field?", "Movement of molten iron in the outer core [1]."),
           Turn("How much of the surface is water?", "About 71% [2].")]


class FakeChat:
    def __init__(self, *replies):
        self.replies, self.calls = list(replies), []

    def chat(self, messages, temperature=0.0, max_tokens=None):
        self.calls.append((messages, temperature, max_tokens))
        reply = self.replies.pop(0)
        if isinstance(reply, Exception):
            raise reply
        return reply


def test_no_history_means_no_model_call():
    chat = FakeChat()
    r = rewrite_query(chat, [], "What is the habitable zone?")
    assert (r.query, r.rewritten) == ("What is the habitable zone?", False) and chat.calls == []


def test_a_follow_up_is_rewritten_into_a_standalone_query():
    chat = FakeChat("What are the effects of the greenhouse effect on Earth's temperature?")
    r = rewrite_query(chat, HISTORY, "and what are its effects?")
    assert r.rewritten and r.query == "What are the effects of the greenhouse effect on Earth's temperature?"
    messages, temperature, max_tokens = chat.calls[0]
    assert temperature == 0.0 and max_tokens == 100
    user = messages[1]["content"]
    assert "Student: What generates Earth's magnetic field?" in user and "Assistant: Movement of molten iron" in user
    assert "Follow-up question: and what are its effects?" in user
    assert "[1]" not in user and "[2]" not in user  # citation markers are noise for the rewrite
    assert "Never answer" in messages[0]["content"]


def test_only_the_most_recent_turns_and_short_answers_are_sent():
    long_history = [Turn(f"question number {i}", "word " * 500) for i in range(8)]
    user = build_prompt(long_history, "and why?")[1]["content"]
    assert "question number 7" in user and "question number 5" in user and "question number 4" not in user
    assert max(len(line) for line in user.splitlines()) < 400 and "..." in user


@pytest.mark.parametrize("raw,expected", [
    ("What causes tides?", "What causes tides?"),
    ('"What causes tides?"', "What causes tides?"),
    ("Rewritten question: What causes tides?", "What causes tides?"),
    ("Standalone query - What causes tides?", "What causes tides?"),
    ("What causes tides?\nThis is because the moon pulls on the oceans.", "What causes tides?"),
    ("\n\n  **What causes tides?**  ", "What causes tides?"),
])
def test_clean_rewrite_accepts_tidy_variants(raw, expected):
    assert clean_rewrite(raw, "why?") == expected


@pytest.mark.parametrize("raw", ["", "   \n ", "I'm sorry, I cannot rewrite that.", "I cannot help with that", "Sorry, which topic?",
                                 "As an AI I do not know.", "x" * (MAX_QUERY_CHARS + 1), "Tides are caused by the moon. " * 20])
def test_clean_rewrite_rejects_answers_and_junk(raw):
    assert clean_rewrite(raw, "why?") is None


def test_an_unusable_rewrite_falls_back_to_the_question_as_typed():
    r = rewrite_query(FakeChat("I'm sorry, I cannot do that."), HISTORY, "and its effects?")
    assert (r.query, r.rewritten) == ("and its effects?", False) and "unusable" in r.note


def test_an_unchanged_question_is_not_counted_as_rewritten():
    r = rewrite_query(FakeChat("What is the habitable zone?"), HISTORY, "what is the habitable zone?")
    assert not r.rewritten and r.query == "what is the habitable zone?" and r.note == ""


def test_other_model_errors_fall_back_but_transient_ones_propagate():
    r = rewrite_query(FakeChat(LLMError("returned 400")), HISTORY, "and its effects?")
    assert (r.query, r.rewritten) == ("and its effects?", False) and "could not rewrite" in r.note
    with pytest.raises(TransientLLMError):
        rewrite_query(FakeChat(TransientLLMError("LM Studio is off")), HISTORY, "and its effects?")


def test_a_rewrite_that_is_far_longer_than_the_question_is_treated_as_an_answer():
    answer = ("The effects include warming of the surface, melting of ice caps, rising sea levels, more droughts and floods, "
              "stronger storms, loss of habitats and changes in farming across the whole world over many decades, as scientists keep warning.")
    assert len(answer) > 3 * len("and its effects?") + 160
    assert clean_rewrite(answer, "and its effects?") is None
