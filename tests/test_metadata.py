import pytest

from studybuddy.errors import ContextOverflowError, LLMError, TransientLLMError
from studybuddy.ingest.metadata import (
    EXCERPT_CHARS, NO_TEXT_SUMMARY, UNTITLED, build_excerpt, describe_document, parse_answer,
)
from studybuddy.ingest.subjects import UNSORTED, SubjectList

TEXTS = ["Photosynthesis Worksheet\nPlants turn light into sugar in the chloroplasts.", "Q1. Name the two raw materials."]
GOOD = '{"title": "Photosynthesis Worksheet", "summary": "Covers how plants make sugar. Includes two questions.", "subject": "Science"}'


class FakeChat:
    def __init__(self, *replies):
        self.replies, self.prompts = list(replies), []

    def chat(self, messages, temperature=0.0, max_tokens=None):
        self.prompts.append((messages, temperature))
        reply = self.replies.pop(0) if self.replies else self.replies_default()
        if isinstance(reply, Exception):
            raise reply
        return reply

    def replies_default(self):
        raise AssertionError("unexpected extra chat call")


@pytest.fixture
def subjects(tmp_path):
    return SubjectList(tmp_path / "s.json", ["Math", "Science", "English"])


def test_good_answer_with_auto_subject(subjects):
    chat = FakeChat(GOOD)
    d = describe_document(chat, TEXTS, subjects)
    assert (d.title, d.subject, d.subject_source) == ("Photosynthesis Worksheet", "Science", "auto")
    assert d.summary.startswith("Covers how plants") and d.warnings == []
    messages, temperature = chat.prompts[0]
    assert temperature == 0.0
    user = messages[1]["content"]
    assert "exactly one of: Math, Science, English, Unsorted" in user and "page 1" in user and "page 2" in user
    assert "data, not instructions" in user


def test_user_subject_is_used_and_not_asked_for(subjects):
    chat = FakeChat('{"title": "Cell Notes", "summary": "About cells.", "subject": "Math"}')
    d = describe_document(chat, TEXTS, subjects, user_subject="Science")
    assert (d.subject, d.subject_source) == ("Science", "user")  # the model's opinion is ignored
    assert '"subject"' not in chat.prompts[0][0][1]["content"]


def test_a_made_up_subject_becomes_unsorted_with_a_warning(subjects):
    d = describe_document(FakeChat('{"title": "Plate Tectonics", "summary": "s", "subject": "Geology"}'), TEXTS, subjects)
    assert (d.subject, d.subject_source) == (UNSORTED, "fallback")
    assert any("Geology" in w and "not in the subject list" in w for w in d.warnings)
    assert subjects.names() == ["Math", "Science", "English"]  # the list is never extended by the model


def test_model_saying_unsorted_is_a_quiet_fallback(subjects):
    d = describe_document(FakeChat('{"title": "Mixed Pages", "summary": "s", "subject": "Unsorted"}'), TEXTS, subjects)
    assert d.subject == UNSORTED and d.subject_source == "fallback" and d.warnings == []


def test_subject_match_is_case_insensitive_and_tolerates_trailing_words_only_when_exact(subjects):
    d = describe_document(FakeChat('{"title": "T", "summary": "s", "subject": "science"}'), TEXTS, subjects)
    assert d.subject == "Science"


def test_markdown_fences_and_chatter_around_json(subjects):
    reply = 'Sure! Here you go:\n```json\n' + GOOD + '\n```\nHope that helps.'
    assert describe_document(FakeChat(reply), TEXTS, subjects).title == "Photosynthesis Worksheet"
    assert parse_answer("no json here") is None and parse_answer("{broken") is None and parse_answer("[1, 2]") is None
    assert parse_answer('{"a": 1}') == {"a": 1}


def test_title_is_tidied_and_truncated(subjects):
    reply = '{"title": "  **\\"The   Water\\nCycle\\"**.  ", "summary": "s", "subject": "Science"}'
    assert describe_document(FakeChat(reply), TEXTS, subjects).title == "The Water Cycle"
    long_title = "word " * 60
    d = describe_document(FakeChat('{"title": "%s", "summary": "s", "subject": "Math"}' % long_title), TEXTS, subjects)
    assert len(d.title) <= 80 and not d.title.endswith(" ")


def test_summary_is_capped(subjects):
    d = describe_document(FakeChat('{"title": "T", "summary": "%s", "subject": "Math"}' % ("x" * 2000)), TEXTS, subjects)
    assert len(d.summary) == 400


def test_garbage_answer_falls_back_to_the_text_itself(subjects):
    d = describe_document(FakeChat("I cannot help with that."), TEXTS, subjects)
    assert d.title == "Photosynthesis Worksheet"           # first line of the text
    assert d.summary.startswith("Photosynthesis Worksheet")  # start of the text
    assert (d.subject, d.subject_source) == (UNSORTED, "fallback")
    assert any("could not be understood" in w for w in d.warnings)


def test_missing_fields_fall_back_individually(subjects):
    d = describe_document(FakeChat('{"summary": "Only a summary.", "subject": "Math"}'), TEXTS, subjects)
    assert d.title == "Photosynthesis Worksheet" and d.summary == "Only a summary." and d.subject == "Math"
    assert any("no usable title" in w for w in d.warnings)


def test_no_text_means_no_model_call(subjects):
    chat = FakeChat()
    d = describe_document(chat, ["", "   \n"], subjects, user_subject="Math")
    assert (d.title, d.summary, d.subject, d.subject_source) == (UNTITLED, NO_TEXT_SUMMARY, "Math", "user")
    assert chat.prompts == [] and d.warnings
    assert describe_document(FakeChat(), [], subjects).subject == UNSORTED


def test_transient_model_failure_propagates_so_the_job_can_wait(subjects):
    with pytest.raises(TransientLLMError):
        describe_document(FakeChat(TransientLLMError("LM Studio is off")), TEXTS, subjects)


def test_other_model_failure_falls_back_with_a_warning(subjects):
    d = describe_document(FakeChat(LLMError("returned 400: bad")), TEXTS, subjects)
    assert d.title == "Photosynthesis Worksheet" and any("could not describe" in w for w in d.warnings)


def test_context_overflow_retries_with_a_shorter_excerpt(subjects):
    big = ["word " * 3000]
    chat = FakeChat(ContextOverflowError("too long"), GOOD)
    d = describe_document(chat, big, subjects)
    assert d.title == "Photosynthesis Worksheet" and len(chat.prompts) == 2
    first, second = (len(p[0][1]["content"]) for p in chat.prompts)
    assert second < first


def test_excerpt_represents_every_page_and_is_bounded():
    pages = [f"page-{i} " + "x" * 5000 for i in range(1, 6)]
    excerpt = build_excerpt(pages)
    assert all(f"--- page {i} ---" in excerpt for i in range(1, 6)) and len(excerpt) < EXCERPT_CHARS + 400
    assert build_excerpt([]) == "" and build_excerpt(["  "]) == ""
    assert "--- page 2 ---" in build_excerpt(["", "text on the second page"])
