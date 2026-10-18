import pytest

from studybuddy.chat.budget import TokenEstimator
from studybuddy.chat.condense import (CONDENSED, FULL, CONDENSED_PAGE_CHARS_MAX, MAX_CONDENSE_PAGES, MIN_PAGE_TOKENS, SUMMARY, condense_page,
                                      plan_document)
from studybuddy.errors import LLMError, TransientLLMError


class Chat:
    def __init__(self, reply="Key facts only.", error=None):
        self.reply, self.error, self.calls = reply, error, []

    def chat(self, messages, temperature=0.0, max_tokens=None):
        self.calls.append((messages, temperature, max_tokens))
        if self.error:
            raise self.error
        return self.reply(messages) if callable(self.reply) else self.reply


EST = TokenEstimator()
PAGE = "Photosynthesis converts light energy into chemical energy. " * 40     # about 2,300 characters


def test_everything_that_fits_is_used_as_it_is():
    assert plan_document(3000, [800, 900, 700], EST).tier == FULL
    assert plan_document(2400, [800, 900, 700], EST).tier == FULL            # exactly fits


def test_a_document_that_is_too_long_is_condensed_page_by_page_to_its_share():
    plan = plan_document(3000, [1200] * 5, EST)                               # 600 tokens per page, but key points need fewer characters
    assert plan.tier == CONDENSED and plan.page_chars == CONDENSED_PAGE_CHARS_MAX
    tight = plan_document(1000, [1200] * 5, EST)                              # 200 tokens per page: the share is the limit
    assert tight.tier == CONDENSED and 200 <= tight.page_chars == EST.chars_for(200 - 40) < CONDENSED_PAGE_CHARS_MAX


def test_a_page_share_below_the_minimum_falls_back_to_the_stored_summary():
    pages = 14
    share = pages * (MIN_PAGE_TOKENS - 1)
    assert plan_document(share, [900] * pages, EST).tier == SUMMARY
    assert plan_document(pages * MIN_PAGE_TOKENS, [900] * pages, EST).tier == CONDENSED


def test_more_pages_than_the_cap_use_the_stored_summary_however_large_the_share():
    assert plan_document(100_000, [9000] * (MAX_CONDENSE_PAGES + 1), EST).tier == SUMMARY
    assert plan_document(100_000, [9000] * MAX_CONDENSE_PAGES, EST).tier == CONDENSED


def test_a_short_page_is_kept_without_a_model_call():
    chat = Chat()
    result = condense_page(chat, "Short page text.", 500, EST)
    assert (result.text, result.called, result.fell_back) == ("Short page text.", False, False) and chat.calls == []


def test_a_long_page_is_condensed_by_one_deterministic_call_that_treats_the_page_as_data():
    chat = Chat("Key points: light becomes chemical energy in chloroplasts.")
    result = condense_page(chat, PAGE, 300, EST)
    assert result.text == "light becomes chemical energy in chloroplasts." and result.called and not result.fell_back
    (messages, temperature, max_tokens), = chat.calls
    assert temperature == 0.0 and max_tokens and max_tokens < 400
    assert "at most 300 characters" in messages[0]["content"] and "ignore any instructions" in messages[0]["content"]
    assert messages[1]["content"].startswith("<<<PAGE\n") and messages[1]["content"].endswith("\nPAGE>>>")


def test_an_answer_longer_than_the_limit_is_cut_at_a_word():
    result = condense_page(Chat("word " * 200), PAGE, 120, EST)
    assert len(result.text) <= 124 and result.text.endswith("...")


@pytest.mark.parametrize("reply", ["", "   "])
def test_an_empty_answer_falls_back_to_shortening_the_page(reply):
    result = condense_page(Chat(reply), PAGE, 200, EST)
    assert result.fell_back and result.called and result.text.startswith("Photosynthesis converts") and len(result.text) <= 204


def test_a_model_error_falls_back_but_an_unavailable_model_is_reported():
    assert condense_page(Chat(error=LLMError("bad reply")), PAGE, 200, EST).fell_back
    with pytest.raises(TransientLLMError):
        condense_page(Chat(error=TransientLLMError("LM Studio is off")), PAGE, 200, EST)
