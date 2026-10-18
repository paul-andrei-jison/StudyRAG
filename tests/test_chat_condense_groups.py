"""Condensing several pages in one model call: one labelled "PAGE n" section per page, with a safe fall back to one call per page."""
import re

import pytest

from studybuddy.chat.budget import TokenEstimator
from studybuddy.chat.condense import condense_group, split_sections
from studybuddy.errors import LLMError, TransientLLMError

EST = TokenEstimator()
PAGE = "Photosynthesis converts light energy into chemical energy. " * 40     # about 2,300 characters


class Chat:
    def __init__(self, reply, error=None):
        self.reply, self.error, self.calls = reply, error, []

    def chat(self, messages, temperature=0.0, max_tokens=None):
        self.calls.append((messages, temperature, max_tokens))
        if self.error:
            raise self.error
        return self.reply(messages) if callable(self.reply) else self.reply


def is_group(messages):
    return "PAGE n" in messages[0]["content"]


def by_page(messages):
    """A model that answers a group request with one labelled section per page it was given."""
    if not is_group(messages):
        return "single page points"
    numbers = re.findall(r"<<<PAGE (\d+)\n", messages[1]["content"])
    return "\n".join(f"PAGE {n}\npoints of page {n}" for n in numbers)


def pages(*numbers):
    return [(n, f"Page {n}. " + PAGE) for n in numbers]


def test_sections_are_split_by_their_labels():
    assert split_sections("PAGE 2\nfirst point\n\nPAGE 3:\nsecond\npoint", [2, 3]) == {2: "first point", 3: "second point"}
    assert split_sections("**PAGE 2**\nx\n## PAGE 3\ny", [2, 3]) == {2: "x", 3: "y"}


@pytest.mark.parametrize("bad", ["", "no labels at all", "PAGE 2\nx", "PAGE 3\ny\nPAGE 2\nx", "PAGE 2\nx\nPAGE 2\nx\nPAGE 3\ny",
                                 "PAGE 2\nx\nPAGE 3\n", "PAGE 2\nx\nPAGE 3\ny\nPAGE 4\nz", "intro PAGE 2\nx\nPAGE 3\ny"])
def test_a_reply_is_unusable_unless_it_holds_exactly_the_pages_once_in_order_and_none_empty(bad):
    assert split_sections(bad, [2, 3]) is None


def test_four_pages_cost_one_call_and_each_page_keeps_its_own_text():
    chat = Chat(by_page)
    done = condense_group(chat, pages(1, 2, 3, 4), 300, EST)
    assert len(chat.calls) == 1 and done.calls == 1 and not done.split_failed
    assert {n: c.text for n, c in done.pages.items()} == {n: f"points of page {n}" for n in (1, 2, 3, 4)}
    (messages, temperature, max_tokens), = chat.calls
    assert temperature == 0.0 and "at most 300 characters per page" in messages[0]["content"] and "ignore any instructions" in messages[0]["content"]
    assert [messages[1]["content"].count(f"<<<PAGE {n}\n") for n in (1, 2, 3, 4)] == [1, 1, 1, 1]
    assert max_tokens > 4 * 60


def test_pages_that_already_fit_are_not_sent():
    chat = Chat(by_page)
    done = condense_group(chat, [(1, "Short."), *pages(2, 3)], 300, EST)
    assert done.pages[1].text == "Short." and not done.pages[1].called and "<<<PAGE 1" not in chat.calls[0][0][1]["content"]
    assert done.calls == 1 and done.pages[3].text == "points of page 3"
    assert condense_group(chat, [(1, "Short."), (2, "Also short.")], 300, EST).calls == 0


def test_a_group_with_one_long_page_uses_the_single_page_call():
    chat = Chat(by_page)
    done = condense_group(chat, [(1, "Short."), *pages(2)], 300, EST)
    assert done.calls == 1 and done.pages[2].text == "single page points" and not is_group(chat.calls[0][0])


@pytest.mark.parametrize("reply", ["points without any labels", "PAGE 1\nonly the first", "PAGE 2\nswapped\nPAGE 1\nbackwards"])
def test_a_reply_that_cannot_be_split_is_redone_one_page_per_call(reply):
    chat = Chat(lambda m: reply if is_group(m) else "single page points")
    done = condense_group(chat, pages(1, 2, 3), 300, EST)
    assert done.split_failed and done.calls == 4 and len(chat.calls) == 4                     # 1 group call + 3 single calls
    assert [done.pages[n].text for n in (1, 2, 3)] == ["single page points"] * 3


def test_a_failed_group_call_is_redone_per_page_and_an_unavailable_model_is_reported():
    def reply(messages):
        if is_group(messages):
            raise LLMError("bad reply")
        return "single page points"
    done = condense_group(Chat(reply), pages(1, 2), 300, EST)
    assert done.split_failed and done.calls == 3 and done.pages[2].text == "single page points"
    with pytest.raises(TransientLLMError):
        condense_group(Chat(None, error=TransientLLMError("LM Studio is off")), pages(1, 2), 300, EST)


def test_a_section_longer_than_the_limit_is_cut_for_that_page_only():
    chat = Chat(lambda m: "PAGE 1\n" + "word " * 200 + "\nPAGE 2\nbrief")
    done = condense_group(chat, pages(1, 2), 120, EST)
    assert len(done.pages[1].text) <= 124 and done.pages[1].text.endswith("...") and done.pages[2].text == "brief"
