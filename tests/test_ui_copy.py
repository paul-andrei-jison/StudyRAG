"""The words on screen: plain, sentence case, no internal terms, one name per action."""
import re

import pytest

from test_ui_app import Served

INTERNAL = re.compile(r"cosine|\bgate\b|needs_choice|\bjobs?\b|\bchunks?\b|\bvectors?\b|embedding|traceback|\bapi\b", re.I)
SHOUTING = re.compile(r"\b[A-Z]{4,}\b")
ALLOWED_CAPS = {"LLM"}


@pytest.fixture
def blocks(tmp_path, settings):
    app = Served(tmp_path, settings)
    yield app.demo
    app.close()


def inside_technical_details(block) -> bool:
    while block is not None:
        if getattr(block, "elem_id", None) == "sb-tech":
            return True
        block = getattr(block, "parent", None)
    return False


def visible_strings(demo):
    """(component id, kind, text) for every label, placeholder, info, button name, choice and fixed text, outside Technical details."""
    out = []
    for block in demo.blocks.values():
        if inside_technical_details(block) or getattr(block, "visible", True) == "hidden":   # (buttons that exist only for the API are hidden too)
            continue
        name = getattr(block, "elem_id", None) or type(block).__name__
        for attr in ("label", "placeholder", "info"):
            value = getattr(block, attr, None)
            if isinstance(value, str) and value:
                out.append((name, attr, value))
        if type(block).__name__ in ("Button", "HTML", "Markdown") and isinstance(getattr(block, "value", None), str):
            out.append((name, "text", re.sub(r"<[^>]+>", " ", block.value)))
        for choice in getattr(block, "choices", None) or []:
            out.append((name, "choice", choice[0] if isinstance(choice, (tuple, list)) else str(choice)))
    return out


def test_no_visible_text_uses_an_internal_term_or_shouts(blocks):
    for name, kind, text in visible_strings(blocks):
        assert not INTERNAL.search(text), (name, kind, text)
        shouting = [w for w in SHOUTING.findall(text) if w not in ALLOWED_CAPS]
        assert not shouting, (name, kind, text)


def test_no_label_or_button_name_ends_in_dots(blocks):
    for name, kind, text in visible_strings(blocks):
        if kind in ("label", "choice") or (kind == "text" and name != "HTML"):
            assert "..." not in text and "…" not in text, (name, kind, text)


def test_the_screens_and_their_actions_have_one_name_each(blocks):
    buttons = {b.value for b in blocks.blocks.values() if type(b).__name__ == "Button" and b.value}   # ("Run" is only the default name of the hidden API buttons)
    assert {"Ask", "New chat", "Add papers", "Retry", "Close", "Change subject", "Remove from Study Buddy", "Keep it",
            "Back to papers", "Show the closest pages"} <= buttons
    banned = {"Upload", "Submit", "Delete", "Delete from index", "Add to my library", "Refresh", "Chat"}
    assert not (banned & buttons) and buttons - {"Run"} >= {"Ask"}
    tabs = [b.label for b in blocks.blocks.values() if type(b).__name__ == "Tab"]
    assert tabs == ["Ask", "Library", "Add"]


def test_labels_are_sentence_case(blocks):
    for name, kind, text in visible_strings(blocks):
        if kind in ("label", "choice"):
            words = text.split()
            capitalised = [w for w in words[1:] if w[:1].isupper() and w.lower() not in ("study", "buddy", "lm", "studio", "pdf") and not w[:1].isdigit()]
            # subject and paper names in choices are the student's own words; fixed labels must not be Title Case
            if kind == "label":
                assert not capitalised, (name, text)
