from dataclasses import dataclass
from datetime import datetime

from studybuddy.chat.answers import Source
from studybuddy.ui.markup import greeting, short_title, slip_caption, suggest_questions


@dataclass
class Row:
    title: str
    subject: str


def source(**kw):
    base = dict(number=1, subject="Science", title="Water cycle", file_path="C:/lib/Science/water-1.jpg", page_number=2,
                image_path="C:/lib/Science/water-1.jpg", text="t", page_quality=0.95, page_flags=[])
    return Source(**{**base, **kw})


def test_the_greeting_follows_the_time_of_day():
    assert "Good morning." in greeting(datetime(2026, 1, 1, 9))
    assert "Good afternoon." in greeting(datetime(2026, 1, 1, 14))
    assert "Good evening." in greeting(datetime(2026, 1, 1, 21))
    assert "What are you studying?" in greeting()


def test_examples_use_the_students_own_papers_and_subjects():
    rows = [Row("Water cycle worksheet", "Science"), Row("Layers of the Earth", "Science"), Row("Fractions practice", "Math")]
    assert suggest_questions(rows) == ["What is Water cycle worksheet about?", "Summarize my Science papers",
                                       "What are the key terms in Layers of the Earth?"]


def test_one_paper_gives_a_short_list_without_repeats_and_no_papers_gives_none():
    assert suggest_questions([Row("Fractions", "Math")]) == ["What is Fractions about?", "What are the main ideas in my Math paper?",
                                                              "What are the key terms in Fractions?"]
    assert suggest_questions([]) == [] and suggest_questions([Row("", "Math")]) == []


def test_long_titles_are_cut_at_a_word_without_dots():
    cut = short_title("A very long title about the movement of tectonic plates across the whole planet")
    assert cut == "A very long title about the movement of" and short_title("Short") == "Short"


def test_the_slip_caption_names_the_file_and_page_and_escapes_everything():
    html = slip_caption(source(title="<b>Tricky</b> & co"))
    assert "<strong>water-1.jpg</strong>, page 2" in html and "<b>Tricky</b>" not in html and "&lt;b&gt;Tricky&lt;/b&gt;" in html


def test_a_hard_to_read_page_gets_a_plain_warning_and_a_cited_one_says_so():
    html = slip_caption(source(page_quality=0.3, page_flags=["many_illegible"], cited=True))
    assert "This page was hard to read" in html and "Check the photo." in html and "Used in the answer." in html
    assert "hard to read" not in slip_caption(source())


def test_a_missing_page_image_is_explained_inside_the_caption():
    assert "The page image is not available" in slip_caption(source(), "The page image is not available: gone")
    assert "not in the last answer" in slip_caption(None, "That source is not in the last answer.")
