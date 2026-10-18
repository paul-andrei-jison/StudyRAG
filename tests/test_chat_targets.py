from datetime import datetime, timedelta, timezone

import pytest

from studybuddy.chat.overview import detect
from studybuddy.chat.targets import (ASK, DOCUMENTS, LIST_MAX, NO_MATCH, ask_which, match_choice, newest_first,
                                     resolve_targets)
from studybuddy.index.store import DocumentRow

NOW = datetime(2026, 10, 18, 12, 0, tzinfo=timezone.utc)


def doc(i, title, subject="Science", summary="", minutes_ago=600):
    when = (NOW - timedelta(minutes=minutes_ago)).isoformat(timespec="seconds")
    return DocumentRow(i, f"hash{i}", f"/lib/{subject}/{i}.jpg", "photo", subject, title, summary, 1, 3, when)


THREE = [doc(1, "Earth's Spheres and Interactions", summary="geosphere biosphere hydrosphere atmosphere"),
         doc(2, "Earth's Systems and Natural Hazards", summary="typhoons deforestation magnetic field"),
         doc(3, "Earth's Conditions for Life", summary="habitable zone liquid water")]


def many(n=24):
    """n documents over three subjects; the newest has the highest id."""
    subjects = ["Science", "Math", "English"]
    return [doc(i, f"Worksheet number {i} on {['plants', 'fractions', 'poems'][i % 3]}", subjects[i % 3],
                minutes_ago=10_000 - i * 100) for i in range(1, n + 1)]


def resolve(question, docs, subject=None, now=NOW):
    return resolve_targets(detect(question), question, docs, subject, now)


# ---- choosing -------------------------------------------------------------------------------------

def test_a_named_document_wins_even_among_many():
    r = resolve("give me an overview of the Earth's Spheres and Interactions handout", THREE + many())
    assert r.kind == DOCUMENTS and r.why == "named" and [d.id for d in r.docs] == [1]


def test_a_partial_name_that_fits_several_names_none():
    r = resolve("list everything about the Earth article", THREE + many())
    assert r.kind == DOCUMENTS and r.why == "few" and [d.id for d in r.docs] == [1, 2, 3]   # 3 papers are about Earth: each is summarised


def test_the_subject_filter_that_leaves_one_document_is_enough():
    docs = THREE + [doc(10, "Fractions drill", "Math")]
    r = resolve("summarize the entire paper", docs, subject="Math")
    assert r.kind == DOCUMENTS and r.why == "only" and r.docs[0].id == 10
    assert resolve("summarize the entire paper", docs, subject="math").docs[0].id == 10      # case does not matter


def test_this_or_latest_means_the_newest_document_if_it_is_recent():
    docs = many(10) + [doc(99, "Fresh upload", minutes_ago=5)]
    r = resolve("summarize this handout", docs)
    assert r.kind == DOCUMENTS and r.why == "latest" and r.docs[0].id == 99
    assert resolve("buod ng papel na ito", docs).docs[0].id == 99


def test_an_old_newest_document_is_not_assumed():
    r = resolve("summarize this handout", many(10))
    assert r.kind == ASK                                    # the newest is 15 hours old and there are 10 papers
    r = resolve("summarize the entire paper", many(10) + [doc(99, "Fresh upload", minutes_ago=5)])
    assert r.kind == ASK                                    # no "this/latest" in the question: do not guess


def test_up_to_three_documents_are_each_summarised():
    r = resolve("give me a summary of the entire paper", THREE)
    assert r.kind == DOCUMENTS and r.why == "few" and [d.id for d in r.docs] == [1, 2, 3]
    assert resolve("give me a summary of the entire paper", THREE[:2]).why == "few"


def test_more_than_three_documents_ask_which_one():
    r = resolve("give me a summary of the entire paper", THREE + [doc(4, "Rocks and minerals")])
    assert r.kind == ASK and len(r.candidates) == 4 and "Which paper do you mean?" in r.text


def test_a_topic_that_matches_no_paper_resolves_nothing_so_the_normal_path_refuses_it():
    assert resolve("summarize the French Revolution paper", THREE).kind == NO_MATCH
    assert resolve("summarize the French Revolution paper", THREE + many()).kind == NO_MATCH


def test_a_topic_narrows_the_choice_to_the_papers_about_it():
    docs = THREE + many(12)
    r = resolve("summary of the paper about fractions", docs)
    assert r.kind == ASK and {d.subject for d in r.candidates} == {"Math"}
    docs = THREE + [doc(20, "Poems of spring", "English", summary="rhyme and verse")]
    r = resolve("summary of the paper about rhyme", docs)
    assert r.kind == DOCUMENTS and r.why == "only" and r.docs[0].id == 20


def test_an_empty_scope_resolves_nothing():
    assert resolve("summarize the entire paper", []).kind == NO_MATCH
    assert resolve("summarize the entire paper", THREE, subject="History").kind == NO_MATCH


# ---- asking with 20+ documents ----------------------------------------------------------------------

def test_the_list_shows_the_eight_most_recent_and_says_how_many_are_hidden():
    docs = many(24)
    r = resolve("summarize the entire paper", docs)
    assert r.kind == ASK and len(r.shown) == LIST_MAX == 8 and len(r.candidates) == 24
    assert [d.id for d in r.shown] == [24, 23, 22, 21, 20, 19, 18, 17]
    lines = r.text.splitlines()
    numbered = [ln for ln in lines if ln[:2] in {f"{i}." for i in range(1, 9)}]
    assert len(numbered) == 8 and numbered[0].startswith("1. Science | Worksheet number 24 ") and "2026-10-" in numbered[0]
    assert "16 more are not shown: pick a subject to narrow it, or type part of a title." in r.text
    assert "9." not in r.text and "Worksheet number 1 " not in r.text


def test_a_short_list_has_no_hidden_note():
    r = ask_which(many(5))
    assert len(r.shown) == 5 and "not shown" not in r.text


def test_newest_first_orders_by_time_then_id():
    assert [d.id for d in newest_first(many(5))] == [5, 4, 3, 2, 1]


@pytest.fixture
def pending():
    return resolve("summarize the entire paper", many(24))


def test_a_number_from_the_shown_list_picks_that_paper(pending):
    c = match_choice("3", pending)
    assert c.kind == "chosen" and c.doc.id == 22
    assert match_choice("1.", pending).doc.id == 24


@pytest.mark.parametrize("reply", ["0", "9", "25", "99"])
def test_a_number_outside_the_list_asks_again_with_the_same_list(pending, reply):
    c = match_choice(reply, pending)
    assert c.kind == "invalid" and c.ask.text.startswith(f"There is no number {reply}") and c.ask.shown == pending.shown


def test_a_full_title_picks_a_paper_even_if_it_is_not_in_the_shown_list(pending):
    c = match_choice("Worksheet number 3 on plants", pending)     # id 3 is hidden: only the 8 newest are listed
    assert c.kind == "chosen" and c.doc.id == 3


def test_part_of_a_title_picks_a_paper_when_it_is_unique(pending):
    c = match_choice("worksheet 7", pending)
    assert c.kind == "chosen" and c.doc.id == 7
    assert match_choice("WORKSHEET NUMBER 12 PLANTS", pending).doc.id == 12


def test_part_of_a_title_that_fits_several_asks_again_naming_them_and_remembers_the_narrower_list(pending):
    c = match_choice("fractions", pending)
    assert c.kind == "ambiguous" and c.doc is None
    assert c.ask.text.startswith("Several papers match.") and len(c.ask.shown) == 8 and len(c.ask.candidates) == 8
    assert all("fractions" in d.title for d in c.ask.candidates)
    assert match_choice("2", c.ask).kind == "chosen"          # the next reply is read against the narrowed list


@pytest.mark.parametrize("reply", ["", "   ", "?", "never mind, what is photosynthesis", "banana", "the", "what about gravity?"])
def test_anything_else_is_not_a_choice(pending, reply):
    assert match_choice(reply, pending).kind == "none"
