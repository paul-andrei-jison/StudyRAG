"""Overview questions through the whole ChatSession: the gate is skipped, whole documents are the sources, refusals are kept."""
import re

import pytest

from chat_helpers import EARTH, LAX_GATE, PLATES, WATER, ScriptedClient, add_doc, make_setup
from studybuddy.chat.answers import ANSWERED, NEEDS_CHOICE, NOT_IN_PAPERS_STATUS, OVERVIEW, SEARCH, ChatAnswer, Progress
from studybuddy.chat.gate import GateConfig
from studybuddy.chat.prompt import OVERVIEW_SYSTEM_PROMPT, SYSTEM_PROMPT
from studybuddy.chat.session import HANDWRITING_NOTE
from studybuddy.errors import ChatContextError
from studybuddy.ingest.subjects import SubjectList

STRICT = GateConfig(0.99, 0.99, 1.0)      # a gate that refuses every question a page does not match: overview must not need it
THREE = ["The magnetic field ", "[1]", ". Water covers the surface ", "[2]", ". Plates move ", "[3]", "."]


@pytest.fixture
def setup(settings, tmp_path):
    s = make_setup(settings, tmp_path)
    yield s
    s.index.close()


class Spy(ScriptedClient):
    """Records condensing calls (chat) and answers them with a fixed short text."""

    def chat(self, messages, temperature=0.0, max_tokens=None):
        self.chat_calls.append(messages)
        return "short points"


def run(setup, client, question, **kw):
    kw.setdefault("gate", STRICT)
    session = setup.session(client, **kw)
    return session, list(session.ask_stream(question))


def final_of(items):
    return next(i for i in reversed(items) if isinstance(i, ChatAnswer))


# ---- the failing question ------------------------------------------------------------------------

def test_a_summary_of_the_entire_paper_is_answered_from_whole_documents_not_refused(setup):
    client = ScriptedClient([THREE])
    session, items = run(setup, client, "give me a summary of the entire paper")
    final = final_of(items)
    assert final.status == ANSWERED and final.mode == OVERVIEW and final.cited == [1, 2, 3] and not final.invalid_citations
    assert [s.title for s in final.sources] == ["Earth notes", "Water facts", "Plate tectonics"]    # the pages, in document order
    assert [s.page_number for s in final.sources] == [1, 1, 1]
    assert client.chat_calls == []                                  # no rewrite, no expansion, no condensing: nothing else was asked
    assert final.overview_notes[0].startswith("This is an overview of 'Earth notes' (1 page); 'Water facts' (1 page);")
    assert final.text.endswith(final.notes[-1]) and "not an answer found on one page" in final.text


def test_the_overview_prompt_is_used_with_whole_pages_and_no_history(setup):
    client = ScriptedClient([THREE, THREE])
    session = setup.session(client, gate=STRICT)
    session.ask("give me a summary of the entire paper")
    session.ask("summarize everything again")
    system, user = client.stream_calls[1][0], client.stream_calls[1][-1]
    assert system["content"] == OVERVIEW_SYSTEM_PROMPT != SYSTEM_PROMPT
    assert len(client.stream_calls[1]) == 2                           # system + this question: no earlier turns
    assert "SOURCE 1 - Science | Earth notes" in user["content"] and EARTH in user["content"] and PLATES in user["content"]
    assert user["content"].index(EARTH) < user["content"].index(WATER) < user["content"].index(PLATES)


@pytest.mark.parametrize("question", [
    "summarize this handout", "what is this worksheet about", "what topics does the quiz cover",
    "list everything about the Earth article", "buod ng papel na ito", "ano ang laman nito", "tungkol saan ang worksheet",
])
def test_variants_are_answered_as_overviews(setup, question):
    final = final_of(run(setup, ScriptedClient([THREE]), question)[1])
    assert final.mode == OVERVIEW and final.status == ANSWERED and final.sources, question


def test_the_gate_is_not_consulted_and_no_search_is_run(setup, monkeypatch):
    calls = []
    original = setup.index.search
    monkeypatch.setattr(setup.index, "search", lambda *a, **k: calls.append(a) or original(*a, **k))
    run(setup, ScriptedClient([THREE]), "give me a summary of the entire paper")
    assert calls == []


# ---- which documents ---------------------------------------------------------------------------------

def test_the_subject_filter_selects_the_documents(setup):
    names = SubjectList(setup.settings.data_dir / "subjects.json", ["Science", "Geography"])
    client = ScriptedClient([["Plates move ", "[1]", "."]])
    session = setup.session(client, subjects=names, gate=STRICT)
    session.set_subject("Geography")
    final = session.ask("summarize the entire paper")
    assert [s.title for s in final.sources] == ["Plate tectonics"] and "only looked" not in final.text


def test_a_named_document_is_the_only_one_used(setup):
    final = final_of(run(setup, ScriptedClient([["Plates move ", "[1]", "."]]), "give me an overview of the Plate tectonics handout")[1])
    assert [s.title for s in final.sources] == ["Plate tectonics"] and final.mode == OVERVIEW


def test_a_topic_in_none_of_the_papers_is_still_refused(setup):
    client = ScriptedClient()
    final = final_of(run(setup, client, "summarize the French Revolution paper")[1])
    assert final.status == NOT_IN_PAPERS_STATUS and final.mode == SEARCH and client.stream_calls == []


def test_ordinary_questions_with_document_words_keep_the_normal_path(setup):
    for question in ["what page is the quiz on", "how many items are in the worksheet", "what is the test for acids",
                     "what do my notes say about gravity"]:
        final = final_of(run(setup, ScriptedClient(), question)[1])
        assert final.mode == SEARCH, question


# ---- asking which paper --------------------------------------------------------------------------------

@pytest.fixture
def five(setup):
    add_doc(setup.index, "rocks", "Igneous rocks form from cooled magma. Sedimentary rocks form in layers.", subject="Science", title="Rocks and minerals")
    add_doc(setup.index, "weather", "Weather is the state of the atmosphere. Climate is the long-term average.", subject="Science", title="Weather and climate")
    return setup


def test_more_than_three_papers_ask_which_one_without_calling_the_model(five):
    client = ScriptedClient()
    session, items = run(five, client, "give me a summary of the entire paper")
    final = final_of(items)
    assert final.status == NEEDS_CHOICE and final.sources == [] and client.stream_calls == []
    assert "Which paper do you mean?" in final.text and "Weather and climate" in final.text and "5." in final.text
    assert items[0] == final.text and session.history == []        # nothing to follow up on yet


def test_a_number_picks_the_paper_and_the_original_question_is_answered(five):
    client = ScriptedClient([["Weather is the atmosphere's state ", "[1]", "."]])
    session = five.session(client, gate=STRICT)
    asked = session.ask("give me a summary of the entire paper")
    assert asked.status == NEEDS_CHOICE
    shown_titles = [line.split(" | ")[1] for line in asked.text.splitlines() if line[:1].isdigit()]
    final = session.ask("1")
    assert final.status == ANSWERED and final.mode == OVERVIEW and [s.title for s in final.sources] == [shown_titles[0]]
    assert final.query == "give me a summary of the entire paper"
    assert session.history[-1].question == "give me a summary of the entire paper"


def test_part_of_a_title_picks_the_paper(five):
    client = ScriptedClient([["Rocks form in layers ", "[1]", "."]])
    session = five.session(client, gate=STRICT)
    session.ask("summarize the entire paper")
    final = session.ask("rocks")
    assert [s.title for s in final.sources] == ["Rocks and minerals"]


def test_an_unclear_reply_asks_again_and_a_bad_number_keeps_the_list(five):
    session = five.session(ScriptedClient([["Water ", "[1]", "."]]), gate=STRICT)
    session.ask("summarize the entire paper")
    again = session.ask("77")
    assert again.status == NEEDS_CHOICE and again.text.startswith("There is no number 77")
    several = session.ask("and")                                   # "and" appears in two titles: not a unique choice...
    assert several.status in (NEEDS_CHOICE, ANSWERED, NOT_IN_PAPERS_STATUS)


def test_something_that_is_not_a_choice_is_an_ordinary_question_and_clears_the_wait(five):
    client = ScriptedClient([["The field ", "[1]", "."]])
    session = five.session(client, gate=LAX_GATE)
    session.ask("summarize the entire paper")
    final = session.ask("What generates the magnetic field of Earth?")
    assert final.mode == SEARCH and final.status == ANSWERED and session._pending is None


def test_new_chat_forgets_the_pending_question(five):
    session = five.session(ScriptedClient(), gate=STRICT)
    session.ask("summarize the entire paper")
    assert session._pending is not None
    session.new()
    assert session._pending is None


def test_twenty_papers_list_the_eight_most_recent(setup):
    for i in range(20):
        add_doc(setup.index, f"w{i}", f"Worksheet {i} about topic number {i} and stuff.", title=f"Worksheet {i:02d}")
    final = final_of(run(setup, ScriptedClient(), "summarize the entire paper")[1])
    lines = [ln for ln in final.text.splitlines() if re.match(r"\d+\. ", ln)]
    assert final.status == NEEDS_CHOICE and len(lines) == 8 and "15 more are not shown" in final.text
    assert "Worksheet 19" in lines[0]


# ---- notes, citations and the budget ---------------------------------------------------------------------

def test_every_page_of_the_overview_gets_the_hard_to_read_note_even_if_not_cited(setup):
    add_doc(setup.index, "blurry", "Blurry text about volcanoes and magma flows.", subject="Geography", title="Blurry volcanoes",
            quality=0.4, flags=["low_agreement"])
    names = SubjectList(setup.settings.data_dir / "subjects.json", ["Science", "Geography"])
    client = ScriptedClient([["Plates move ", "[1]", "."]])
    session = setup.session(client, subjects=names, gate=STRICT)
    session.set_subject("Geography")
    final = session.ask("summarize the whole handout")
    assert [s.title for s in final.sources] == ["Plate tectonics", "Blurry volcanoes"] and final.cited == [1]
    assert any("was hard to read" in n and "Blurry volcanoes" in n for n in final.notes)     # the page the answer did not cite
    assert HANDWRITING_NOTE not in final.notes


def test_citations_are_validated_against_the_real_pages(setup):
    final = final_of(run(setup, ScriptedClient([["Earth ", "[1]", " and something ", "[9]", "."]]), "summarize the entire paper")[1])
    assert final.cited == [1] and final.invalid_citations == [9] and "[9]" not in final.text


def test_an_answer_that_cites_nothing_carries_the_usual_caution(setup):
    final = final_of(run(setup, ScriptedClient([["Everything is about Earth."]]), "summarize the entire paper")[1])
    assert final.uncited and any("no source was cited" in n or "wording matches" in n for n in final.notes)


# ---- "this" and the age of the documents ----------------------------------------------------------------

def age_all(setup, minutes, title=None):
    """Pretend every document (or the one titled `title`) was added `minutes` ago."""
    from datetime import datetime, timedelta, timezone
    when = (datetime.now(timezone.utc) - timedelta(minutes=minutes)).isoformat(timespec="seconds")
    if title:
        setup.index.store._db.execute("UPDATE documents SET created_at = ? WHERE title = ?", (when, title))
    else:
        setup.index.store._db.execute("UPDATE documents SET created_at = ?", (when,))


def test_this_means_the_latest_upload_only_while_it_is_recent(setup):
    names = SubjectList(setup.settings.data_dir / "subjects.json", ["Science", "Geography"])
    final = setup.session(ScriptedClient([["Plates move ", "[1]", "."]]), subjects=names, gate=STRICT).ask("summarize this handout")
    assert len(final.sources) == 1 and final.mode == OVERVIEW                                       # added a moment ago: one document, the latest


def test_this_with_a_few_old_documents_summarises_all_of_them(setup):
    age_all(setup, 90)
    final = final_of(run(setup, ScriptedClient([THREE]), "summarize this handout")[1])
    assert final.mode == OVERVIEW and len(final.sources) == 3                                       # up to 3 papers are each summarised


def test_this_with_many_old_documents_asks_which_paper(five):
    age_all(five, 90)
    client = ScriptedClient()
    final = final_of(run(five, client, "summarize this handout")[1])
    assert final.status == NEEDS_CHOICE and "Which paper do you mean?" in final.text and client.stream_calls == []
    for question in ("buod ng papel na ito", "ano ang laman nito"):
        assert final_of(run(five, ScriptedClient(), question)[1]).status == NEEDS_CHOICE, question


def test_this_with_many_documents_and_a_recent_one_uses_the_latest(five):
    age_all(five, 90)
    age_all(five, 5, "Weather and climate")
    final = final_of(run(five, ScriptedClient([["Weather is the state of the atmosphere ", "[1]", "."]]), "summarize this handout")[1])
    assert [s.title for s in final.sources] == ["Weather and climate"]


def long_text(n):
    return " ".join(f"Sentence {i} explains idea number {i} about the planet in detail." for i in range(n))


def test_a_long_document_is_condensed_page_by_page_and_the_student_is_told(settings, tmp_path):
    setup = make_setup(settings, tmp_path)
    try:
        add_doc(setup.index, "long", "", subject="Math", title="Long reviewer", pages=[long_text(60) for _ in range(4)])
        names = SubjectList(setup.settings.data_dir / "subjects.json", ["Science", "Math"])
        client = Spy([["Short points ", "[1]", " and ", "[4]", "."]], context=2400)
        session = setup.session(client, subjects=names, gate=STRICT)
        session.set_subject("Math")
        items = list(session.ask_stream("summarize this reviewer"))
        final = final_of(items)
        progress = [i for i in items if isinstance(i, Progress)]
        assert [p.text for p in progress] == [f"Summarizing page {n} of 4 of 'Long reviewer' first..." for n in range(1, 5)]
        assert final.model_calls_condensing == 4 and len(client.chat_calls) == 4
        assert [s.page_number for s in final.sources] == [1, 2, 3, 4] and all(s.text == "short points" for s in final.sources)
        assert any("summarized each page first" in n for n in final.notes) and final.cited == [1, 4]
    finally:
        setup.index.close()


def test_a_very_long_document_falls_back_to_its_stored_summary_and_says_so(settings, tmp_path):
    setup = make_setup(settings, tmp_path)
    try:
        add_doc(setup.index, "huge", "", subject="Math", title="Huge book", summary="A whole year of algebra, from linear equations to quadratics.",
                pages=[long_text(5) for _ in range(40)])
        names = SubjectList(setup.settings.data_dir / "subjects.json", ["Science", "Math"])
        client = Spy([["Algebra from linear to quadratic ", "[1]", "."]], context=2400)
        session = setup.session(client, subjects=names, gate=STRICT)
        session.set_subject("Math")
        final = session.ask("summarize the whole paper")
        assert len(final.sources) == 1 and final.sources[0].text.startswith("A whole year of algebra") and final.sources[0].page_number == 1
        assert client.chat_calls == [] and final.model_calls_condensing == 0
        assert any("stored summary only" in n and "40 pages" in n for n in final.notes) and final.cited == [1]
    finally:
        setup.index.close()


def test_a_context_too_small_for_the_overview_fails_clearly(setup):
    with pytest.raises(ChatContextError, match="loaded context"):
        run(setup, ScriptedClient(context=1300), "summarize the entire paper")


def test_the_model_can_still_say_there_is_nothing_to_summarize(setup):
    final = final_of(run(setup, ScriptedClient([["NOT_IN_PAPERS"]]), "summarize the entire paper")[1])
    assert final.status == NOT_IN_PAPERS_STATUS


def test_an_empty_library_gives_the_normal_refusal(settings, tmp_path):
    import dataclasses
    from conftest import FakeEmbedder
    from studybuddy.index.index import Index
    s = dataclasses.replace(settings, data_dir=tmp_path / "data", library_dir=tmp_path / "lib")
    index = Index.open(s, FakeEmbedder())
    try:
        from studybuddy.chat.session import ChatSession
        final = ChatSession(index, ScriptedClient(), s, gate=STRICT).ask("summarize the entire paper")
        assert final.status == NOT_IN_PAPERS_STATUS
    finally:
        index.close()
