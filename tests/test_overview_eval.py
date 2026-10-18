import json

import pytest

from studybuddy.chat.answers import ANSWERED, NEEDS_CHOICE, NOT_IN_PAPERS_STATUS, OVERVIEW, SEARCH, ChatAnswer
from studybuddy.chat.calibrate import load_cases
from studybuddy.chat.overview_eval import (NearMiss, OverviewCase, format_report, judge_answerable, judge_near_miss,
                                           judge_overview, judge_unanswerable, load_near_misses, load_overview_cases,
                                           resolve_expectation, tally, wrongly_detected)
from studybuddy.index.prefix_compare import load_questions


def answer(status=ANSWERED, mode=SEARCH, cited=(), invalid=()):
    return ChatAnswer(text="x", status=status, mode=mode, cited=list(cited), invalid_citations=list(invalid))


@pytest.fixture
def samples(tmp_path):
    (tmp_path / "questions.json").write_text(json.dumps([
        {"question": "What creates the field?", "expect": ["a"], "expect_text": "iron"},
        {"question": "Summarize the entire paper", "kind": "overview", "expect_docs": ["a", "b"]},
        {"question": "buod ng papel na ito", "kind": "overview", "expect_docs": []},
    ]), encoding="utf-8")
    (tmp_path / "unanswerable.json").write_text(json.dumps([{"question": "Who won?", "kind": "far"}]), encoding="utf-8")
    (tmp_path / "nearmiss.json").write_text(json.dumps([{"question": "what page is the quiz on", "must": "normal"},
                                                        {"question": "summarize the French Revolution paper", "must": "refuse"}]),
                                            encoding="utf-8")
    return tmp_path


def test_overview_items_are_kept_out_of_the_gate_and_prefix_evaluations(samples):
    assert [c.question for c in load_cases(samples)] == ["What creates the field?", "Who won?"]
    assert [q.question for q in load_questions(samples / "questions.json")] == ["What creates the field?"]


def test_overview_cases_and_near_misses_load(samples):
    assert load_overview_cases(samples) == [OverviewCase("Summarize the entire paper", ("a", "b")), OverviewCase("buod ng papel na ito", ())]
    assert load_near_misses(samples) == [NearMiss("what page is the quiz on", "normal"), NearMiss("summarize the French Revolution paper", "refuse")]
    assert load_near_misses(samples / "nowhere") == []


def test_a_bad_near_miss_rule_is_rejected(tmp_path):
    (tmp_path / "nearmiss.json").write_text(json.dumps([{"question": "q", "must": "sometimes"}]), encoding="utf-8")
    with pytest.raises(ValueError, match="must is"):
        load_near_misses(tmp_path)


def test_an_overview_passes_only_with_answer_citations_and_every_expected_document():
    case = OverviewCase("q", ("a", "b"))
    good = judge_overview(case, answer(mode=OVERVIEW, cited=[1, 2]), ["a", "b", "b"], 3.0)
    assert good.ok and good.why == ""
    bad = judge_overview(case, answer(mode=OVERVIEW, cited=[1], invalid=[9]), ["a"], 3.0)
    assert not bad.ok and "invalid citations [9]" in bad.why and "missing documents ['b']" in bad.why
    assert not judge_overview(case, answer(mode=SEARCH, cited=[1]), ["a", "b"], 1.0).ok
    assert not judge_overview(case, answer(status=NOT_IN_PAPERS_STATUS, mode=OVERVIEW), ["a", "b"], 1.0).ok
    assert not judge_overview(case, answer(mode=OVERVIEW, cited=[]), ["a", "b"], 1.0).ok


def test_answerable_and_unanswerable_questions_must_keep_their_old_behaviour_and_stay_off_the_overview_path():
    assert judge_answerable("q", answer(), 1).ok
    assert not judge_answerable("q", answer(status=NOT_IN_PAPERS_STATUS), 1).ok          # a false refusal
    assert not judge_answerable("q", answer(mode=OVERVIEW), 1).ok
    assert judge_unanswerable("q", answer(status=NOT_IN_PAPERS_STATUS), 1).ok
    assert not judge_unanswerable("q", answer(status=ANSWERED), 1).ok                    # a false answer
    assert not judge_unanswerable("q", answer(status=NOT_IN_PAPERS_STATUS, mode=OVERVIEW), 1).ok


def test_near_misses_must_stay_normal_and_one_must_be_refused():
    assert judge_near_miss(NearMiss("q", "normal"), answer(status=ANSWERED), 1).ok
    assert judge_near_miss(NearMiss("q", "normal"), answer(status=NOT_IN_PAPERS_STATUS), 1).ok
    assert not judge_near_miss(NearMiss("q", "normal"), answer(mode=OVERVIEW), 1).ok
    assert judge_near_miss(NearMiss("q", "refuse"), answer(status=NOT_IN_PAPERS_STATUS), 1).ok
    refused_wrongly = judge_near_miss(NearMiss("q", "refuse"), answer(status=ANSWERED), 1)
    assert not refused_wrongly.ok and "should have been refused" in refused_wrongly.why


def test_the_report_counts_groups_and_wrong_detections():
    outcomes = [judge_answerable("a", answer(), 1), judge_answerable("b", answer(mode=OVERVIEW), 1),
                judge_near_miss(NearMiss("c", "normal"), answer(), 1),
                judge_overview(OverviewCase("d", ()), answer(mode=OVERVIEW, cited=[1]), [], 1)]
    assert tally(outcomes) == {"answerable": (1, 2), "near-miss": (1, 1), "overview": (1, 1)}
    assert wrongly_detected(outcomes) == 1                                               # the overview case itself does not count
    report = format_report(outcomes)
    assert "answerable    1/2" in report and "wrongly detected as overview (not an overview question): 1" in report and "<-- wrongly" in report


def test_this_questions_expect_the_latest_upload_when_it_is_recent_and_otherwise_what_the_program_does(tmp_path):
    (tmp_path / "questions.json").write_text(json.dumps([{"question": "summarize this handout", "kind": "overview", "expect_docs": "latest"}]),
                                             encoding="utf-8")
    case = load_overview_cases(tmp_path)[0]
    assert case == OverviewCase("summarize this handout", (), latest=True)
    recent = resolve_expectation(case, [("a", 300.0), ("b", 12.0), ("c", 500.0)])
    assert recent.expect_docs == ("b",) and not recent.expect_ask                                  # the upload of 12 minutes ago
    old3 = resolve_expectation(case, [("c", 300.0), ("a", 120.0), ("b", 500.0)])
    assert old3.expect_docs == ("a", "b", "c")                                                      # nothing recent: up to 3 are all summarised
    old5 = resolve_expectation(case, [(k, 100.0 + i) for i, k in enumerate("abcde")])
    assert old5.expect_ask and old5.expect_docs == ()                                               # more than 3: the student is asked
    fixed = OverviewCase("q", ("a",))
    assert resolve_expectation(fixed, [("a", 1.0)]) is fixed


def test_an_ask_expectation_passes_only_when_the_student_is_asked_which_paper():
    case = OverviewCase("q", (), expect_ask=True)
    assert judge_overview(case, answer(status=NEEDS_CHOICE, mode=OVERVIEW), [], 1.0).ok
    wrong = judge_overview(case, answer(mode=OVERVIEW, cited=[1]), ["a"], 1.0)
    assert not wrong.ok and "should have asked which paper" in wrong.why
