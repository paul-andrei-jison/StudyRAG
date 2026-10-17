import dataclasses
import json
import threading

import pytest

from studybuddy.ingest.subjects import UNSORTED, SubjectError, SubjectList, clean_subject_name

DEFAULTS = ["Math", "Science", "English"]


@pytest.fixture
def subjects(tmp_path):
    return SubjectList(tmp_path / "subjects.json", DEFAULTS)


def test_seeded_from_defaults_and_persisted(tmp_path):
    s = SubjectList(tmp_path / "s.json", DEFAULTS)
    assert s.names() == DEFAULTS and json.loads((tmp_path / "s.json").read_text()) == DEFAULTS
    assert SubjectList(tmp_path / "s.json").names() == DEFAULTS  # reopened, no defaults needed


def test_existing_file_wins_over_defaults(tmp_path):
    (tmp_path / "s.json").write_text('["Chemistry"]')
    assert SubjectList(tmp_path / "s.json", DEFAULTS).names() == ["Chemistry"]


def test_find_is_case_insensitive_and_returns_the_canonical_spelling(subjects):
    assert subjects.find("science") == "Science" and subjects.find("  MATH ") == "Math"
    assert subjects.find("History") is None and subjects.find("") is None
    assert subjects.find("unsorted") == UNSORTED  # the fallback always exists


def test_choices_end_with_the_fallback(subjects):
    assert subjects.choices() == DEFAULTS + [UNSORTED]


def test_add_new_subject_is_saved(subjects):
    assert subjects.add("  Earth   Science ") == "Earth Science"
    assert "Earth Science" in SubjectList(subjects.path).names()


def test_add_existing_subject_keeps_one_entry_with_the_original_spelling(subjects):
    assert subjects.add("SCIENCE") == "Science"
    assert subjects.names() == DEFAULTS


def test_add_cleans_invalid_characters(subjects):
    assert subjects.add("Earth/Space: notes?") == "Earth Space notes"


def test_unicode_names_are_kept(subjects):
    assert subjects.add("Araling Panlipunan") == "Araling Panlipunan"
    assert subjects.add("Géographie") == "Géographie"


@pytest.mark.parametrize("bad", ["", "   ", "???", "Unsorted", "unsorted", "_Unsorted", "failed", "Inbox", "CON", "nul", "x" * 41])
def test_bad_subject_names_are_rejected_not_guessed(subjects, bad):
    with pytest.raises(SubjectError):
        subjects.add(bad)
    assert subjects.names() == DEFAULTS


def test_clean_subject_name():
    assert clean_subject_name("  Math ") == "Math"
    with pytest.raises(SubjectError, match="reserved"):
        clean_subject_name("Unsorted")
    with pytest.raises(SubjectError, match="40 characters"):
        clean_subject_name("y" * 50)


def test_resolve(subjects):
    assert subjects.resolve("math") == "Math"
    with pytest.raises(SubjectError, match="Unknown subject"):
        subjects.resolve("History")
    assert subjects.resolve("History", add_if_new=True) == "History" and "History" in subjects.names()
    assert subjects.resolve("history") == "History"


def test_hand_edits_to_the_file_are_picked_up_on_add(subjects):
    subjects.path.write_text('["Math", "Science", "English", "Art"]')
    subjects.add("Music")
    assert SubjectList(subjects.path).names() == ["Math", "Science", "English", "Art", "Music"]


def test_corrupt_or_wrong_shape_file_is_an_error_not_overwritten(tmp_path):
    bad = tmp_path / "s.json"
    bad.write_text("{not json")
    with pytest.raises(SubjectError, match="not valid JSON"):
        SubjectList(bad, DEFAULTS)
    assert bad.read_text() == "{not json"
    bad.write_text('{"a": 1}')
    with pytest.raises(SubjectError, match="JSON list"):
        SubjectList(bad)
    bad.write_text('["Math", ""]')
    with pytest.raises(SubjectError):
        SubjectList(bad)


def test_duplicate_entries_in_the_file_collapse(tmp_path):
    (tmp_path / "s.json").write_text('["Math", "math", "Science"]')
    assert SubjectList(tmp_path / "s.json").names() == ["Math", "Science"]


def test_write_is_atomic_and_leaves_no_temp_file(subjects):
    subjects.add("Art")
    assert sorted(p.name for p in subjects.path.parent.iterdir()) == ["subjects.json"]


def test_concurrent_adds_lose_nothing(subjects):
    names = [f"Subject {i}" for i in range(25)]
    threads = [threading.Thread(target=subjects.add, args=(n,)) for n in names]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert set(SubjectList(subjects.path).names()) == set(DEFAULTS) | set(names)


@pytest.mark.parametrize("answer,expected", [
    ("Science", "Science"), ("science", "Science"), ("  Science.  ", "Science"), ("**Science**", "Science"),
    ('"Math"', "Math"), ("Subject: English", "English"), ("The subject is Science", "Science"),
    ("Science\nBecause the text discusses cells", "Science"),
    ("History", UNSORTED), ("Earth Science", UNSORTED), ("Science and Math", UNSORTED), ("", UNSORTED),
    ("Unsorted", UNSORTED), ("I am not sure", UNSORTED),
])
def test_model_answers_map_to_the_list_or_unsorted(subjects, answer, expected):
    assert subjects.match_model_answer(answer) == expected
    assert subjects.names() == DEFAULTS  # auto-detect never adds a subject


def test_from_settings_inside_the_repo_uses_a_data_dir_copy(settings, tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    template = repo / "subjects.json"
    template.write_text('["Math", "Science"]')
    s = dataclasses.replace(settings, subjects_file=template, data_dir=tmp_path / "data")
    live = SubjectList.from_settings(s, repo_root=repo)
    live.add("Art")
    assert live.path == tmp_path / "data" / "subjects.json"
    assert json.loads(template.read_text()) == ["Math", "Science"]  # the tracked template is untouched
    assert SubjectList.from_settings(s, repo_root=repo).names() == ["Math", "Science", "Art"]


def test_from_settings_outside_the_repo_uses_that_file_directly(settings, tmp_path):
    mine = tmp_path / "mine" / "subjects.json"
    mine.parent.mkdir()
    mine.write_text('["Chemistry"]')
    s = dataclasses.replace(settings, subjects_file=mine, data_dir=tmp_path / "data")
    live = SubjectList.from_settings(s, repo_root=tmp_path / "repo")
    assert live.path == mine and live.names() == ["Chemistry"]
    live.add("Physics")
    assert json.loads(mine.read_text()) == ["Chemistry", "Physics"]


def test_from_settings_with_no_template_starts_empty(settings, tmp_path):
    s = dataclasses.replace(settings, subjects_file=tmp_path / "repo" / "nope.json", data_dir=tmp_path / "data")
    assert SubjectList.from_settings(s, repo_root=tmp_path / "repo").names() == []
