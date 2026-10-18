import threading
from pathlib import Path

import pytest

from chat_helpers import add_doc
from ingest_helpers import make_env, make_photo
from studybuddy.ingest.results import INDEXED
from studybuddy.ingest.service import IngestService
from studybuddy.ui.files import LibraryGuard, PathNotAllowedError
from studybuddy.ui.library import BUSY_MESSAGE, TABLE_HEADERS, LibraryView


@pytest.fixture
def lab(tmp_path, settings):
    env = make_env(tmp_path, settings)
    service = IngestService(env.settings, env.index, env.reader, None, env.chat, env.subjects, poll_interval=0.02)
    service.start()
    view = LibraryView(service, LibraryGuard(env.library))
    yield env, service, view
    service.stop()
    env.index.close()


def file_one(env, service, name="a.jpg", subject="Math", **kw):
    result = service.submit([make_photo(env.inbox / name, **kw)], subject=subject).result(timeout=20)
    assert result.status == INDEXED
    return result


def test_rows_describe_each_document_newest_first(lab):
    env, service, view = lab
    first = file_one(env, service, "a.jpg", "Math")
    second = file_one(env, service, "b.jpg", "Science", color=(9, 9, 9))
    rows = view.rows()
    assert [r.id for r in rows] == [second.doc_id, first.doc_id]
    row = rows[0]
    assert (row.subject, row.title, row.pages, row.flagged) == ("Science", "Photosynthesis Worksheet", 1, 0)
    assert row.summary.startswith("Explains how plants") and row.path == second.final_path
    assert len(row.as_list()) == len(TABLE_HEADERS) and len(row.added) == len("2026-10-18 12:00")
    assert [r.id for r in view.rows("science")] == [second.doc_id]   # the subject filter ignores case
    assert view.table()[0][0] == second.doc_id


def test_flagged_pages_are_counted_and_explained_in_plain_language(lab):
    env, service, view = lab
    doc_id = add_doc(env.index, "blurry", "Tectonic plates slowly move.", subject="Geography", title="Blurry",
                           pages=["Tectonic plates slowly move.", "Fine second page about the mantle."], quality=0.4,
                           flags=["low_agreement", "ocr_failed"]).doc_id
    row = next(r for r in view.rows() if r.id == doc_id)
    assert row.flagged == 2
    detail = view.detail(doc_id)
    text = detail.describe()
    assert "Flagged pages:" in text and "the two readers disagree a lot" in text and "ocr_failed" not in text
    assert "page 1: reading quality 0.40" in text
    assert view.detail(9999) is None


def test_a_document_with_no_flags_says_so(lab):
    env, service, view = lab
    result = file_one(env, service)
    assert "No pages were flagged." in view.detail(result.doc_id).describe()


def test_a_page_opens_through_the_guard(lab):
    env, service, view = lab
    result = file_one(env, service, size=(300, 400))
    assert view.page_image(result.doc_id, 1).size == (300, 400)
    with pytest.raises(Exception, match="no page 2"):
        view.page_image(result.doc_id, 2)


def test_a_page_whose_stored_path_leaves_the_library_is_refused(lab, tmp_path):
    env, service, view = lab
    outside = make_photo(tmp_path / "elsewhere" / "secret.jpg")
    doc_id = add_doc(env.index, "sneaky", "Some text about plates.", path=str(outside)).doc_id
    with pytest.raises(PathNotAllowedError, match="outside the library"):
        view.page_image(doc_id, 1)


def test_change_subject_updates_documents_and_search_but_not_the_file(lab):
    env, service, view = lab
    result = file_one(env, service, subject="Math")
    filed = Path(result.final_path)
    outcome = view.change_subject(result.doc_id, "science")
    assert outcome.ok and "now filed under Science" in outcome.message and str(filed) in outcome.message
    assert env.index.documents()[0].subject == "Science"
    assert filed.exists() and filed.parent.name == "Math"                      # nothing was moved
    hits = env.index.search("photosynthesis chloroplasts glucose", k=3, subject="Science")
    assert hits and not env.index.search("photosynthesis chloroplasts glucose", k=3, subject="Math")


def test_a_new_subject_joins_the_list_only_when_the_change_worked(lab):
    env, service, view = lab
    result = file_one(env, service)
    stale = view.change_subject(99999, "Astronomy")
    assert not stale.ok and "no longer exists" in stale.message and "Astronomy" not in env.subjects.names()
    assert view.change_subject(result.doc_id, "  astronomy club ").ok
    assert "astronomy club" in env.subjects.names()


@pytest.mark.parametrize("bad", ["", "   ", "CON", "x" * 60])
def test_bad_subject_names_are_explained_and_change_nothing(lab, bad):
    env, service, view = lab
    result = file_one(env, service)
    outcome = view.change_subject(result.doc_id, bad)
    assert not outcome.ok and outcome.message and not outcome.busy
    assert env.index.documents()[0].subject == "Math"


def test_delete_needs_confirmation_and_keeps_the_original_file(lab):
    env, service, view = lab
    result = file_one(env, service)
    filed = Path(result.final_path)
    assert not view.delete(result.doc_id).ok and len(env.index.documents()) == 1
    outcome = view.delete(result.doc_id, confirmed=True)
    assert outcome.ok and "removed from the index" in outcome.message and "still in your library" in outcome.message
    assert env.index.documents() == [] and filed.exists()
    assert not env.index.search("photosynthesis", k=3)
    again = view.delete(result.doc_id, confirmed=True)
    assert not again.ok and "no longer exists" in again.message


def test_a_removed_document_can_be_added_again(lab):
    env, service, view = lab
    result = file_one(env, service)
    assert view.delete(result.doc_id, confirmed=True).ok
    again = service.submit([make_photo(env.inbox / "a.jpg")], subject="Math").result(timeout=20)
    assert again.status == INDEXED


def test_changes_say_busy_while_a_document_is_being_read_and_do_not_run_later(tmp_path, settings):
    env = make_env(tmp_path, settings)
    service = IngestService(env.settings, env.index, env.reader, None, env.chat, env.subjects, poll_interval=0.02)
    view = LibraryView(service, LibraryGuard(env.library))
    real = service.processor.process
    started, release = threading.Event(), threading.Event()

    def slow(job):
        started.set()
        assert release.wait(30)
        return real(job)

    service.processor.process = slow
    service.start()
    try:
        first = service.submit([make_photo(env.inbox / "a.jpg")], subject="Math")
        assert started.wait(10)
        doc_id = add_doc(env.index, "other", "Volcano text about magma.", subject="Geography", title="Volcanoes").doc_id
        for outcome in (view.change_subject(doc_id, "Science"), view.delete(doc_id, confirmed=True)):
            assert not outcome.ok and outcome.busy and outcome.message == BUSY_MESSAGE
        assert service.status()["exclusive_waiting"] == 0
        release.set()
        first.result(timeout=30)
        assert service.wait_idle(10)
        assert next(d for d in env.index.documents() if d.id == doc_id).subject == "Geography"  # neither refused action ran later
    finally:
        release.set()
        service.stop()
        env.index.close()
