import json
import os
import time

import pytest
from PIL import Image

from ingest_helpers import FakeChat, FakeReader, make_env, make_photo
from studybuddy.errors import TransientLLMError
from studybuddy.ingest.results import INDEXED
from studybuddy.ingest.service import IngestService
from studybuddy.ui.status import StatusBoard
from studybuddy.ui.uploads import AUTO_DETECT, submit_uploads, sweep_empty_uploads, uploads_dir


def build(tmp_path, settings, started=True, **kw):
    env = make_env(tmp_path, settings, **kw)
    service = IngestService(env.settings, env.index, env.reader, None, env.chat, env.subjects, poll_interval=0.02)
    board = StatusBoard(service)
    if started:
        service.start()
    return env, service, board


@pytest.fixture
def lab(tmp_path, settings):
    env, service, board = build(tmp_path, settings)
    yield env, service, board, tmp_path / "dropped"
    service.stop()
    env.index.close()


def dropped(folder, name, **kw):
    return make_photo(folder / name, **kw)


def make_pdf(path):
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (200, 300), "white").save(path, "PDF")
    return path


def test_one_photo_is_copied_queued_and_filed_with_the_chosen_subject(lab):
    env, service, board, drop = lab
    photo = dropped(drop, "IMG 1.jpg")
    before = photo.read_bytes()
    outcome = submit_uploads(service, [str(photo)], "Science", album=False)
    assert outcome.ok and outcome.queued == 1 and "Queued 1 document" in outcome.message
    result = outcome.tickets[0].result(timeout=30)
    assert result.status == INDEXED and result.subject == "Science" and result.subject_source == "user"
    assert photo.read_bytes() == before                                  # the dropped file is never touched
    assert not [p for p in uploads_dir(service).rglob("*") if p.is_file()]  # our copy was consumed by filing
    assert env.index.documents()[0].subject == "Science"


def test_auto_detect_leaves_the_subject_to_the_worker(lab):
    env, service, board, drop = lab
    outcome = submit_uploads(service, [dropped(drop, "a.jpg")], AUTO_DETECT)
    assert outcome.tickets[0].result(timeout=30).subject_source == "auto"
    assert submit_uploads(service, [dropped(drop, "b.jpg", color=(9, 9, 9))], None).ok


def test_several_files_without_the_album_box_are_separate_documents(lab):
    env, service, board, drop = lab
    files = [dropped(drop, "a.jpg"), dropped(drop, "b.jpg", color=(20, 20, 20)), make_pdf(drop / "c.pdf")]
    outcome = submit_uploads(service, files, "Math")
    assert outcome.ok and outcome.queued == 3
    assert sorted(t.result(timeout=60).page_count for t in outcome.tickets) == [1, 1, 1]
    assert len(env.index.documents()) == 3


def test_an_album_is_one_document_in_natural_name_order(tmp_path, settings):
    env, service, board = build(tmp_path, settings, started=False)   # keep the job in the queue so we can look at it
    drop = tmp_path / "dropped"
    files = [dropped(drop, name, color=(i * 30, 5, 5)) for i, name in enumerate(["img10.jpg", "img2.jpg", "img1.jpg"], 1)]
    outcome = submit_uploads(service, files, "Science", album=True)
    assert outcome.ok and outcome.queued == 1 and "album" in outcome.message
    (job,) = service.jobs.load_all()
    assert job.is_album and job.origin == "ui" and job.subject == "Science" and job.kinds == ["photo-file"] * 3
    assert [os.path.basename(f)[:2] for f in job.files] == ["01", "02", "03"]
    assert [os.path.basename(f)[3:] for f in job.files] == ["img1.jpg", "img2.jpg", "img10.jpg"]
    env.index.close()


@pytest.mark.parametrize("make, expected", [
    (lambda d: [dropped(d, "only.jpg")], "at least two photos"),
    (lambda d: [dropped(d, "a.jpg"), make_pdf(d / "b.pdf")], "only hold photos"),
])
def test_album_rules_are_explained_and_nothing_is_queued(lab, make, expected):
    env, service, board, drop = lab
    outcome = submit_uploads(service, make(drop), "Math", album=True)
    assert not outcome.ok and expected in outcome.message and service.jobs.load_all() == []
    assert not uploads_dir(service).exists() or not list(uploads_dir(service).iterdir())


def test_one_bad_file_in_an_album_stops_the_album(lab):
    env, service, board, drop = lab
    bad = drop / "notes.txt"
    bad.parent.mkdir(parents=True, exist_ok=True)
    bad.write_text("hello", encoding="utf-8")
    outcome = submit_uploads(service, [dropped(drop, "a.jpg"), dropped(drop, "b.jpg"), bad], "Math", album=True)
    assert not outcome.ok and "notes.txt" in outcome.message and service.jobs.load_all() == []


def test_bad_files_are_skipped_with_a_reason_and_good_ones_still_go(lab):
    env, service, board, drop = lab
    drop.mkdir(parents=True, exist_ok=True)
    gif = drop / "anim.gif"
    gif.write_bytes(b"GIF89a" + b"\x00" * 20)
    empty = drop / "empty.jpg"
    empty.write_bytes(b"")
    outcome = submit_uploads(service, [gif, empty, dropped(drop, "ok.jpg")], "Math")
    assert outcome.ok and outcome.queued == 1 and len(outcome.skipped) == 2
    assert "anim.gif" in outcome.message and "GIF" in outcome.message and "empty.jpg" in outcome.message
    assert outcome.tickets[0].result(timeout=30).status == INDEXED


def test_a_file_that_is_not_what_its_name_says_is_judged_by_its_bytes(lab):
    env, service, board, drop = lab
    fake = drop / "photo.jpg"
    fake.parent.mkdir(parents=True, exist_ok=True)
    fake.write_text("<html></html>", encoding="utf-8")
    outcome = submit_uploads(service, [fake], "Math")
    assert not outcome.ok and "unsupported or damaged" in outcome.message


def test_nothing_dropped_and_bad_subjects_are_explained(lab):
    env, service, board, drop = lab
    assert "Drop at least one" in submit_uploads(service, [], "Math").message
    assert "Drop at least one" in submit_uploads(service, None, "Math").message
    outcome = submit_uploads(service, [dropped(drop, "a.jpg")], "CON")
    assert not outcome.ok and "reserved" in outcome.message and service.jobs.load_all() == []


def test_a_typed_new_subject_is_added_only_when_the_document_is_filed(lab):
    env, service, board, drop = lab
    outcome = submit_uploads(service, [dropped(drop, "a.jpg")], "  Astronomy  Club ")
    assert outcome.ok
    result = outcome.tickets[0].result(timeout=30)
    assert result.subject == "Astronomy Club" and "Astronomy Club" in env.subjects.names()


def test_a_stopping_service_refuses_and_leaves_no_copies(tmp_path, settings):
    env, service, board = build(tmp_path, settings)
    service.stop()
    outcome = submit_uploads(service, [dropped(tmp_path / "d", "a.jpg")], "Math")
    assert not outcome.ok and "stopping" in outcome.message
    assert not any(p.is_file() for p in uploads_dir(service).rglob("*"))
    env.index.close()


def test_sweep_removes_only_old_empty_folders(tmp_path):
    root = tmp_path / "uploads"
    (root / "old").mkdir(parents=True)
    (root / "fresh").mkdir()
    (root / "full").mkdir()
    (root / "full" / "x.jpg").write_bytes(b"x")
    now = time.time()
    os.utime(root / "old", (now - 600, now - 600))
    os.utime(root / "full", (now - 600, now - 600))
    assert sweep_empty_uploads(root, grace=60) == 1
    assert sorted(p.name for p in root.iterdir()) == ["fresh", "full"]


# ---- status ----------------------------------------------------------------------------------

def test_status_lists_queued_jobs_and_recent_results(tmp_path, settings):
    env, service, board = build(tmp_path, settings, started=False)
    drop = tmp_path / "d"
    submit_uploads(service, [dropped(drop, "a.jpg"), dropped(drop, "b.jpg", color=(9, 9, 9))], "Math")
    snap = board.snapshot()
    assert len(snap.queued) == 2 and snap.headline() == "Status: 0 reading, 2 queued, 0 waiting to retry, 0 failed"
    assert "Queued:" in snap.jobs_text() and snap.recent_text() == "No results yet in this session."
    service.start()
    try:
        assert service.wait_idle(30)
        done = board.snapshot()
        assert done.queued == [] and len(done.recent) == 2 and done.recent[0].startswith("INDEXED")
        assert done.jobs_text() == "Nothing is waiting."
    finally:
        service.stop()
        env.index.close()


def test_a_deferred_job_shows_its_reason_and_retry_time(tmp_path, settings):
    chat = FakeChat(TransientLLMError("LM Studio is not running"))
    env, service, board = build(tmp_path, settings, chat=chat)
    try:
        outcome = submit_uploads(service, [dropped(tmp_path / "d", "a.jpg")], "Math")
        assert outcome.tickets[0].reported.result(timeout=30).status == "deferred"
        snap = board.snapshot()
        assert len(snap.deferred) == 1 and "a.jpg" in snap.deferred[0] and "retry in" in snap.deferred[0]
        assert "attempt 1" in snap.deferred[0] and snap.headline().count("1 waiting to retry") == 1
        assert snap.recent and snap.recent[0].startswith("DEFERRED")
    finally:
        service.stop()
        env.index.close()


def test_failed_documents_are_listed_and_retry_requeues_them(tmp_path, settings):
    env, service, board = build(tmp_path, settings, reader=FakeReader(texts=[RuntimeError("reader exploded")]))
    try:
        drop = tmp_path / "d"
        outcome = submit_uploads(service, [dropped(drop, "a.jpg")], "Math")
        assert outcome.tickets[0].result(timeout=30).status == "failed"
        snap = board.snapshot()
        assert len(snap.failed) == 1 and "reader exploded" in snap.failed[0] and "failed" in snap.headline()
        assert board.retry_failed().startswith("Requeued 1 of 1")
        assert service.wait_idle(30)
        assert len(env.index.documents()) == 1 and board.snapshot().failed == []
        assert board.retry_failed() == "Nothing has failed."
    finally:
        service.stop()
        env.index.close()


def test_an_unreadable_reason_file_is_still_listed_and_a_missing_file_cannot_be_requeued(tmp_path, settings):
    env, service, board = build(tmp_path, settings, started=False)
    broken = env.data / "failed" / "job-1"
    broken.mkdir(parents=True)
    (broken / "reason.json").write_text("{not json", encoding="utf-8")
    gone = env.data / "failed" / "job-2"
    gone.mkdir()
    (gone / "reason.json").write_text(json.dumps({"files": [str(gone / "missing.jpg")], "error": "bad"}), encoding="utf-8")
    snap = board.snapshot()
    assert len(snap.failed) == 2 and any("unreadable" in f for f in snap.failed)
    assert "could not be requeued" in board.retry_failed()
    env.index.close()
