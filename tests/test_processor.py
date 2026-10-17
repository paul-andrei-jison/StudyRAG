import json
import sqlite3
from pathlib import Path

import pytest
from PIL import Image

from conftest import FakeEmbedder
from ingest_helpers import GOOD_JSON, PAGE_TEXT, FakeChat, FakeReader, make_env, make_photo, tree
from studybuddy.errors import LLMError, ReaderError, TransientLLMError
from studybuddy.ingest import filing as filing_mod
from studybuddy.ingest.files import sha256_file
from studybuddy.ingest.filing import Filer
from studybuddy.ingest.results import DEFERRED, DUPLICATE, FAILED, INDEXED
from test_pdf import make_scanned_pdf, make_text_pdf


@pytest.fixture
def env(tmp_path, settings):
    return make_env(tmp_path, settings)


def assert_nothing_indexed_or_filed(env):
    assert env.counts() == {"documents": 0, "pages": 0, "chunks": 0}
    assert env.index.vectors.count() == 0 and env.index.check().ok
    assert [p for p in tree(env.library) if "/" in p or not (env.library / p).is_dir()] == []  # no files in the library


# ---- success ---------------------------------------------------------------------------------

def test_a_photo_is_read_titled_indexed_and_filed(env):
    src = make_photo(env.inbox / "IMG_2041.jpg")
    original = src.read_bytes()
    result = env.processor.process(env.job(src, subject="Science"))
    assert result.status == INDEXED and result.ok and result.error is None
    assert (result.subject, result.subject_source, result.title) == ("Science", "user", "Photosynthesis Worksheet")
    assert result.summary.startswith("Explains how plants") and result.page_count == 1 and result.flagged_pages == []
    filed = Path(result.final_path)
    assert filed == env.library / "Science" / "2026-10-17_1425_Science_photosynthesis-worksheet.jpg"
    assert filed.read_bytes() == original and not src.exists()  # verified copy filed, inbox copy removed
    assert env.counts() == {"documents": 1, "pages": 1, "chunks": 1} and env.index.vectors.count() == 1
    (doc,) = env.index.documents()
    assert (doc.file_path, doc.title, doc.subject, doc.source_kind) == (str(filed), "Photosynthesis Worksheet", "Science", "photo-file")
    assert result.doc_id == doc.id and result.source_kinds == ["photo-file"] and result.duration_seconds >= 0


def test_chunks_point_at_the_filed_original_not_the_inbox_copy(env):
    src = make_photo(env.inbox / "a.jpg")
    result = env.processor.process(env.job(src, subject="Science"))
    hits = env.index.search("photosynthesis chloroplasts")
    assert hits and all(h.image_path == result.final_path and h.file_path == result.final_path for h in hits)
    assert hits[0].page_number == 1 and hits[0].subject == "Science"


def test_a_typed_subject_is_added_to_the_list(env):
    result = env.processor.process(env.job(make_photo(env.inbox / "a.jpg"), subject="Earth Science"))
    assert result.subject == "Earth Science" and "Earth Science" in env.subjects.names()
    assert (env.library / "Earth Science").is_dir()


def test_subject_spelling_follows_the_list(env):
    result = env.processor.process(env.job(make_photo(env.inbox / "a.jpg"), subject="science"))
    assert result.subject == "Science" and env.subjects.names() == ["Math", "Science", "English"]


def test_auto_detect_picks_from_the_list(env):
    result = env.processor.process(env.job(make_photo(env.inbox / "a.jpg")))
    assert (result.subject, result.subject_source) == ("Science", "auto")
    assert Path(result.final_path).parent.name == "Science"


def test_a_made_up_subject_goes_to_unsorted_and_never_creates_a_folder(env):
    env.chat.default = json.dumps({"title": "Plate Tectonics", "summary": "s", "subject": "Geology"})
    result = env.processor.process(env.job(make_photo(env.inbox / "a.jpg")))
    assert (result.subject, result.subject_source) == ("Unsorted", "fallback")
    assert any("Geology" in w for w in result.warnings)
    assert sorted(p.name for p in env.library.iterdir()) == ["Unsorted"] and env.subjects.names() == ["Math", "Science", "English"]


def test_flagged_pages_are_reported_with_reasons_but_still_indexed(env):
    env.reader.default = "x"  # almost nothing read from a whole page
    src = make_photo(env.inbox / "a.jpg", size=(3000, 4000))
    result = env.processor.process(env.job(src, subject="Math"))
    assert result.status == INDEXED
    (flagged,) = result.flagged_pages
    assert flagged.page_number == 1 and "too_short" in flagged.flags
    assert any("very little text" in r for r in flagged.reasons)


def test_album_becomes_one_document_with_a_folder_and_a_pdf(env):
    pages = [make_photo(env.inbox / "Science" / "trip" / f"{i}.jpg", color=(i * 40, 90, 90)) for i in (1, 2, 3)]
    env.reader.texts = [f"Page {i}. " + PAGE_TEXT for i in (1, 2, 3)]
    result = env.processor.process(env.job(*pages, subject="Science", album=True, kinds=["photo"] * 3))
    assert result.status == INDEXED and result.page_count == 3
    folder = Path(result.final_path)
    assert folder.is_dir() and folder.parent == env.library / "Science"
    assert sorted(p.name for p in folder.iterdir()) == [folder.name + ".pdf", "page-01.jpg", "page-02.jpg", "page-03.jpg"]
    assert result.pdf_path == str(folder / (folder.name + ".pdf"))
    assert env.counts()["pages"] == 3 and env.index.documents()[0].source_kind == "photo"
    chunk_paths = {c.image_path for c in env.index.store.chunks_by_id(sorted(env.index.store.chunk_ids())).values()}
    assert chunk_paths == {str(folder / f"page-0{i}.jpg") for i in (1, 2, 3)}  # each page cites its own photo
    assert not any(p.exists() for p in pages) and not (env.inbox / "Science" / "trip").exists()  # emptied album folder removed


def test_album_order_matters_for_duplicate_detection(env):
    a, b = make_photo(env.inbox / "a.jpg", color=(1, 1, 1)), make_photo(env.inbox / "b.jpg", color=(2, 2, 2))
    keep_a, keep_b = a.read_bytes(), b.read_bytes()
    first = env.processor.process(env.job(a, b, album=True, subject="Math", cleanup=False))
    assert first.status == INDEXED
    same = env.processor.process(env.job(a, b, album=True, subject="Math", cleanup=False))
    assert same.status == DUPLICATE
    reordered = env.processor.process(env.job(b, a, album=True, subject="Math", cleanup=False))
    assert reordered.status == INDEXED and env.counts()["documents"] == 2
    assert a.read_bytes() == keep_a and b.read_bytes() == keep_b


def test_text_layer_pdf_skips_the_reader(env):
    pdf = env.inbox / "notes.pdf"
    make_text_pdf(pdf, ["Photosynthesis happens in chloroplasts of plant cells", "Stomata let gases in and out of leaves"])
    result = env.processor.process(env.job(pdf, subject="Science", kinds=["pdf"]))
    assert result.status == INDEXED and result.page_count == 2 and env.reader.calls == 0
    assert Path(result.final_path).suffix == ".pdf" and env.index.documents()[0].source_kind == "pdf"
    assert {r["source_kind"] for r in env.index.store.pages(result.doc_id)} == {"pdf-text"}


def test_scanned_pdf_pages_go_through_the_reader(env):
    pdf = env.inbox / "scan.pdf"
    make_scanned_pdf(pdf, 3)
    result = env.processor.process(env.job(pdf, subject="Science", kinds=["pdf"]))
    assert result.status == INDEXED and result.page_count == 3 and env.reader.calls == 3
    assert {r["image_path"] for r in env.index.store.pages(result.doc_id)} == {result.final_path}  # PDF path + page number


def test_documents_kind_is_detected_from_the_bytes(env):
    pdf = env.inbox / "from-telegram.bin"
    make_text_pdf(pdf, ["A text layer long enough to be used directly"])
    result = env.processor.process(env.job(pdf, subject="Science", kinds=["document"]))
    assert result.status == INDEXED and env.index.documents()[0].source_kind == "pdf"
    assert result.final_path.endswith(".pdf")


def test_cleanup_sources_false_leaves_the_source_alone(env):
    src = make_photo(env.inbox / "a.jpg")
    result = env.processor.process(env.job(src, subject="Math", cleanup=False))
    assert result.status == INDEXED and src.exists() and Path(result.final_path).exists()


def test_windows_hostile_title_becomes_a_valid_filename(env):
    env.chat.default = json.dumps({"title": 'CON: "Forces/Motion"?', "summary": "s", "subject": "Science"})
    result = env.processor.process(env.job(make_photo(env.inbox / "a.jpg")))
    assert Path(result.final_path).name == "2026-10-17_1425_Science_con-forces-motion.jpg" and Path(result.final_path).exists()


def test_two_documents_with_the_same_title_never_overwrite_each_other(env):
    a = make_photo(env.inbox / "a.jpg", color=(1, 2, 3))
    b = make_photo(env.inbox / "b.jpg", color=(9, 8, 7))
    r1, r2 = env.processor.process(env.job(a)), env.processor.process(env.job(b))
    assert r1.final_path != r2.final_path and Path(r2.final_path).name.endswith("_2.jpg")
    assert env.counts()["documents"] == 2


# ---- duplicates ------------------------------------------------------------------------------

def test_a_duplicate_is_reported_moved_aside_and_costs_no_model_call(env):
    first = make_photo(env.inbox / "a.jpg")
    env.processor.process(env.job(first, subject="Science"))
    calls = (env.reader.calls, env.chat.calls, len(env.embedder.calls))
    again = make_photo(env.inbox / "again.jpg")  # byte-identical content
    result = env.processor.process(env.job(again, subject="Science"))
    assert result.status == DUPLICATE and result.title == "Photosynthesis Worksheet"
    assert result.existing["title"] == "Photosynthesis Worksheet" and result.existing["file_exists"] is True
    assert result.existing["file_path"] == result.final_path
    assert (env.reader.calls, env.chat.calls, len(env.embedder.calls)) == calls  # nothing was re-read
    assert env.counts()["documents"] == 1 and not again.exists()
    kept = list((env.data / "duplicates").iterdir())
    assert len(kept) == 1 and kept[0].name.endswith("_again.jpg") and result.duplicate_dir == str(env.data / "duplicates")
    assert sha256_file(kept[0]) == sha256_file(Path(result.final_path))  # nothing was thrown away


def test_duplicate_whose_library_copy_is_missing_warns_and_keeps_the_new_copy(env):
    env.processor.process(env.job(make_photo(env.inbox / "a.jpg"), subject="Science"))
    Path(env.index.documents()[0].file_path).unlink()
    result = env.processor.process(env.job(make_photo(env.inbox / "b.jpg"), subject="Science"))
    assert result.status == DUPLICATE and result.existing["file_exists"] is False
    assert any("missing from the library" in w for w in result.warnings)
    assert len(list((env.data / "duplicates").iterdir())) == 1


# ---- all or nothing --------------------------------------------------------------------------

def test_a_page_that_cannot_be_read_stops_the_whole_document_permanently(env):
    pages = [make_photo(env.inbox / f"{i}.jpg", color=(i * 30, 5, 5)) for i in (1, 2, 3)]
    env.reader.texts = [PAGE_TEXT, ReaderError("cut off at 4096 tokens"), PAGE_TEXT]
    result = env.processor.process(env.job(*pages, subject="Math", album=True))
    assert result.status == FAILED and result.transient is False
    assert "1 of 3 page(s) could not be read" in result.error and "page 2" in result.error and "cut off" in result.error
    assert_nothing_indexed_or_filed(env)
    assert env.chat.calls == 0  # it never got as far as describing the document


def test_a_transient_page_failure_defers_the_document_and_leaves_the_files_in_place(env):
    src = make_photo(env.inbox / "a.jpg")
    env.reader.texts = [TransientLLMError("Cannot reach LM Studio")]
    result = env.processor.process(env.job(src, subject="Math"))
    assert result.status == DEFERRED and result.transient is True and "LM Studio" in result.error
    assert src.exists() and not (env.data / "failed").exists()
    assert_nothing_indexed_or_filed(env)


def test_a_transient_title_failure_defers(env):
    src = make_photo(env.inbox / "a.jpg")
    env.chat.replies = [TransientLLMError("busy")]
    result = env.processor.process(env.job(src, subject="Math"))
    assert result.status == DEFERRED and src.exists()
    assert_nothing_indexed_or_filed(env)


def test_a_filing_failure_leaves_no_partial_document(env):
    src = make_photo(env.inbox / "a.jpg")

    def failing_copy(s, d):
        raise OSError("disk full")

    env.processor.filer = Filer(env.library, failing_copy)
    result = env.processor.process(env.job(src, subject="Math"))
    assert result.status == DEFERRED and "disk full" in result.error and src.exists()
    assert_nothing_indexed_or_filed(env)
    assert tree(env.library) == []  # even the empty subject folder was cleaned up


def test_an_embedding_failure_after_filing_rolls_the_filed_copy_back(env):
    src = make_photo(env.inbox / "a.jpg")
    env.index.embedder = FakeEmbedder(fail_after_calls=0)
    result = env.processor.process(env.job(src, subject="Math"))
    assert result.status == DEFERRED and src.exists()
    assert_nothing_indexed_or_filed(env)
    assert tree(env.library) == []


def test_a_vector_write_failure_removes_the_row_and_the_filed_copy(env, monkeypatch):
    src = make_photo(env.inbox / "a.jpg")

    def boom(*a, **k):
        raise sqlite3.OperationalError("database is locked")

    monkeypatch.setattr(env.index.vectors, "add", boom)
    result = env.processor.process(env.job(src, subject="Math"))
    assert result.status == DEFERRED and "database is locked" in result.error and src.exists()
    assert env.counts() == {"documents": 0, "pages": 0, "chunks": 0} and tree(env.library) == []


def test_an_embedding_model_mismatch_defers_with_a_reindex_hint(env, monkeypatch):
    from studybuddy.errors import IndexMismatchError

    def mismatch(*a, **k):
        raise IndexMismatchError("EMBEDDING MODEL CHANGED. Run: python scripts/index_admin.py reindex")

    monkeypatch.setattr(env.index, "add_document", mismatch)
    result = env.processor.process(env.job(make_photo(env.inbox / "a.jpg"), subject="Math"))
    assert result.status == DEFERRED and "reindex" in result.error and "stays queued" in result.error
    assert tree(env.library) == []


def test_the_staged_files_are_recorded_before_the_index_write(env, monkeypatch):
    seen = {}
    real_add = env.index.add_document

    def spy(meta, records, expected_pages=None):
        seen["staged_at_index_time"] = list(env.persisted[-1])
        return real_add(meta, records, expected_pages=expected_pages)

    monkeypatch.setattr(env.index, "add_document", spy)
    job = env.job(make_photo(env.inbox / "a.jpg"), subject="Math")
    result = env.processor.process(job)
    assert seen["staged_at_index_time"] == [result.final_path]  # a crash here could clean exactly this file up
    assert job.staged == [] and env.persisted[-1] == []


def test_leftovers_from_a_crashed_attempt_are_cleaned_up_first(env):
    src = make_photo(env.inbox / "a.jpg")
    leftover = env.library / "Math" / "2026-10-17_1425_Math_photosynthesis-worksheet.jpg"
    leftover.parent.mkdir(parents=True)
    leftover.write_bytes(src.read_bytes())  # filed by the attempt that crashed before indexing
    job = env.job(src, subject="Math")
    job.staged = [str(leftover)]
    result = env.processor.process(job)
    assert result.status == INDEXED and Path(result.final_path) == leftover  # same name again, not "_2"
    assert len(list((env.library / "Math").iterdir())) == 1


# ---- permanent failures ----------------------------------------------------------------------

@pytest.mark.parametrize("name,data,reason", [
    ("corrupt.jpg", b"\xff\xd8\xff\xe0 truncated garbage", "damaged or unreadable"),
    ("anim.gif", b"GIF89a\x01\x00\x01\x00", "GIF"),
    ("empty.jpg", b"", "empty"),
    ("notes.docx", b"PK\x03\x04 word document", "unsupported"),
])
def test_unusable_files_fail_permanently_into_failed_with_a_reason(env, name, data, reason):
    src = env.inbox / name
    src.write_bytes(data)
    result = env.processor.process(env.job(src, subject="Math", origin="inbox"))
    assert result.status == FAILED and result.transient is False and reason in result.error
    failed = Path(result.failed_dir)
    assert failed == env.data / "failed" / result.job_id
    assert (failed / name).read_bytes() == data and not src.exists()  # kept, byte for byte
    info = json.loads((failed / "reason.json").read_text())
    assert info["kind"] == "permanent" and reason in info["error"] and info["subject"] == "Math" and info["origin"] == "inbox"
    assert info["files"] == [str(failed / name)] and info["original_files"] == [str(src.resolve())]
    assert_nothing_indexed_or_filed(env)
    assert env.reader.calls == 0 and env.chat.calls == 0


def test_a_source_that_vanished_is_a_permanent_failure(env):
    result = env.processor.process(env.job(env.inbox / "gone.jpg", subject="Math"))
    assert result.status == FAILED and "no longer there" in result.error


def test_an_invalid_typed_subject_fails_clearly(env):
    src = make_photo(env.inbox / "a.jpg")
    result = env.processor.process(env.job(src, subject="CON"))
    assert result.status == FAILED and "reserved" in result.error
    assert_nothing_indexed_or_filed(env)
    assert (Path(result.failed_dir) / "a.jpg").exists()


def test_an_unexpected_bug_is_a_permanent_failure_with_a_traceback_not_a_crash(env):
    env.chat.replies = [KeyError("boom")]
    result = env.processor.process(env.job(make_photo(env.inbox / "a.jpg"), subject="Math"))
    assert result.status == FAILED and result.error.startswith("unexpected error: KeyError")
    detail = json.loads((Path(result.failed_dir) / "reason.json").read_text())["detail"]
    assert "Traceback" in detail and "KeyError" in detail


def test_a_non_transient_model_error_fails_permanently(env):
    env.index.embedder = type("Bad", (), {"embed_documents": lambda self, t: (_ for _ in ()).throw(LLMError("returned 400: input too long")),
                                          "embed_queries": lambda self, t: []})()
    result = env.processor.process(env.job(make_photo(env.inbox / "a.jpg"), subject="Math"))
    assert result.status == FAILED and "input too long" in result.error
    assert tree(env.library) == []


def test_failed_permanent_job_with_cleanup_false_keeps_files_where_they_are(env):
    src = env.inbox / "anim.gif"
    src.write_bytes(b"GIF89a\x01\x00")
    result = env.processor.process(env.job(src, subject="Math", cleanup=False))
    assert result.status == FAILED and src.exists()
    assert (Path(result.failed_dir) / "reason.json").exists() and not (Path(result.failed_dir) / "anim.gif").exists()


def test_the_result_for_a_failed_job_still_lists_the_submitted_files(env):
    src = env.inbox / "x.gif"
    src.write_bytes(b"GIF89a\x01\x00")
    job = env.job(src, kinds=["photo"])
    result = env.processor.process(job)
    assert result.files == [str(src.resolve())] and result.source_kinds == ["photo"] and result.attempts == 1


def test_subject_folders_in_the_inbox_are_kept_but_emptied_album_folders_are_removed(env):
    loose = make_photo(env.inbox / "Science" / "a.jpg", color=(1, 1, 1))
    album = [make_photo(env.inbox / "Math" / "set" / "part1" / "1.jpg", color=(2, 2, 2)),
             make_photo(env.inbox / "Math" / "set" / "part2" / "2.jpg", color=(3, 3, 3))]
    env.processor.process(env.job(loose, subject="Science"))
    env.processor.process(env.job(*album, subject="Math", album=True))
    assert (env.inbox / "Science").is_dir() and (env.inbox / "Math").is_dir()  # the student's subject folders stay
    assert not (env.inbox / "Math" / "set").exists()  # the emptied album folder and its sub-folders are gone


def test_an_album_folder_that_still_holds_other_files_is_kept(env):
    pages = [make_photo(env.inbox / "Math" / "set" / f"{i}.jpg", color=(i * 40, 3, 3)) for i in (1, 2)]
    note = env.inbox / "Math" / "set" / "readme-for-me.txt"
    note.write_text("do not lose me")
    env.processor.process(env.job(*pages, subject="Math", album=True))
    assert note.read_text() == "do not lose me"
