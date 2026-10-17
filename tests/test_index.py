import dataclasses

import pytest

from conftest import FakeEmbedder
from studybuddy.embedding_guard import META_FILENAME, read_meta, write_meta
from studybuddy.errors import IndexMismatchError, IndexStoreError, LLMError
from studybuddy.index.index import Index, reset_index
from studybuddy.index.store import DocumentMeta
from studybuddy.reader.quality import PageRecord

PARA = "Photosynthesis converts light energy into chemical energy inside the chloroplasts of green plants."


def rec(n, text=PARA, path="C:/lib/Science/a.jpg", kind="image", **kw):
    return PageRecord(n, path, kind, text=text, quality=0.9, **kw)


def meta(h="h1", subject="Science", title="Worksheet", path="C:/lib/Science/a.jpg"):
    return DocumentMeta(content_hash=h, file_path=path, source_kind="photo", subject=subject, title=title)


@pytest.fixture
def emb():
    return FakeEmbedder()


@pytest.fixture
def index(settings, emb):
    idx = Index.open(settings, emb)
    yield idx
    idx.close()


def test_add_document_stores_pages_chunks_and_vectors(index):
    res = index.add_document(meta(), [rec(1), rec(2, "Rivers carry water to the sea.")], expected_pages=2)
    assert res.created and res.chunk_count == 2
    assert index.store.counts() == {"documents": 1, "pages": 2, "chunks": 2}
    assert index.vectors.count() == 2 and index.check().ok
    (doc,) = index.documents()
    assert (doc.subject, doc.title, doc.page_count) == ("Science", "Worksheet", 2)


def test_every_chunk_has_the_original_image_path(index):
    index.add_document(meta(), [rec(1, path="C:/lib/Science/a.jpg"), rec(2, path="C:/lib/Science/a.jpg")])
    ids = sorted(index.store.chunk_ids())
    assert all(c.image_path == "C:/lib/Science/a.jpg" for c in index.store.chunks_by_id(ids).values())


def test_empty_and_failed_pages_are_kept_but_not_chunked(index):
    pages = [rec(1), rec(2, text="", flags=["empty"]), rec(3, text="partial", error="ReaderError: cut off")]
    index.add_document(meta(), pages, expected_pages=3)
    doc_id = index.documents()[0].id
    rows = index.store.pages(doc_id)
    assert [r["page_number"] for r in rows] == [1, 2, 3] and rows[2]["error"] == "ReaderError: cut off"
    assert index.store.counts()["chunks"] == 1  # only page 1 has usable text


def test_expected_page_count_mismatch_writes_nothing(index, emb):
    with pytest.raises(IndexStoreError, match="Page mismatch"):
        index.add_document(meta(), [rec(1)], expected_pages=2)
    assert index.store.counts()["documents"] == 0 and emb.calls == []


def test_documents_are_embedded_with_embed_documents_only(index, emb):
    index.add_document(meta(), [rec(1)])
    assert emb.methods() == ["documents"]


def test_prefix_is_applied_to_stored_chunks(settings):
    emb = FakeEmbedder(doc_prefix="search_document: ")
    idx = Index.open(dataclasses.replace(settings, embed_doc_prefix="search_document: "), emb)
    idx.add_document(meta(), [rec(1)])
    assert emb.calls[0][1][0].startswith("search_document: Photosynthesis")
    idx.close()


def test_duplicate_content_hash_is_not_embedded_again(index, emb):
    first = index.add_document(meta(), [rec(1)])
    again = index.add_document(meta(), [rec(1)])
    assert again.doc_id == first.doc_id and not again.created and again.chunk_count == 0
    assert len(emb.calls) == 1 and index.store.counts()["documents"] == 1


def test_embedding_failure_leaves_no_trace(settings):
    idx = Index.open(settings, FakeEmbedder(fail_after_calls=0))
    with pytest.raises(LLMError):
        idx.add_document(meta(), [rec(1)])
    assert idx.store.counts() == {"documents": 0, "pages": 0, "chunks": 0} and idx.vectors.count() == 0
    idx.close()


def test_vector_write_failure_rolls_the_sqlite_rows_back(index, monkeypatch):
    def boom(*a, **k):
        raise RuntimeError("disk full")
    monkeypatch.setattr(index.vectors, "add", boom)
    with pytest.raises(RuntimeError):
        index.add_document(meta(), [rec(1)])
    assert index.store.counts() == {"documents": 0, "pages": 0, "chunks": 0}
    monkeypatch.undo()
    assert index.add_document(meta(), [rec(1)]).created  # and the same document can be added afterwards


def test_delete_document_removes_from_both_stores(index):
    a = index.add_document(meta("a"), [rec(1)])
    index.add_document(meta("b"), [rec(1, "Rivers carry water.")])
    assert index.delete_document(a.doc_id) == 1
    assert index.store.counts()["documents"] == 1 and index.vectors.count() == 1 and index.check().ok


def test_set_subject_updates_sqlite_and_the_vector_filter(index, emb):
    res = index.add_document(meta(subject="Unsorted"), [rec(1)])
    index.set_subject(res.doc_id, "Earth Science")
    qv = emb.embed_queries(["photosynthesis"])[0]
    assert len(index.vectors.search(qv, 5, subject_key="earth science")) == 1
    assert index.vectors.search(qv, 5, subject_key="unsorted") == []
    assert index.documents()[0].subject == "Earth Science"


# ---- embedding guard -----------------------------------------------------------------------

def test_reopening_with_another_model_is_refused_with_a_hint(settings, emb):
    idx = Index.open(settings, emb)
    idx.add_document(meta(), [rec(1)])
    idx.close()
    with pytest.raises(IndexMismatchError, match="EMBEDDING MODEL CHANGED.*index_admin.py reindex"):
        Index.open(dataclasses.replace(settings, embed_model="other-model"), emb)
    Index.open(dataclasses.replace(settings, embed_model="other-model"), emb, check_guard=False).close()


def test_changed_document_prefix_is_refused_but_query_prefix_is_not(settings, emb):
    idx = Index.open(settings, emb)
    idx.add_document(meta(), [rec(1)])
    idx.close()
    with pytest.raises(IndexMismatchError, match="DOCUMENT PREFIX CHANGED"):
        Index.open(dataclasses.replace(settings, embed_doc_prefix="search_document: "), emb)
    Index.open(dataclasses.replace(settings, embed_query_prefix="search_query: "), emb).close()


def test_a_different_vector_size_is_refused_on_open(settings, emb):
    idx = Index.open(settings, emb)
    idx.add_document(meta(), [rec(1)])
    idx.close()
    write_meta(settings.data_dir, settings.embed_model, 999, settings.embed_doc_prefix)
    with pytest.raises(IndexMismatchError):
        Index.open(settings, emb)


def test_a_dimension_change_is_caught_at_write_time(settings, emb):
    idx = Index.open(settings, emb)
    idx.add_document(meta("a"), [rec(1)])
    idx.embedder = FakeEmbedder(dim=16)  # same model name, different size: must not be mixed in
    with pytest.raises(IndexMismatchError):
        idx.add_document(meta("b"), [rec(1, "Other words entirely.")])
    assert idx.store.counts()["documents"] == 1
    idx.close()


def test_stale_record_on_an_empty_index_is_replaced(settings, emb):
    write_meta(settings.data_dir, "some-old-model", 768)  # e.g. left by check_models.py
    idx = Index.open(settings, emb)  # nothing indexed yet: nothing to protect
    idx.add_document(meta(), [rec(1)])
    assert read_meta(settings.data_dir) == {"model": "embed-m", "dim": 64, "doc_prefix": ""}
    idx.close()


# ---- consistency ---------------------------------------------------------------------------

def test_check_finds_orphan_and_missing_vectors_and_repair_fixes_both(index, emb):
    index.add_document(meta(), [rec(1), rec(2, "Rivers carry water to the sea.")])
    ids = sorted(index.store.chunk_ids())
    index.vectors.delete([ids[0]])                                   # a chunk lost its vector
    index.vectors.add([9999], [emb.embed_documents(["ghost"])[0]], ["x"], [1])  # a vector with no chunk
    found = index.check()
    assert found.missing_vectors == {ids[0]} and found.orphan_vectors == {9999} and not found.ok
    index.repair()
    assert index.check().ok and index.vectors.count() == 2
    assert 9999 not in index.vectors.ids() and ids[0] in index.vectors.ids()


def test_status_reports_counts_and_the_recorded_model(index):
    index.add_document(meta(), [rec(1)])
    s = index.status()
    assert (s.documents, s.pages, s.chunks, s.vectors, s.dim) == (1, 1, 1, 1, 64)
    assert s.recorded_model == "embed-m" and s.configured_model == "embed-m" and s.guard_problem is None
    assert s.consistency.ok


def test_status_flags_a_model_change_without_raising(settings, emb):
    idx = Index.open(settings, emb)
    idx.add_document(meta(), [rec(1)])
    idx.close()
    changed = Index.open(dataclasses.replace(settings, embed_model="new-model"), emb, check_guard=False)
    assert "reindex" in changed.status().guard_problem
    changed.close()


# ---- reindex and reset ---------------------------------------------------------------------

def test_reindex_re_embeds_everything_with_the_new_model(settings, emb):
    idx = Index.open(settings, emb)
    for i in range(3):
        idx.add_document(meta(f"h{i}"), [rec(1, f"{PARA} Variant number {i} of the text.")])
    idx.close()

    new_settings = dataclasses.replace(settings, embed_model="bigger-model", embed_doc_prefix="doc: ")
    new_emb = FakeEmbedder(dim=32, doc_prefix="doc: ")
    idx = Index.open(new_settings, new_emb, check_guard=False)
    progress = []
    assert idx.reindex(progress=lambda done, total: progress.append((done, total))) == 3
    assert progress[-1] == (3, 3)
    assert idx.vectors.dim == 32 and idx.vectors.count() == 3 and idx.check().ok
    assert all(t.startswith("doc: ") for _, texts in new_emb.calls for t in texts)
    assert read_meta(settings.data_dir) == {"model": "bigger-model", "dim": 32, "doc_prefix": "doc: "}
    idx.close()
    Index.open(new_settings, new_emb).close()              # the new configuration is accepted now
    with pytest.raises(IndexMismatchError):
        Index.open(settings, emb)                          # and the old one is not


def test_reindex_failure_keeps_the_old_index_untouched(settings, emb, tmp_path):
    idx = Index.open(settings, emb)
    for i in range(3):  # > REINDEX_BATCH chunks in total so a failure can strike mid-way
        idx.add_document(meta(f"h{i}"), [rec(n, f"{PARA} Page {n} of document {i} with unique words {n}{i}.")
                                         for n in range(1, 15)])
    idx.close()
    new_settings = dataclasses.replace(settings, embed_model="bigger-model")
    failing = FakeEmbedder(dim=32, fail_after_calls=1)
    idx = Index.open(new_settings, failing, check_guard=False)
    before = idx.vectors.count()
    with pytest.raises(LLMError):
        idx.reindex()
    assert idx.vectors.dim == 64 and idx.vectors.count() == before and idx.check().ok
    assert read_meta(settings.data_dir)["model"] == "embed-m"
    assert not (settings.data_dir / "index" / "vectors.new").exists()
    idx.close()
    Index.open(settings, emb).close()  # the old configuration still opens


def test_reindex_of_an_empty_index_is_a_no_op(index):
    assert index.reindex() == 0 and index.vectors.count() == 0


def test_reset_deletes_the_index_but_never_the_library(settings, emb):
    idx = Index.open(settings, emb)
    idx.add_document(meta(), [rec(1)])
    idx.close()
    settings.library_dir.mkdir(parents=True, exist_ok=True)
    photo = settings.library_dir / "keep-me.jpg"
    photo.write_bytes(b"original")
    removed = reset_index(settings)
    assert not (settings.data_dir / "index").exists() and not (settings.data_dir / META_FILENAME).exists()
    assert len(removed) == 2 and photo.read_bytes() == b"original"
    fresh = Index.open(settings, emb)
    assert fresh.store.counts()["documents"] == 0
    fresh.close()
    assert reset_index(settings) == [settings.data_dir / "index"]  # the fresh open recreated the (empty) index folder
    assert reset_index(settings) == []  # nothing left: reset is idempotent


def test_concurrent_writes_straight_to_the_index_stay_consistent(settings):
    """Even without the ingest service in front, the index serialises its own writes."""
    import threading
    idx = Index.open(settings, FakeEmbedder())
    errors = []

    def add(i):
        try:
            idx.add_document(meta(f"h{i}", title=f"Doc {i}"), [rec(1, f"{PARA} Distinct words number {i} here.")])
        except Exception as exc:  # pragma: no cover - the assertion below reports it
            errors.append(exc)

    threads = [threading.Thread(target=add, args=(i,)) for i in range(15)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert errors == []
    assert idx.store.counts() == {"documents": 15, "pages": 15, "chunks": 15} and idx.vectors.count() == 15
    assert idx.check().ok
    idx.close()
