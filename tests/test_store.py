import pytest

from studybuddy.errors import IndexStoreError
from studybuddy.index.store import ChunkData, DocumentMeta, PageData, Store, subject_key


@pytest.fixture
def store(tmp_path):
    s = Store(tmp_path / "index" / "t.db")
    yield s
    s.close()


def meta(h="h1", subject="Science", title="Photosynthesis worksheet", path="C:/lib/Science/a.jpg"):
    return DocumentMeta(content_hash=h, file_path=path, source_kind="photo", subject=subject, title=title)


def pages(n=2, base="C:/lib/Science/a.jpg"):
    return [PageData(i, f"{base}#{i}" if n > 1 else base, "image", f"page {i} text", quality=0.9, flags=["glare"])
            for i in range(1, n + 1)]


def chunk(page, idx, text):
    return ChunkData(page, idx, text)


def add(store, h="h1", subject="Science", title="Photosynthesis worksheet", texts=("chloroplasts absorb sunlight",)):
    chunks = [chunk(1, i, t) for i, t in enumerate(texts)]
    return store.add_document(meta(h, subject, title), pages(1), chunks)


def test_add_and_read_back_everything(store):
    doc_id, chunk_ids, created = store.add_document(
        meta(), pages(2), [chunk(1, 0, "alpha text"), chunk(2, 0, "beta text"), chunk(2, 1, "gamma text")])
    assert created and len(chunk_ids) == 3
    assert store.counts() == {"documents": 1, "pages": 2, "chunks": 3}
    (doc,) = store.documents()
    assert (doc.id, doc.page_count, doc.chunk_count, doc.subject, doc.title) == (doc_id, 2, 3, "Science", "Photosynthesis worksheet")
    rows = store.chunks_by_id(chunk_ids)
    assert rows[chunk_ids[1]].page_number == 2 and rows[chunk_ids[1]].text == "beta text"
    assert rows[chunk_ids[1]].file_path == "C:/lib/Science/a.jpg"
    assert rows[chunk_ids[1]].page_flags == ["glare"] and rows[chunk_ids[1]].page_quality == 0.9


def test_every_chunk_carries_the_original_image_path(store):
    _, ids, _ = store.add_document(meta(), pages(2), [chunk(1, 0, "a"), chunk(2, 0, "b")])
    got = store.chunks_by_id(ids)
    assert got[ids[0]].image_path == "C:/lib/Science/a.jpg#1" and got[ids[1]].image_path == "C:/lib/Science/a.jpg#2"
    assert all(c.image_path for c in got.values())


def test_every_input_page_is_stored_even_empty_or_failed(store):
    ps = [PageData(1, "p", "image", "text"), PageData(2, "p", "image", "", quality=0.0, flags=["empty"]),
          PageData(3, "p", "image", "", quality=0.0, flags=["error"], error="boom")]
    doc_id, _, _ = store.add_document(meta(), ps, [chunk(1, 0, "text")])
    rows = store.pages(doc_id)
    assert [r["page_number"] for r in rows] == [1, 2, 3] and rows[2]["error"] == "boom"
    assert store.documents()[0].page_count == 3


def test_duplicate_hash_writes_nothing(store):
    first, ids, created = add(store)
    again, ids2, created2 = add(store, title="other title")
    assert created and not created2 and again == first and ids2 == []
    assert store.counts()["chunks"] == len(ids) and store.documents()[0].title == "Photosynthesis worksheet"
    assert store.find_by_hash("h1") == first and store.find_by_hash("nope") is None


def test_failed_add_rolls_back_completely(store):
    with pytest.raises(IndexStoreError):
        store.add_document(meta(), pages(1), [chunk(5, 0, "refers to a page that does not exist")])
    with pytest.raises(IndexStoreError):
        store.add_document(meta("h2"), [PageData(1, "p", "image", "a"), PageData(1, "p", "image", "b")], [])
    assert store.counts() == {"documents": 0, "pages": 0, "chunks": 0}


def test_keyword_search_stems_and_ranks(store):
    _, ids, _ = add(store, texts=("Photosynthesis converts light energy", "The stomata exchange gases", "Rivers carry water"))
    hits = store.keyword_search('"convert"', 10)
    assert [h[0] for h in hits] == [ids[0]] and hits[0][1] > 0  # porter stemming: convert ~ converts
    assert store.keyword_search('"nothing"', 10) == []


def test_keyword_search_matches_the_title_column_too(store):
    _, ids, _ = add(store, title="Goldilocks zone notes", texts=("plain unrelated text",))
    assert [h[0] for h in store.keyword_search('"goldilocks"', 10)] == [ids[0]]


def test_subject_filter_is_case_insensitive(store):
    _, a, _ = add(store, "h1", "Science", texts=("water cycle",))
    _, b, _ = add(store, "h2", "History", texts=("water mills",))
    assert {h[0] for h in store.keyword_search('"water"', 10)} == {a[0], b[0]}
    assert [h[0] for h in store.keyword_search('"water"', 10, subject="science")] == [a[0]]
    assert [h[0] for h in store.keyword_search('"water"', 10, subject=" HISTORY ")] == [b[0]]
    assert subject_key(" Science ") == "science" and subject_key(None) == ""


def test_delete_removes_pages_chunks_and_fts(store):
    doc_id, ids, _ = add(store, texts=("alpha beta", "gamma delta"))
    assert store.delete_document(doc_id) == ids
    assert store.counts() == {"documents": 0, "pages": 0, "chunks": 0}
    assert store.keyword_search('"alpha"', 10) == []


def test_ids_are_not_reused_after_delete(store):
    doc_id, ids, _ = add(store, texts=("one",))
    store.delete_document(doc_id)
    _, new_ids, _ = add(store, "h9", texts=("two",))
    assert new_ids[0] > ids[0]  # stale vectors for a deleted chunk can never alias a new one


def test_update_subject_and_title_propagate_to_chunks_and_fts(store):
    doc_id, ids, _ = add(store, subject="", title="", texts=("lithosphere plates",))
    assert store.keyword_search('"lithosphere"', 10, subject="Earth Science") == []
    assert store.update_document(doc_id, subject="Earth Science", title="Plate tectonics") == ids
    assert [h[0] for h in store.keyword_search('"lithosphere"', 10, subject="earth science")] == [ids[0]]
    assert [h[0] for h in store.keyword_search('"tectonics"', 10)] == [ids[0]]  # title is indexed
    assert store.chunks_by_id(ids)[ids[0]].title == "Plate tectonics"
    store.update_document(doc_id, summary="s", file_path="D:/moved.jpg")
    assert store.documents()[0].summary == "s" and store.documents()[0].file_path == "D:/moved.jpg"
    with pytest.raises(IndexStoreError):
        store.update_document(999, title="x")


def test_all_chunks_and_ids(store):
    _, ids, _ = add(store, texts=("a b", "c d"))
    assert [c[0] for c in store.all_chunks()] == ids and store.chunk_ids() == set(ids)
    assert store.all_chunks()[0][1:] == ("a b", "science", store.documents()[0].id)


def test_schema_version_is_checked_and_reopen_keeps_data(tmp_path):
    path = tmp_path / "x.db"
    s = Store(path)
    add(s)
    s.close()
    s2 = Store(path)
    assert s2.counts()["documents"] == 1
    s2._db.execute("PRAGMA user_version = 99")
    s2.close()
    with pytest.raises(IndexStoreError, match="schema version"):
        Store(path)


def test_fts_survives_special_characters_in_stored_text(store):
    _, ids, _ = add(store, texts=('He said "hi" AND (left) NEAR* 100% -done',))
    assert [h[0] for h in store.keyword_search('"done"', 10)] == [ids[0]]


def test_failure_in_the_middle_of_a_write_rolls_back_the_document(store):
    import sqlite3
    bad_pages = [PageData(1, "p", "image", "fine"), PageData(2, "p", "image", None)]  # NOT NULL violation on page 2
    with pytest.raises(sqlite3.IntegrityError):
        store.add_document(meta("h5"), bad_pages, [chunk(1, 0, "fine")])
    assert store.counts() == {"documents": 0, "pages": 0, "chunks": 0}
    assert store.find_by_hash("h5") is None
    assert store.keyword_search('"fine"', 10) == []
    add(store, "h5")  # and the store is still usable
    assert store.counts()["documents"] == 1
