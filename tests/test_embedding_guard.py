from studybuddy.embedding_guard import META_FILENAME, check_embedding


def test_first_run_records(tmp_path):
    r = check_embedding(tmp_path / "d", "m1", 3)
    assert r.ok and r.recorded
    assert (tmp_path / "d" / META_FILENAME).exists()


def test_same_model_ok(tmp_path):
    check_embedding(tmp_path, "m1", 3)
    r = check_embedding(tmp_path, "m1", 3)
    assert r.ok and not r.recorded


def test_changed_model_flagged(tmp_path):
    check_embedding(tmp_path, "m1", 3)
    r = check_embedding(tmp_path, "m2", 3)
    assert not r.ok and "CHANGED" in r.message and "m1" in r.message


def test_changed_dim_flagged(tmp_path):
    check_embedding(tmp_path, "m1", 3)
    assert not check_embedding(tmp_path, "m1", 4).ok


def test_corrupt_meta_flagged(tmp_path):
    (tmp_path / META_FILENAME).write_text("{not json")
    assert not check_embedding(tmp_path, "m1", 3).ok


def test_doc_prefix_recorded_and_matched(tmp_path):
    from studybuddy.embedding_guard import read_meta
    assert check_embedding(tmp_path, "m1", 3, "search_document: ").recorded
    assert read_meta(tmp_path) == {"model": "m1", "dim": 3, "doc_prefix": "search_document: "}
    r = check_embedding(tmp_path, "m1", 3, "search_document: ")
    assert r.ok and not r.recorded


def test_changed_doc_prefix_flagged(tmp_path):
    check_embedding(tmp_path, "m1", 3, "search_document: ")
    r = check_embedding(tmp_path, "m1", 3, "")
    assert not r.ok and "DOCUMENT PREFIX CHANGED" in r.message and "EMBED_DOC_PREFIX" in r.message


def test_record_without_prefix_is_completed_not_rejected(tmp_path):
    from studybuddy.embedding_guard import read_meta
    check_embedding(tmp_path, "m1", 3)  # e.g. written by check_models.py
    assert "doc_prefix" not in read_meta(tmp_path)
    r = check_embedding(tmp_path, "m1", 3, "search_document: ")
    assert r.ok and r.recorded and read_meta(tmp_path)["doc_prefix"] == "search_document: "
    assert not check_embedding(tmp_path, "m1", 3, "other: ").ok


def test_prefix_not_checked_when_not_given(tmp_path):
    check_embedding(tmp_path, "m1", 3, "search_document: ")
    assert check_embedding(tmp_path, "m1", 3).ok


def test_write_meta_overwrites(tmp_path):
    from studybuddy.embedding_guard import read_meta, write_meta
    check_embedding(tmp_path, "m1", 3, "a: ")
    write_meta(tmp_path, "m2", 5, "b: ")
    assert read_meta(tmp_path) == {"model": "m2", "dim": 5, "doc_prefix": "b: "}
    assert check_embedding(tmp_path, "m2", 5, "b: ").ok
