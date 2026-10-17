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
