import pytest

from studybuddy.errors import IndexStoreError
from studybuddy.index.vectors import VectorStore


@pytest.fixture
def vs(tmp_path):
    store = VectorStore(tmp_path / "vec", dim=4)
    store.add([1, 2, 3, 4], [[1, 0, 0, 0], [0.9, 0.1, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0]],
              ["science", "history", "science", "history"], [10, 20, 10, 20])
    return store


def test_cosine_similarity_ranking(vs):
    hits = vs.search([1, 0, 0, 0], 10)
    assert [h[0] for h in hits][:2] == [1, 2]
    assert hits[0][1] == pytest.approx(1.0, abs=1e-5)
    assert hits[1][1] == pytest.approx(0.9939, abs=1e-3)
    assert hits[2][1] == pytest.approx(0.0, abs=1e-5)  # orthogonal


def test_search_respects_n(vs):
    assert len(vs.search([1, 0, 0, 0], 2)) == 2
    assert vs.search([1, 0, 0, 0], 0) == []


def test_subject_filter_applies_before_ranking(vs):
    # the best history chunk (id 2) would not be in the top 1 overall; with the filter it is the winner
    assert [h[0] for h in vs.search([1, 0, 0, 0], 1, subject_key="history")] == [2]
    assert {h[0] for h in vs.search([1, 0, 0, 0], 10, subject_key="science")} == {1, 3}
    assert vs.search([1, 0, 0, 0], 10, subject_key="nothing") == []


def test_subject_with_a_quote_is_escaped(tmp_path):
    store = VectorStore(tmp_path / "v", dim=2)
    store.add([1], [[1, 0]], ["o'brien's class"], [1])
    assert [h[0] for h in store.search([1, 0], 5, subject_key="o'brien's class")] == [1]
    assert store.search([1, 0], 5, subject_key="x' OR '1'='1") == []


def test_delete_and_count_and_ids(vs):
    assert vs.count() == 4 and vs.ids() == {1, 2, 3, 4}
    vs.delete([1, 3])
    assert vs.count() == 2 and vs.ids() == {2, 4}
    assert 1 not in [h[0] for h in vs.search([1, 0, 0, 0], 10)]
    vs.delete([])  # no-op


def test_update_subject_moves_every_chunk_of_a_document(vs):
    vs.update_subject(10, "earth science")
    assert {h[0] for h in vs.search([1, 0, 0, 0], 10, subject_key="earth science")} == {1, 3}
    assert {h[0] for h in vs.search([1, 0, 0, 0], 10, subject_key="science")} == set()


def test_persists_across_reopen_and_reads_dimension(vs, tmp_path):
    reopened = VectorStore(tmp_path / "vec")
    assert reopened.dim == 4 and reopened.count() == 4
    assert reopened.search([0, 1, 0, 0], 1)[0][0] == 3


def test_dimension_conflicts_are_rejected(vs, tmp_path):
    with pytest.raises(IndexStoreError, match="4 dimensions"):
        VectorStore(tmp_path / "vec", dim=8)
    with pytest.raises(IndexStoreError, match="3 dimensions"):
        vs.add([9], [[1, 0, 0]], ["x"], [9])
    with pytest.raises(IndexStoreError, match="Query vector"):
        vs.search([1, 0], 3)
    assert vs.count() == 4  # the bad add wrote nothing


def test_mismatched_lengths_are_rejected(vs):
    with pytest.raises(IndexStoreError, match="same length"):
        vs.add([9, 10], [[1, 0, 0, 0]], ["x"], [9])


def test_empty_store_searches_to_nothing_and_dimension_comes_from_first_add(tmp_path):
    store = VectorStore(tmp_path / "lazy")
    assert store.dim is None and store.count() == 0 and store.ids() == set()
    assert store.search([1, 0], 5) == []
    store.add([1], [[1, 0, 0]], ["a"], [1])
    assert store.dim == 3 and store.count() == 1
    store.add([], [], [], [])  # empty add is fine


def test_empty_created_table_searches_to_nothing(tmp_path):
    store = VectorStore(tmp_path / "e", dim=3)
    assert store.search([1, 0, 0], 5) == [] and store.count() == 0
