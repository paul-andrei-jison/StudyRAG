import dataclasses

import pytest

from conftest import FakeEmbedder
from studybuddy.errors import IndexMismatchError
from studybuddy.index.index import Index
from studybuddy.index.search import build_match, make_snippet, query_terms, rrf, search
from studybuddy.index.store import DocumentMeta
from studybuddy.reader.quality import PageRecord


class MapEmbedder:
    """Embedder with hand-placed vectors: any text containing a marker word gets that marker's direction."""

    def __init__(self, markers, dim=4, query_prefix=""):
        self.markers, self.dim, self.query_prefix = markers, dim, query_prefix
        self.calls = []

    def _vec(self, text):
        v = [0.0] * self.dim
        for word, axis in self.markers.items():
            if word in text.lower():
                v[axis] += 1.0
        return v if any(v) else [0.0] * (self.dim - 1) + [0.01]

    def embed_documents(self, texts):
        self.calls.append(("documents", texts))
        return [self._vec(t) for t in texts]

    def embed_queries(self, texts):
        self.calls.append(("queries", texts))
        return [self._vec(self.query_prefix + t) for t in texts]


def add(index, h, text, subject="Science", title="", path="C:/lib/a.jpg", page=1, **kw):
    rec = PageRecord(page, path, "image", text=text, quality=0.8, flags=kw.get("flags", []))
    return index.add_document(DocumentMeta(h, path, "photo", subject, title), [rec])


@pytest.fixture
def index(settings):
    idx = Index.open(settings, FakeEmbedder())
    yield idx
    idx.close()


# ---- pure helpers --------------------------------------------------------------------------

def test_query_terms_drop_stopwords_but_never_everything():
    assert query_terms("What is the habitable zone?") == ["habitable", "zone"]
    assert query_terms("what is the") == ["what", "is", "the"]  # nothing else left: keep them
    assert query_terms("Earth's 71% water, water") == ["earth", "71", "water"]
    assert query_terms("   ") == []


def test_build_match_quotes_every_term():
    assert build_match("habitable zone") == '"habitable" OR "zone"'
    assert build_match("???") is None
    assert build_match('say "hi"') == '"say" OR "hi"'


def test_rrf_math_and_weights():
    scores = rrf([[1, 2, 3], [3, 1]], k=60)
    assert scores[1] == pytest.approx(1 / 61 + 1 / 62)
    assert scores[2] == pytest.approx(1 / 62)
    assert scores[3] == pytest.approx(1 / 63 + 1 / 61)
    assert sorted(scores, key=scores.get, reverse=True) == [1, 3, 2]
    weighted = rrf([[1], [2]], k=60, weights=[1.0, 3.0])
    assert weighted[2] > weighted[1]
    assert rrf([]) == {} and rrf([[], []]) == {}


def test_snippet_centres_on_the_first_matching_term():
    text = "filler " * 80 + "the habitable zone is special " + "filler " * 80
    snippet = make_snippet(text, ["habitable"])
    assert "habitable zone" in snippet and len(snippet) < 300 and snippet.startswith("...") and snippet.endswith("...")
    assert make_snippet("short text", ["short"]) == "short text"


# ---- retrieval -----------------------------------------------------------------------------

def test_hybrid_returns_complete_hits(index):
    add(index, "a", "The habitable zone is the region where liquid water can exist.", title="Earth notes",
        path="C:/lib/Science/earth.jpg", page=3, flags=["glare"])
    (hit,) = search(index, "habitable zone", k=5)
    assert hit.text.startswith("The habitable zone") and hit.subject == "Science" and hit.title == "Earth notes"
    assert (hit.file_path, hit.image_path, hit.page_number) == ("C:/lib/Science/earth.jpg",) * 2 + (3,)
    assert hit.page_quality == 0.8 and hit.page_flags == ["glare"]
    assert hit.vector_rank == 1 and hit.keyword_rank == 1
    assert 0 < hit.vector_score <= 1 and hit.keyword_score > 0
    assert hit.score == pytest.approx(2 / 61)
    assert "habitable" in hit.snippet


def test_hybrid_beats_each_mode_alone(settings):
    emb = MapEmbedder({"paraphrase": 0, "goldilocks": 1, "filler": 2})
    idx = Index.open(settings, emb)
    add(idx, "b", "A restatement using other words: paraphrase.", title="B")          # vector-near, no keyword
    add(idx, "a", "Earth sits in the Goldilocks zone, neither hot nor cold.", title="A")  # keyword hit, vector-far
    add(idx, "c", "Unrelated filler about rivers and lakes.", title="C")
    query = "paraphrase goldilocks"
    # the query embeds near the paraphrase chunk (axis 0) and the goldilocks chunk (axis 1) equally; skew it
    vector_only = [h.title for h in search(idx, "paraphrase", k=2, mode="vector")]
    keyword_only = [h.title for h in search(idx, "goldilocks", k=2, mode="keyword")]
    hybrid = [h.title for h in search(idx, query, k=2, mode="hybrid")]
    assert vector_only[0] == "B" and keyword_only == ["A"]
    assert set(hybrid) == {"A", "B"}
    idx.close()


def test_keyword_catches_an_exact_term_the_vector_misses(settings):
    emb = MapEmbedder({"water": 0, "magnetic": 1})
    idx = Index.open(settings, emb)
    add(idx, "1", "Liquid water covers most of the planet.", title="water")
    add(idx, "2", "The magnetic field protects the atmosphere. Lithosphere plates move.", title="magnetic")
    # "lithosphere" has no embedding direction (no marker): vector search cannot tell the chunks apart,
    # the keyword index finds the right one exactly
    hybrid = search(idx, "lithosphere", k=1)
    assert hybrid[0].title == "magnetic" and hybrid[0].keyword_rank == 1
    idx.close()


def test_vector_catches_a_paraphrase_the_keywords_miss(settings):
    emb = MapEmbedder({"protective shield": 0, "magnetic": 0, "rivers": 1})
    idx = Index.open(settings, emb)
    add(idx, "1", "Earth has a magnetic field.", title="field")
    add(idx, "2", "Rivers carry water.", title="rivers")
    hits = search(idx, "protective shield", k=2, mode="hybrid")
    assert hits[0].title == "field" and hits[0].keyword_rank is None and hits[0].vector_rank == 1
    idx.close()


def test_a_chunk_found_by_both_lists_outranks_one_found_by_one(settings):
    emb = MapEmbedder({"alpha": 0})
    idx = Index.open(settings, emb)
    add(idx, "both", "alpha beta gamma", title="both")
    add(idx, "vec", "alpha only vector here", title="vec")  # contains alpha -> vector near; keyword also matches "alpha"
    add(idx, "kw", "gamma delta", title="kw")
    hits = search(idx, "alpha gamma", k=3)
    assert hits[0].title == "both"
    idx.close()


def test_subject_filter_works_in_every_mode(index):
    add(index, "s", "Water covers the planet and drives the water cycle.", subject="Science")
    add(index, "h", "Water mills powered medieval towns with water wheels.", subject="History")
    for mode in ("hybrid", "vector", "keyword"):
        both = search(index, "water", k=5, mode=mode)
        assert {h.subject for h in both} == {"Science", "History"}, mode
        only = search(index, "water", k=5, mode=mode, subject="history")
        assert [h.subject for h in only] == ["History"], mode
        assert search(index, "water", k=5, mode=mode, subject="Math") == [], mode


def test_k_limits_results_and_order_is_best_first(index):
    for i in range(6):
        add(index, f"h{i}", f"Water fact number {i}. " + "filler " * i, title=str(i))
    hits = search(index, "water fact", k=3)
    assert len(hits) == 3
    assert [h.score for h in hits] == sorted((h.score for h in hits), reverse=True)


def test_empty_index_blank_query_and_no_keyword_terms(index, settings):
    emb = FakeEmbedder()
    fresh = Index.open(dataclasses.replace(settings, data_dir=settings.data_dir / "other"), emb)
    assert search(fresh, "anything", k=5) == []
    assert search(index, "   ") == [] and search(index, "x", k=0) == []
    assert emb.calls == []  # a blank query costs no embedding call
    add(index, "a", "Some text about plates.")
    assert search(index, "???", k=3, mode="keyword") == []
    fresh.close()


@pytest.mark.parametrize("nasty", ['"', 'AND', 'OR NOT', 'NEAR(a b)', '*', 'a:b', '-x', '(unclosed', 'col:text AND',
                                   "'; DROP TABLE chunks; --", '^start', 'a b" OR "c', '\\', 'x*y'])
def test_hostile_queries_never_break_keyword_search(index, nasty):
    add(index, "a", "Plates move. AND OR NEAR are just words here.")
    search(index, nasty, k=3, mode="keyword")  # must not raise
    search(index, nasty, k=3, mode="hybrid")
    assert index.store.counts()["chunks"] == 1


def test_queries_use_embed_queries_not_embed_documents(settings):
    emb = FakeEmbedder(query_prefix="search_query: ")
    idx = Index.open(settings, emb)
    add(idx, "a", "Plates move slowly.")
    emb.calls.clear()
    search(idx, "how do plates move")
    assert emb.methods() == ["queries"] and emb.calls[0][1] == ["search_query: how do plates move"]
    idx.close()


def test_keyword_mode_does_not_call_the_embedder(index):
    add(index, "a", "Plates move slowly.")
    index.embedder.calls.clear()
    assert [h.keyword_rank for h in search(index, "plates", mode="keyword")] == [1]
    assert index.embedder.calls == []


def test_orphan_vectors_are_skipped_not_cited(index):
    add(index, "a", "Plates move slowly across the mantle.")
    index.vectors.add([9999], [index.embedder.embed_queries(["plates move slowly mantle"])[0]], ["science"], [1])
    ids = [h.chunk_id for h in search(index, "plates move slowly mantle", k=5)]
    assert 9999 not in ids and len(ids) == 1


def test_query_embedding_of_the_wrong_size_points_at_reindex(settings):
    idx = Index.open(settings, FakeEmbedder(dim=64))
    add(idx, "a", "Plates move.")
    idx.embedder = FakeEmbedder(dim=16)
    with pytest.raises(IndexMismatchError, match="reindex"):
        search(idx, "plates")
    idx.close()


def test_unknown_mode_is_rejected(index):
    with pytest.raises(ValueError):
        search(index, "x", mode="semantic")


def test_index_search_convenience_matches_the_function(index):
    add(index, "a", "Plates move slowly.")
    assert [h.chunk_id for h in index.search("plates")] == [h.chunk_id for h in search(index, "plates")]
