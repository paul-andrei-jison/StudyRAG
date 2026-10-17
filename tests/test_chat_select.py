from collections import Counter

from studybuddy.chat.select import spread
from studybuddy.index.search import Hit


def hit(chunk_id, doc, page, score=1.0):
    return Hit(chunk_id=chunk_id, doc_id=doc, score=score, text=f"t{chunk_id}", snippet="", subject="S", title=f"D{doc}",
               file_path=f"f{doc}.jpg", page_number=page, image_path=f"f{doc}.jpg", page_quality=1.0, page_flags=[])


def ids(hits):
    return [h.chunk_id for h in hits]


def test_five_chunks_of_one_page_are_not_all_taken_when_other_pages_exist():
    hits = [hit(i, 1, 1) for i in range(1, 6)] + [hit(6, 2, 1), hit(7, 3, 1), hit(8, 2, 2)]
    got = spread(hits)
    assert Counter((h.doc_id, h.page_number) for h in got)[(1, 1)] == 2
    assert {h.doc_id for h in got} == {1, 2, 3} and len(got) == 5
    assert ids(got) == [1, 2, 6, 7, 8]  # best first, still in retrieval order


def test_per_document_limit():
    hits = [hit(1, 1, 1), hit(2, 1, 2), hit(3, 1, 3), hit(4, 1, 4), hit(5, 2, 1), hit(6, 3, 1)]
    got = spread(hits, n=5)
    assert Counter(h.doc_id for h in got) == {1: 3, 2: 1, 3: 1}
    assert ids(got) == [1, 2, 3, 5, 6]


def test_the_limits_are_relaxed_when_the_question_is_about_one_page():
    hits = [hit(i, 1, 1) for i in range(1, 8)]
    got = spread(hits)
    assert ids(got) == [1, 2, 3, 4, 5]  # nothing else to choose from: the best five of that page, not just two


def test_relaxation_prefers_the_best_skipped_hits():
    hits = [hit(1, 1, 1), hit(2, 1, 1), hit(3, 1, 1), hit(4, 1, 1), hit(5, 2, 1)]
    got = spread(hits, n=4)
    assert ids(got) == [1, 2, 3, 5]  # 5 (another document) beats the lower-ranked 4 from the crowded page
    assert ids(spread(hits, n=5)) == [1, 2, 3, 4, 5]


def test_fewer_hits_than_places():
    assert ids(spread([hit(1, 1, 1), hit(2, 2, 1)], n=5)) == [1, 2]
    assert spread([], n=5) == [] and spread([hit(1, 1, 1)], n=0) == []


def test_order_is_always_best_first_and_input_is_not_modified():
    hits = [hit(1, 2, 1), hit(2, 1, 1), hit(3, 1, 1), hit(4, 1, 1), hit(5, 3, 1)]
    before = list(hits)
    got = spread(hits, n=4, per_page=1, per_doc=1)
    assert ids(got) == sorted(ids(got)) == [1, 2, 3, 5] and hits == before  # increasing id = decreasing score here


def test_custom_limits():
    hits = [hit(i, 1, 1) for i in range(1, 5)] + [hit(5, 2, 1)]
    assert ids(spread(hits, n=3, per_page=1, per_doc=1)) == [1, 2, 5]  # 1 and 5 by the limits, 2 fills the last place
