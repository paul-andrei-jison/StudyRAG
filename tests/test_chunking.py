import random
import re

import pytest

from studybuddy.index.chunking import MAX_CHARS, MIN_TAIL_CHARS, TARGET_CHARS, chunk_page


def words(text):
    return re.findall(r"\S+", text)


def para(n, tag="w"):
    return " ".join(f"{tag}{i}" for i in range(n))


def test_blank_page_gives_no_chunks():
    assert chunk_page("", 1) == []
    assert chunk_page("  \n\n \t ", 3) == []


def test_short_page_is_one_chunk():
    (c,) = chunk_page("  Photosynthesis is the process.\n", 4)
    assert (c.page_number, c.chunk_index, c.text) == (4, 0, "Photosynthesis is the process.")


def test_paragraphs_are_packed_up_to_target_and_never_split_when_they_fit():
    paragraphs = [f"P{i} " + para(60, f"p{i}x") for i in range(12)]  # ~400 chars each
    page = "\n\n".join(paragraphs)
    chunks = chunk_page(page, 1, overlap=0)
    assert len(chunks) > 1
    assert all(len(c.text) <= TARGET_CHARS for c in chunks)
    for c in chunks:  # every chunk is a run of whole paragraphs
        assert all(p in paragraphs for p in c.text.split("\n\n"))
    assert [c.chunk_index for c in chunks] == list(range(len(chunks)))


def test_table_is_kept_together_when_it_fits():
    table = "\n".join(["| Object | Mass |", "| --- | --- |", "| Car | 1200 |", "| Book | 0.5 |"])
    page = para(150, "intro") + "\n\n" + table + "\n\n" + para(150, "outro")
    chunks = chunk_page(page, 1, target=700, overlap=0)
    assert any(table in c.text for c in chunks)


def test_oversized_table_splits_on_row_boundaries():
    rows = [f"| row{i} | {'x' * 40} | {i} |" for i in range(80)]
    chunks = chunk_page("\n".join(rows), 1, target=500, overlap=0)
    assert len(chunks) > 3
    for c in chunks:
        assert all(line.startswith("| row") and line.endswith("|") for line in c.text.split("\n"))
    assert [ln for c in chunks for ln in c.text.split("\n")] == rows


def test_long_paragraph_splits_at_sentence_ends():
    sentences = [f"This is sentence number {i} about the water cycle and clouds." for i in range(60)]
    chunks = chunk_page(" ".join(sentences), 1, target=500, overlap=0)
    assert len(chunks) > 3
    assert all(c.text.rstrip().endswith(".") for c in chunks)
    assert all(len(c.text) <= 500 for c in chunks)


def test_unbroken_long_text_splits_on_words_under_the_hard_cap():
    chunks = chunk_page(para(3000), 1)
    assert all(len(c.text) <= MAX_CHARS for c in chunks)
    assert words(" ".join(c.text for c in chunks)).count("w0") >= 1


def test_a_single_huge_token_is_cut_by_characters_under_the_cap():
    chunks = chunk_page("x" * 5000, 1, overlap=0)
    assert all(len(c.text) <= MAX_CHARS for c in chunks)
    assert "".join(c.text for c in chunks) == "x" * 5000


def test_figure_line_stays_with_the_following_block():
    page = "Intro paragraph.\n\n[figure: a block with an arrow]\n\nCaption and labels follow here."
    (c,) = chunk_page(page, 1)
    assert "[figure: a block with an arrow]\nCaption and labels follow here." in c.text
    chunks = chunk_page(para(100, "a") + "\n\n[figure: x]\n\n" + para(100, "b"), 1, target=700, overlap=0)
    assert any("[figure: x]\nb0" in c.text for c in chunks)


def test_trailing_figure_line_is_not_lost():
    assert "[figure: last]" in chunk_page("Text.\n\n[figure: last]", 1)[0].text


def test_overlap_repeats_whole_words_from_the_previous_chunk():
    page = "\n\n".join(para(60, f"q{i}x") for i in range(10))
    chunks = chunk_page(page, 1, overlap=100)
    for prev, nxt in zip(chunks, chunks[1:]):
        head = nxt.text.split("\n", 1)[0]
        assert head and head in prev.text
        assert all(w in words(prev.text) for w in words(head))  # no half words
        assert len(head) <= 100


def test_overlap_never_pushes_a_chunk_over_the_cap():
    big = para(190, "z")  # ~1,000 chars
    page = "\n\n".join([big, big, big])
    for c in chunk_page(page, 1, target=1000, max_chars=1050, overlap=150):
        assert len(c.text) <= 1050


def test_tiny_trailing_chunk_merges_into_the_previous_one():
    page = para(150, "a") + "\n\nShort ending sentence."
    chunks = chunk_page(page, 1, target=900, overlap=0)
    assert len(chunks) == 1 and chunks[0].text.endswith("Short ending sentence.")
    assert MIN_TAIL_CHARS == 200


def test_windows_line_endings():
    chunks = chunk_page("First paragraph.\r\n\r\nSecond paragraph.", 1)
    assert chunks[0].text == "First paragraph.\n\nSecond paragraph."


def test_target_above_max_is_rejected():
    with pytest.raises(ValueError):
        chunk_page("x", 1, target=2000, max_chars=1000)


@pytest.mark.parametrize("seed", range(40))
def test_random_pages_lose_nothing_and_respect_the_cap(seed):
    rng = random.Random(seed)
    vocab = [f"word{i}" for i in range(300)] + ["Earth's", "71%", "CO2.", "end.", "(about", "21%),"]
    blocks = []
    for _ in range(rng.randint(1, 12)):
        kind = rng.random()
        n = rng.choice([3, 20, 120, 400, 900])
        if kind < 0.15:
            blocks.append("\n".join(f"| {rng.choice(vocab)} | {rng.choice(vocab)} |" for _ in range(rng.randint(2, 60))))
        else:
            blocks.append(" ".join(rng.choice(vocab) for _ in range(n)))
    page = ("\n\n" if rng.random() < 0.8 else "\n").join(blocks)

    exact = chunk_page(page, 2, overlap=0)  # no overlap: concatenation must equal the page, word for word
    assert words(" ".join(c.text for c in exact)) == words(page)
    assert all(c.page_number == 2 for c in exact)

    with_overlap = chunk_page(page, 2)  # default overlap: nothing lost, nothing over the cap
    assert all(len(c.text) <= MAX_CHARS for c in with_overlap)
    seen = set(words(" ".join(c.text for c in with_overlap)))
    assert set(words(page)) <= seen
