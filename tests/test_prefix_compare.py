import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from studybuddy.index.prefix_compare import (
    ChunkText, Question, compare, cosine, format_results, is_correct, load_questions, prefixes_for,
)

ROOT = Path(__file__).resolve().parents[1]
DOC, QUERY = "search_document: ", "search_query: "

CHUNKS = [ChunkText("a", "alpha water cycle"), ChunkText("b", "beta magnetic field"), ChunkText("c", "gamma plates")]
QUESTIONS = [Question("where does water go", ("a",)), Question("what makes the field", ("b",), "magnetic")]


class SpyEmbed:
    """Axis per marker word; a text carrying the right prefix gets a bonus on a shared 'prefix' axis."""

    def __init__(self):
        self.calls = []

    def __call__(self, texts):
        self.calls.append(list(texts))
        out = []
        for t in texts:
            low = t.lower()
            v = [float("water" in low or "alpha" in low), float("field" in low or "magnetic" in low or "beta" in low),
                 float("plates" in low or "gamma" in low), 0.0]
            out.append(v)
        return out


def test_cosine():
    assert cosine([1, 0], [1, 0]) == pytest.approx(1.0)
    assert cosine([1, 0], [0, 1]) == 0.0 and cosine([0, 0], [1, 0]) == 0.0


def test_prefixes_for_each_variant():
    assert prefixes_for("none", DOC, QUERY) == ("", "")
    assert prefixes_for("both", DOC, QUERY) == (DOC, QUERY)
    assert prefixes_for("doc-only", DOC, QUERY) == (DOC, "")
    assert prefixes_for("query-only", DOC, QUERY) == ("", QUERY)


def test_is_correct_needs_stem_and_optional_text():
    chunk = ChunkText("b", "Beta MAGNETIC field")
    assert is_correct(chunk, Question("q", ("b",), "magnetic"))
    assert not is_correct(chunk, Question("q", ("b",), "volcano"))
    assert not is_correct(chunk, Question("q", ("a",)))
    assert is_correct(chunk, Question("q", ("a", "b")))


def test_metrics_for_a_perfect_embedder():
    (result,) = compare(CHUNKS, QUESTIONS, SpyEmbed(), DOC, QUERY, variants=("none",))
    assert result.hit_at(1) == 1.0 and result.hit_at(3) == 1.0 and result.mrr == 1.0
    assert result.mean_margin > 0 and [r.rank for r in result.results] == [1, 1]


def test_rank_and_mrr_when_the_right_chunk_is_not_first():
    chunks = [ChunkText("x", "water water water"), ChunkText("a", "alpha water plates")]
    (result,) = compare(chunks, [Question("water", ("a",))], SpyEmbed(), "", "", variants=("none",))
    assert result.results[0].rank == 2 and result.mrr == 0.5 and result.hit_at(1) == 0.0 and result.hit_at(3) == 1.0
    assert result.results[0].margin < 0  # a wrong chunk scored higher


def test_variants_embed_documents_and_questions_with_the_right_prefixes():
    spy = SpyEmbed()
    compare(CHUNKS, QUESTIONS, spy, DOC, QUERY)
    flat = [(t.startswith(DOC), t.startswith(QUERY)) for call in spy.calls for t in call]
    docs_with = [c for c in spy.calls if c and c[0].startswith(DOC)]
    queries_with = [c for c in spy.calls if c and c[0].startswith(QUERY)]
    assert len(docs_with) == 1 and len(queries_with) == 1  # prefixed versions are embedded once and reused
    assert len(spy.calls) == 4  # docs plain, docs prefixed, queries plain, queries prefixed
    assert any(a for a, _ in flat) and any(b for _, b in flat)


def test_unanswerable_question_is_rejected_loudly():
    with pytest.raises(ValueError, match="No chunk can answer"):
        compare(CHUNKS, [Question("q", ("a",), "volcano")], SpyEmbed(), DOC, QUERY)


def test_load_questions_validation(tmp_path):
    good = tmp_path / "q.json"
    good.write_text(json.dumps([{"question": "q1", "expect": "a"}, {"question": "q2", "expect": ["a", "b"],
                                                                    "expect_text": "x"}]), encoding="utf-8")
    qs = load_questions(good)
    assert qs[0].expect == ("a",) and qs[1].expect == ("a", "b") and qs[1].expect_text == "x"
    for bad in ("[]", "{}", '[{"question": "q"}]', '[{"expect": "a"}]'):
        (tmp_path / "bad.json").write_text(bad, encoding="utf-8")
        with pytest.raises(ValueError):
            load_questions(tmp_path / "bad.json")


def test_format_results_table_and_rows():
    results = compare(CHUNKS, QUESTIONS, SpyEmbed(), DOC, QUERY)
    text = format_results(results, len(CHUNKS))
    assert "2 questions over 3 chunks" in text and "small sample" in text
    for name in ("none", "both", "doc-only", "query-only"):
        assert name in text
    assert "hit@1" in text and "MRR" in text and "Per question" in text


def run_script(tmp_path, samples, **env_extra):
    env = {**os.environ, "SB_READER_MODEL": "r", "SB_CHAT_MODEL": "c", "SB_EMBED_MODEL": "e",
           "SB_LIBRARY_DIR": str(tmp_path / "lib"), "SB_DATA_DIR": str(tmp_path / "data"),
           "SB_SAMPLES_DIR": str(samples), "EMBED_DOC_PREFIX": DOC, "EMBED_QUERY_PREFIX": QUERY, **env_extra}
    return subprocess.run([sys.executable, str(ROOT / "scripts" / "compare_prefixes.py")],
                          capture_output=True, text=True, env=env, cwd=tmp_path, timeout=60)


def test_script_without_samples_exits_cleanly(tmp_path):
    out = run_script(tmp_path, tmp_path / "nope")
    assert out.returncode == 0 and "Nothing to compare" in out.stdout


def test_script_refuses_when_no_prefixes_are_configured(tmp_path):
    out = run_script(tmp_path, tmp_path / "nope", EMBED_DOC_PREFIX="", EMBED_QUERY_PREFIX="")
    assert out.returncode == 1 and "both empty" in out.stdout
