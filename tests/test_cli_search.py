import dataclasses
import io
import subprocess
import sys
from pathlib import Path

from conftest import FakeEmbedder
from studybuddy.index.cli import admin_main
from studybuddy.index.search_cli import search_main

ROOT = Path(__file__).resolve().parents[1]


def run(args, settings, emb=None):
    out = io.StringIO()
    code = search_main(args, settings, emb or FakeEmbedder(), out)
    return code, out.getvalue()


def add(settings, tmp_path, name, text, subject):
    f = tmp_path / name
    f.write_text(text, encoding="utf-8")
    assert admin_main(["add-text", str(f), "--subject", subject], settings, FakeEmbedder(), io.StringIO()) == 0
    return f


def test_search_prints_rank_scores_file_page_and_image(settings, tmp_path):
    f = add(settings, tmp_path, "earth.txt", "The habitable zone is where liquid water can exist on a planet.", "Science")
    code, text = run(["habitable zone"], settings)
    assert code == 0
    assert text.startswith("1. [Science] earth - ")
    assert str(f.resolve()) in text and "page 1" in text
    assert "vec #1 cos" in text and "kw #1 bm25" in text and "image:" in text and "page quality 1.00" in text
    assert "habitable zone is where" in text


def test_subject_mode_and_k_options(settings, tmp_path):
    add(settings, tmp_path, "s.txt", "Water covers the planet.", "Science")
    add(settings, tmp_path, "h.txt", "Water mills powered towns.", "History")
    _, both = run(["water", "--k", "5"], settings)
    assert "[Science]" in both and "[History]" in both
    _, only = run(["water", "--subject", "history"], settings)
    assert "[History]" in only and "[Science]" not in only
    _, one = run(["water", "--k", "1"], settings)
    assert one.count("\n1. ") + one.startswith("1. ") == 1 and "\n2. " not in one
    _, kw = run(["water", "--mode", "keyword", "--subject", "science"], settings)
    assert "kw #1" in kw and "vec -" in kw
    _, vec = run(["water", "--mode", "vector", "--subject", "science"], settings)
    assert "vec #1" in vec and "kw -" in vec


def test_text_flag_prints_the_full_chunk(settings, tmp_path):
    body = "First line of notes.\n\nSecond paragraph about plates moving slowly."
    add(settings, tmp_path, "n.txt", body, "Science")
    _, snippet = run(["plates"], settings)
    _, full = run(["plates", "--text"], settings)
    assert "First line of notes." in full and "Second paragraph" in full
    assert len(full) >= len(snippet)


def test_no_results(settings):
    code, text = run(["anything"], settings)
    assert code == 0 and text.strip() == "No results."


def test_model_change_is_a_clean_failure_with_a_hint(settings, tmp_path):
    add(settings, tmp_path, "a.txt", "Plates move slowly.", "Science")
    code, text = run(["plates"], dataclasses.replace(settings, embed_model="other"))
    assert code == 1 and text.startswith("[FAIL]") and "reindex" in text


def test_search_uses_the_query_prefix(settings, tmp_path):
    add(settings, tmp_path, "a.txt", "Plates move slowly.", "Science")
    emb = FakeEmbedder(query_prefix="search_query: ")
    run(["plates"], settings, emb)
    assert emb.calls[-1] == ("queries", ["search_query: plates"])


def test_script_help_runs():
    out = subprocess.run([sys.executable, str(ROOT / "scripts" / "search.py"), "--help"],
                         capture_output=True, text=True, timeout=60)
    assert out.returncode == 0 and "--subject" in out.stdout and "--mode" in out.stdout
