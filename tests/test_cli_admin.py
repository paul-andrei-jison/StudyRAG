import dataclasses
import io
import subprocess
import sys
from pathlib import Path

import pytest

from conftest import FakeEmbedder
from studybuddy.embedding_guard import read_meta
from studybuddy.index.cli import admin_main
from studybuddy.index.index import Index
from studybuddy.index.store import DocumentMeta
from studybuddy.reader.quality import PageRecord

ROOT = Path(__file__).resolve().parents[1]
TEXT = "Photosynthesis converts light energy into chemical energy inside chloroplasts."


def run(args, settings, emb=None):
    out = io.StringIO()
    code = admin_main(args, settings, emb or FakeEmbedder(), out)
    return code, out.getvalue()


def seed(settings, n=2):
    idx = Index.open(settings, FakeEmbedder())
    for i in range(n):
        rec = PageRecord(1, f"C:/lib/{i}.jpg", "image", text=f"{TEXT} Variant {i}.", quality=0.9)
        idx.add_document(DocumentMeta(f"h{i}", f"C:/lib/{i}.jpg", "photo", "Science", f"Doc {i}"), [rec])
    idx.close()


def test_status_on_a_fresh_index(settings):
    code, text = run(["status"], settings)
    assert code == 0 and "Documents 0, pages 0, chunks 0, vectors 0" in text and "Consistent" in text


def test_status_reports_counts_and_models(settings):
    seed(settings)
    code, text = run(["status"], settings)
    assert code == 0 and "Documents 2, pages 2, chunks 2, vectors 2 (dimension 64)" in text
    assert "recorded: 'embed-m'  configured: 'embed-m'" in text


def test_status_works_and_exits_nonzero_when_the_model_changed(settings):
    seed(settings)
    changed = dataclasses.replace(settings, embed_model="new-model")
    code, text = run(["status"], changed)  # must not refuse: status is how you diagnose the mismatch
    assert code == 1 and "[PROBLEM]" in text and "reindex" in text


def test_check_and_repair(settings):
    seed(settings)
    idx = Index.open(settings, FakeEmbedder())
    victim = sorted(idx.store.chunk_ids())[0]
    idx.vectors.delete([victim])
    idx.close()
    code, text = run(["check"], settings)
    assert code == 1 and "1 chunk(s) without a vector" in text and "repair" in text
    code, text = run(["repair"], settings)
    assert code == 0 and "Fixed 1 chunk(s)" in text
    assert run(["check"], settings)[0] == 0


def test_reindex_after_a_model_change(settings):
    seed(settings, n=3)
    new_settings = dataclasses.replace(settings, embed_model="bigger", embed_doc_prefix="doc: ")
    code, text = run(["reindex"], new_settings, FakeEmbedder(dim=32, doc_prefix="doc: "))
    assert code == 0 and "embedded 3/3 chunks" in text and "Re-indexed 3 chunks with 'bigger'" in text
    assert read_meta(settings.data_dir) == {"model": "bigger", "dim": 32, "doc_prefix": "doc: "}
    assert run(["status"], new_settings, FakeEmbedder(dim=32))[0] == 0


def test_reindex_failure_is_reported_and_leaves_the_index_alone(settings):
    seed(settings)
    new_settings = dataclasses.replace(settings, embed_model="bigger")
    code, text = run(["reindex"], new_settings, FakeEmbedder(dim=32, fail_after_calls=0))
    assert code == 1 and "[FAIL]" in text and "went away" in text
    assert read_meta(settings.data_dir)["model"] == "embed-m"
    assert run(["status"], settings)[0] == 0


def test_reset_needs_yes_and_never_touches_the_library(settings):
    seed(settings)
    settings.library_dir.mkdir(parents=True, exist_ok=True)
    photo = settings.library_dir / "keep.jpg"
    photo.write_bytes(b"original")
    code, text = run(["reset"], settings)
    assert code == 2 and "Would delete" in text and "--yes" in text and "never touched" in text
    assert (settings.data_dir / "index").exists()
    code, text = run(["reset", "--yes"], settings)
    assert code == 0 and "Deleted" in text
    assert not (settings.data_dir / "index").exists() and photo.read_bytes() == b"original"
    assert "Documents 0" in run(["status"], settings)[1]


def test_reset_with_nothing_to_delete(settings):
    code, text = run(["reset"], settings)
    assert code == 2 and "nothing to delete" in text


def test_add_text_file_and_folder(settings, tmp_path):
    notes = tmp_path / "notes"
    notes.mkdir()
    (notes / "a.txt").write_text(TEXT, encoding="utf-8")
    (notes / "b.md").write_text("Rivers carry water to the sea.", encoding="utf-8")
    (notes / "skip.pdf").write_bytes(b"%PDF")
    code, text = run(["add-text", str(notes), "--subject", "Science"], settings)
    assert code == 0 and "a.txt: indexed, 1 chunk(s)" in text and "b.md: indexed" in text and "skip.pdf" not in text
    code, text = run(["add-text", str(notes / "a.txt"), "--subject", "Science"], settings)
    assert "already indexed, skipped" in text
    idx = Index.open(settings, FakeEmbedder())
    docs = {d.title: d for d in idx.documents()}
    assert set(docs) == {"a", "b"} and docs["a"].source_kind == "text-file" and docs["a"].page_count == 1
    assert idx.search("photosynthesis")[0].title == "a"
    idx.close()


def test_add_text_single_file_title_and_errors(settings, tmp_path):
    f = tmp_path / "x.txt"
    f.write_text("Plates move slowly.", encoding="utf-8")
    run(["add-text", str(f), "--subject", "Science", "--title", "Plate notes"], settings)
    idx = Index.open(settings, FakeEmbedder())
    assert idx.documents()[0].title == "Plate notes"
    idx.close()
    bad = tmp_path / "bad.txt"
    bad.write_bytes(b"\xff\xfe\x00bad")
    code, text = run(["add-text", str(bad), "--subject", "Science"], settings)
    assert code == 1 and "[FAIL] bad.txt" in text
    code, text = run(["add-text", str(tmp_path / "nope.txt"), "--subject", "S"], settings)
    assert code == 1 and "no .txt/.md files" in text
    empty = tmp_path / "empty"
    empty.mkdir()
    assert run(["add-text", str(empty), "--subject", "S"], settings)[0] == 1


def test_add_text_is_refused_when_the_model_changed(settings, tmp_path):
    seed(settings)
    f = tmp_path / "x.txt"
    f.write_text("Plates move.", encoding="utf-8")
    code, text = run(["add-text", str(f), "--subject", "S"], dataclasses.replace(settings, embed_model="other"))
    assert code == 1 and "EMBEDDING MODEL CHANGED" in text and "reindex" in text


def test_missing_settings_is_a_clean_failure(tmp_path, monkeypatch):
    for name in ("SB_READER_MODEL", "SB_CHAT_MODEL", "SB_EMBED_MODEL", "SB_LIBRARY_DIR", "SB_DATA_DIR"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr("studybuddy.config.load_dotenv", lambda *a, **k: None)
    out = io.StringIO()
    assert admin_main(["status"], None, FakeEmbedder(), out) == 1 and "Missing required settings" in out.getvalue()


def test_script_help_runs():
    out = subprocess.run([sys.executable, str(ROOT / "scripts" / "index_admin.py"), "--help"],
                         capture_output=True, text=True, timeout=60)
    assert out.returncode == 0 and "reindex" in out.stdout and "reset" in out.stdout
