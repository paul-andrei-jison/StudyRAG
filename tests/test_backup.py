import json
import sqlite3

import pytest

from conftest import FakeEmbedder
from studybuddy.backup import MANIFEST, PREFIX, BackupError, make_backup, verify_backup
from studybuddy.index.index import Index
from studybuddy.index.store import DocumentMeta
from studybuddy.reader.quality import PageRecord


@pytest.fixture
def world(settings, tmp_path):
    """A library with two photos, an index with one document, and the usual data-folder neighbours."""
    lib, data = settings.library_dir, settings.data_dir
    (lib / "Science").mkdir(parents=True)
    (lib / "Science" / "a.jpg").write_bytes(b"photo-a" * 100)
    (lib / "Science" / "b.pdf").write_bytes(b"%PDF-b")
    (lib / "Science" / "half.jpg.partial").write_bytes(b"incomplete")
    for folder, name in (("bot/items", "x.json"), ("queue", "job.json"), ("logs", "studybuddy.log"), ("samples", "s.jpg"), ("backup", "old.bak")):
        (data / folder).mkdir(parents=True, exist_ok=True)
        (data / folder / name).write_text("data", encoding="utf-8")
    (data / "embed_meta.json").write_text(json.dumps({"model": "embed-m", "dim": 3}), encoding="utf-8")
    index = Index.open(settings, FakeEmbedder())
    index.add_document(DocumentMeta(content_hash="h", file_path=str(lib / "Science" / "a.jpg"), source_kind="photo", subject="Science", title="Worksheet"),
                       [PageRecord(1, str(lib / "Science" / "a.jpg"), "image", text="Photosynthesis happens in leaves of plants.", quality=0.9)], expected_pages=1)
    index.close()
    return settings, tmp_path / "usb"


def names(folder):
    return {p.relative_to(folder).as_posix() for p in folder.rglob("*") if p.is_file()}


def test_a_backup_holds_the_library_and_the_index_but_not_logs_or_samples(world):
    settings, dest = world
    result = make_backup(settings, dest, repo_root=dest.parent / "repo")
    found = names(result.folder)
    assert result.folder.name.startswith(PREFIX) and not list(dest.glob("*.partial"))
    assert {"library/Science/a.jpg", "library/Science/b.pdf", "data/bot/items/x.json", "data/queue/job.json",
            "data/embed_meta.json", "data/index/studybuddy.db", MANIFEST, "RESTORE.txt"} <= found
    assert not any(n.endswith(".partial") or "/logs/" in n or "/samples/" in n or "/backup/" in n for n in found)
    assert not any(n.endswith(("-wal", "-shm")) for n in found)
    assert any(n.startswith("data/index/vectors/") for n in found)
    assert verify_backup(result.folder) == []


def test_the_copied_database_opens_and_has_the_document(world):
    settings, dest = world
    result = make_backup(settings, dest, repo_root=dest.parent / "repo")
    with sqlite3.connect(result.folder / "data" / "index" / "studybuddy.db") as db:
        assert db.execute("select count(*) from documents").fetchone()[0] == 1


def test_originals_are_untouched(world):
    settings, dest = world
    before = (settings.library_dir / "Science" / "a.jpg").read_bytes()
    make_backup(settings, dest, repo_root=dest.parent / "repo")
    assert (settings.library_dir / "Science" / "a.jpg").read_bytes() == before


def test_verify_finds_a_changed_a_missing_and_an_extra_file(world):
    settings, dest = world
    folder = make_backup(settings, dest, repo_root=dest.parent / "repo").folder
    (folder / "library" / "Science" / "a.jpg").write_bytes(b"tampered")
    (folder / "library" / "Science" / "b.pdf").unlink()
    (folder / "library" / "extra.txt").write_text("x")
    problems = verify_backup(folder)
    assert "changed: library/Science/a.jpg" in problems and "missing: library/Science/b.pdf" in problems
    assert "unexpected file: library/extra.txt" in problems


def test_verify_says_so_when_it_is_not_a_backup(tmp_path):
    assert "not a Study Buddy backup" in verify_backup(tmp_path)[0]


@pytest.mark.parametrize("where", ["library", "data", "repo"])
def test_a_destination_inside_the_library_data_or_repo_is_refused(world, where):
    settings, dest = world
    inside = {"library": settings.library_dir / "backups", "data": settings.data_dir / "out", "repo": dest.parent / "repo" / "out"}[where]
    with pytest.raises(BackupError, match="must not be inside or contain"):
        make_backup(settings, inside, repo_root=dest.parent / "repo")
    assert not inside.exists()


def test_a_backup_never_overwrites_an_earlier_one(world):
    settings, dest = world
    first = make_backup(settings, dest, now=lambda: 1_800_000_000, repo_root=dest.parent / "repo")
    with pytest.raises(BackupError, match="already exists"):
        make_backup(settings, dest, now=lambda: 1_800_000_000, repo_root=dest.parent / "repo")
    assert verify_backup(first.folder) == []


def test_a_failed_backup_leaves_no_half_folder(world, monkeypatch):
    settings, dest = world
    monkeypatch.setattr("studybuddy.backup.verify_backup", lambda folder: ["changed: library/x"])
    with pytest.raises(BackupError, match="did not match"):
        make_backup(settings, dest, repo_root=dest.parent / "repo")
    assert list(dest.iterdir()) == []


def test_a_missing_library_is_explained(settings, tmp_path):
    with pytest.raises(BackupError, match="does not exist"):
        make_backup(settings, tmp_path / "usb", repo_root=tmp_path / "repo")
