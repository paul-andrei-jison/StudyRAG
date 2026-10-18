"""Moving a filed document to another subject folder: files, index and every stored path together, or not at all."""
import json
import shutil
from pathlib import Path

import pytest

from ingest_helpers import make_env, make_photo, tree
from studybuddy.index.index import Index
from studybuddy.ingest.filing import FilingError
from studybuddy.ingest.relocate import Relocator, RelocationError
from studybuddy.ingest.results import INDEXED
from studybuddy.ui.files import LibraryGuard


@pytest.fixture
def env(tmp_path, settings):
    e = make_env(tmp_path, settings)
    yield e
    e.index.close()


def relocator(env, **kw):
    return Relocator(env.index, env.library, env.data / "relocations", **kw)


def file_single(env, name="a.jpg", subject="Math", **kw):
    result = env.processor.process(env.job(make_photo(env.inbox / name, **kw), subject=subject))
    assert result.status == INDEXED, result.error
    return result


def file_album(env, subject="Math"):
    a, b = make_photo(env.inbox / "t" / "1.jpg", color=(1, 2, 3)), make_photo(env.inbox / "t" / "2.jpg", color=(9, 9, 9))
    result = env.processor.process(env.job(a, b, subject=subject, album=True))
    assert result.status == INDEXED, result.error
    return result


def stored_paths(env, doc_id):
    store = env.index.store
    doc = store.get_document(doc_id)
    paths = {doc.file_path}
    paths |= {p["image_path"] for p in store.pages(doc_id)}
    paths |= {json.loads(p["record_json"])["source_path"] for p in store.pages(doc_id)}
    with store._lock:
        paths |= {r["image_path"] for r in store._db.execute("SELECT image_path FROM chunks WHERE doc_id = ?", (doc_id,))}
    return paths


def test_a_single_file_moves_with_its_subject_name_and_every_stored_path(env):
    result = file_single(env)
    old = Path(result.final_path)
    assert old.parent.name == "Math" and "_Math_" in old.name
    moved = relocator(env).relocate(result.doc_id, "Science")
    new = Path(moved.new_path)
    assert moved.moved and new.parent.name == "Science" and "_Science_" in new.name and new.name.startswith(old.name[:16])
    assert new.is_file()
    assert not old.exists() and not old.parent.exists()                      # the emptied old subject folder is removed
    assert stored_paths(env, result.doc_id) == {str(new)}                    # document, page, record and chunks all follow
    assert env.index.documents()[0].subject == "Science"
    assert env.index.search("photosynthesis chloroplasts glucose", k=3, subject="Science")
    assert not env.index.search("photosynthesis chloroplasts glucose", k=3, subject="Math")
    assert not list((env.data / "relocations").glob("*.json"))              # no journal is left behind


def test_the_moved_page_still_opens_through_the_library_guard(env):
    result = file_single(env, size=(300, 400))
    moved = relocator(env).relocate(result.doc_id, "Science")
    page = env.index.store.pages(result.doc_id)[0]
    assert LibraryGuard(env.library).render_page(page["image_path"], 1).size == (300, 400)
    assert page["image_path"] == moved.new_path


def test_the_copy_is_byte_identical(env):
    result = file_single(env)
    original = Path(result.final_path).read_bytes()
    moved = relocator(env).relocate(result.doc_id, "Science")
    assert Path(moved.new_path).read_bytes() == original


def test_the_old_folder_is_kept_when_other_documents_are_in_it(env):
    first = file_single(env, "a.jpg")
    other = file_single(env, "b.jpg", color=(7, 7, 7))
    relocator(env).relocate(first.doc_id, "Science")
    assert Path(other.final_path).is_file() and Path(other.final_path).parent.is_dir()


def test_moving_to_the_same_subject_changes_nothing(env):
    result = file_single(env)
    moved = relocator(env).relocate(result.doc_id, "math")
    assert not moved.moved and Path(result.final_path).is_file()


def test_a_name_collision_gets_a_numbered_name(env):
    first = file_single(env, "a.jpg", subject="Math")
    second = file_single(env, "b.jpg", subject="Science", color=(7, 7, 7))
    clash = Path(first.final_path).parent.parent / "Science" / Path(first.final_path).name.replace("_Math_", "_Science_")
    clash.parent.mkdir(exist_ok=True)
    clash.write_bytes(b"someone else's file")
    moved = relocator(env).relocate(first.doc_id, "Science")
    assert Path(moved.new_path).name.endswith("_2.jpg") and clash.read_bytes() == b"someone else's file"
    assert Path(second.final_path).is_file()


def test_an_album_folder_moves_with_its_pages_and_pdf(env):
    result = file_album(env)
    old = Path(result.final_path)
    assert old.is_dir() and {p.name for p in old.iterdir()} >= {"page-01.jpg", "page-02.jpg"}
    moved = relocator(env).relocate(result.doc_id, "Science")
    new = Path(moved.new_path)
    assert new.is_dir() and new.parent.name == "Science" and not old.exists()
    assert {p.name for p in new.iterdir()} == {"page-01.jpg", "page-02.jpg", f"{new.name}.pdf"}      # the PDF follows the folder's new name
    assert stored_paths(env, result.doc_id) == {str(new), str(new / "page-01.jpg"), str(new / "page-02.jpg")}
    assert [Path(p["image_path"]).name for p in env.index.store.pages(result.doc_id)] == ["page-01.jpg", "page-02.jpg"]


def test_a_copy_that_does_not_match_the_source_moves_nothing(env):
    result = file_single(env)
    before = tree(env.library)

    def bad_copy(src, dst):
        Path(dst).write_bytes(b"garbage")

    with pytest.raises(RelocationError, match="nothing was changed"):
        relocator(env, copy_fn=bad_copy).relocate(result.doc_id, "Science")
    assert tree(env.library) == before and env.index.documents()[0].subject == "Math"
    assert stored_paths(env, result.doc_id) == {result.final_path}
    assert not list((env.data / "relocations").glob("*.json"))


def test_a_database_failure_puts_everything_back(env, monkeypatch):
    result = file_single(env)
    before = tree(env.library)

    def boom(doc_id, subject, mapping):
        raise RuntimeError("disk full")

    monkeypatch.setattr(env.index.store, "rewrite_paths", boom)
    with pytest.raises(RelocationError, match="disk full"):
        relocator(env).relocate(result.doc_id, "Science")
    assert tree(env.library) == before and env.index.documents()[0].subject == "Math"


def test_a_vector_failure_restores_the_database_and_the_files(env, monkeypatch):
    result = file_single(env)
    before = tree(env.library)

    def boom(doc_id, key):
        raise RuntimeError("vector store unavailable")

    monkeypatch.setattr(env.index.vectors, "update_subject", boom)
    with pytest.raises(RelocationError, match="vector store unavailable"):
        relocator(env).relocate(result.doc_id, "Science")
    assert tree(env.library) == before
    assert env.index.documents()[0].subject == "Math" and stored_paths(env, result.doc_id) == {result.final_path}
    assert env.index.search("photosynthesis chloroplasts glucose", k=3, subject="Math")


def test_a_missing_file_is_refused_without_changing_anything(env):
    result = file_single(env)
    Path(result.final_path).unlink()
    with pytest.raises(RelocationError, match="missing from the library"):
        relocator(env).relocate(result.doc_id, "Science")
    assert env.index.documents()[0].subject == "Math"


def test_a_file_outside_the_library_is_refused(env, tmp_path):
    outside = make_photo(tmp_path / "elsewhere" / "x.jpg")
    from chat_helpers import add_doc
    doc_id = add_doc(env.index, "outside", "Some text about plates.", subject="Math", path=str(outside)).doc_id
    with pytest.raises(RelocationError, match="outside the library"):
        relocator(env).relocate(doc_id, "Science")
    assert outside.is_file()


def test_unknown_document_is_explained(env):
    with pytest.raises(RelocationError, match="no longer exists"):
        relocator(env).relocate(404, "Science")


def test_a_subject_with_characters_windows_refuses_gets_a_safe_folder(env):
    result = file_single(env)
    moved = relocator(env).relocate(result.doc_id, "Art: A/B?")
    assert Path(moved.new_path).is_file() and ":" not in Path(moved.new_path).parent.name and "?" not in moved.new_path


def test_a_hand_renamed_file_keeps_its_name_and_only_changes_folder(env):
    result = file_single(env)
    old = Path(result.final_path)
    renamed = old.with_name("my-favourite-worksheet.jpg")
    old.rename(renamed)
    env.index.update_document(result.doc_id, file_path=str(renamed))
    with env.index.store.transaction() as db:
        db.execute("UPDATE pages SET image_path = ? WHERE doc_id = ?", (str(renamed), result.doc_id))
    moved = relocator(env).relocate(result.doc_id, "Science")
    assert Path(moved.new_path).name == "my-favourite-worksheet.jpg" and Path(moved.new_path).parent.name == "Science"


def test_a_failed_removal_of_the_old_file_is_only_a_warning(env, monkeypatch):
    result = file_single(env)
    real_unlink = Path.unlink

    def stubborn(self, *a, **k):
        if self == Path(result.final_path):
            raise PermissionError("in use")
        return real_unlink(self, *a, **k)

    monkeypatch.setattr(Path, "unlink", stubborn)
    moved = relocator(env).relocate(result.doc_id, "Science")
    assert moved.moved and any("could not be removed" in w for w in moved.warnings)
    assert Path(moved.new_path).is_file() and env.index.documents()[0].subject == "Science"


# ---- a crash in the middle ------------------------------------------------------------------------------

def crash_midway(env, monkeypatch, result, after="copy"):
    """Run a move that 'crashes' (the process dies: no undo runs) at the given point; leaves the journal behind."""
    class Crash(Exception):
        pass

    monkeypatch.setattr("studybuddy.ingest.relocate.Filer.rollback", staticmethod(lambda created: None))   # a dead process undoes nothing
    monkeypatch.setattr(Relocator, "_drop_journal", lambda self, doc_id: None)
    if after == "copy":
        monkeypatch.setattr(env.index, "relocate_document", lambda *a: (_ for _ in ()).throw(Crash()))
    else:
        real = env.index.relocate_document
        monkeypatch.setattr(env.index, "relocate_document", lambda *a: real(*a))
        monkeypatch.setattr(Relocator, "_remove_old", lambda *a, **k: (_ for _ in ()).throw(Crash()))
    with pytest.raises(Exception):
        relocator(env).relocate(result.doc_id, "Science")


def test_recover_undoes_a_move_that_crashed_before_the_database_changed(env, monkeypatch):
    result = file_single(env)
    before_files = tree(env.library)
    crash_midway(env, monkeypatch, result, after="copy")
    monkeypatch.undo()
    assert tree(env.library) != before_files and list((env.data / "relocations").glob("*.json"))     # the half-made copy is there
    notes = relocator(env).recover()
    assert notes and "undone" in notes[0]
    assert tree(env.library) == before_files and env.index.documents()[0].subject == "Math"
    assert not list((env.data / "relocations").glob("*.json"))


def test_recover_finishes_a_move_that_crashed_after_the_database_changed(env, monkeypatch):
    result = file_single(env)
    crash_midway(env, monkeypatch, result, after="clean-up")
    monkeypatch.undo()
    doc = env.index.documents()[0]
    assert doc.subject == "Science" and Path(result.final_path).exists() and Path(doc.file_path).exists()     # both copies exist
    notes = relocator(env).recover()
    assert notes and "finished" in notes[0]
    assert not Path(result.final_path).exists() and Path(doc.file_path).is_file()
    assert not list((env.data / "relocations").glob("*.json"))


def test_recover_with_no_journals_does_nothing(env):
    assert relocator(env).recover() == []
