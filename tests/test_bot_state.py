import threading
from datetime import datetime

import pytest

from studybuddy.bot.spool import Spool
from studybuddy.bot.state import ASKING, CLOSED, DONE, Item, Question, StateStore, new_id
from studybuddy.ingest.subjects import SPOOL_FOLDER

JPEG = b"\xff\xd8\xff\xe0" + b"\x00" * 40
PDF = b"%PDF-1.4\n" + b"x" * 40


@pytest.fixture
def spool(tmp_path):
    return Spool(tmp_path / "library", clock=lambda: datetime(2026, 10, 18, 9, 5, 7))


def test_a_file_is_written_into_the_unsorted_folder_with_its_real_type_and_hash(spool, tmp_path):
    saved = spool.write(JPEG, "My Worksheet (final).JPG")
    assert saved.path.parent == tmp_path / "library" / SPOOL_FOLDER and saved.path.read_bytes() == JPEG
    assert saved.path.name == "2026-10-18_090507_My Worksheet (final).jpg" and saved.kind == "jpeg" and saved.size == len(JPEG)
    import hashlib
    assert saved.sha256 == hashlib.sha256(JPEG).hexdigest()
    assert list(saved.path.parent.glob(".*.partial")) == []


def test_the_extension_follows_the_bytes_not_the_name(spool):
    assert spool.write(PDF, "scan.jpg").path.suffix == ".pdf"
    odd = spool.write(b"not an image at all", "notes.docx")
    assert odd.path.suffix == ".bin" and odd.kind is None and odd.path.read_bytes() == b"not an image at all"   # kept, ingest will explain


def test_same_second_same_name_never_overwrites(spool):
    first, second = spool.write(JPEG, "a.jpg"), spool.write(JPEG + b"1", "a.jpg")
    assert first.path != second.path and first.path.read_bytes() == JPEG and second.path.name.endswith("_2.jpg")


def test_unsafe_names_are_made_safe_for_windows(spool):
    saved = spool.write(JPEG, 'CON: <bad>|name?.jpg')
    assert not any(c in saved.path.name for c in '<>:"|?*')
    assert spool.write(JPEG, None).path.name.startswith("2026-10-18_090507_telegram")


def test_a_failed_write_leaves_nothing_behind(spool, monkeypatch):
    import os
    monkeypatch.setattr(os, "fsync", lambda fd: (_ for _ in ()).throw(OSError("disk full")))
    with pytest.raises(OSError):
        spool.write(JPEG, "a.jpg")
    assert list(spool.dir.iterdir()) == []


def test_start_up_removes_half_written_leftovers_only(spool):
    kept = spool.write(JPEG, "kept.jpg").path
    (spool.dir / ".2026-10-18_x.jpg.partial").write_bytes(b"half")
    (spool.dir / "2026-10-18_claimed.jpg").write_bytes(b"")
    assert spool.sweep_partials() == 2 and [p.name for p in spool.dir.iterdir()] == [kept.name]
    assert Spool(spool.dir.parent / "nowhere").sweep_partials() == 0


# ---- the pending store -------------------------------------------------------------------------------

def item(**over):
    base = dict(id=new_id(), chat_id=42, message_id=7, kind="photo", file_id="F1", file_unique_id="U1", created_at=1.0)
    base.update(over)
    return Item(**base)


@pytest.fixture
def store(tmp_path):
    return StateStore(tmp_path / "bot")


def test_items_and_questions_survive_a_restart(store, tmp_path):
    first = item(caption="Science", spool_path="C:/x.jpg", media_group_id="G", stage=ASKING, question_id="q1")
    store.save_item(first)
    store.save_question(Question("q1", 42, [first.id], ["Math", "Science"], message_id=99, deadline=500.0, album=True, created_at=2.0))
    reopened = StateStore(tmp_path / "bot")
    assert reopened.item(first.id) == first
    q = reopened.question("q1")
    assert q.item_ids == [first.id] and q.subjects == ["Math", "Science"] and q.deadline == 500.0 and q.album and q.message_id == 99
    assert reopened.item("missing") is None and reopened.question("missing") is None


def test_listing_is_oldest_first_and_find_filters(store):
    late, early = item(created_at=5.0, kind="pdf"), item(created_at=1.0)
    store.save_item(late), store.save_item(early)
    assert [i.id for i in store.all_items()] == [early.id, late.id]
    assert [i.id for i in store.find_item(lambda i: i.kind == "pdf")] == [late.id]


def test_a_corrupt_file_is_set_aside_not_fatal(store, tmp_path):
    good = item()
    store.save_item(good)
    (tmp_path / "bot" / "items" / "broken.json").write_text("{not json", encoding="utf-8")
    assert [i.id for i in store.all_items()] == [good.id]
    assert (tmp_path / "bot" / "items" / "corrupt" / "broken.json").exists()


def test_unknown_fields_from_a_newer_version_are_ignored(store, tmp_path):
    good = item()
    store.save_item(good)
    path = tmp_path / "bot" / "items" / f"{good.id}.json"
    path.write_text(path.read_text(encoding="utf-8").replace('"stage"', '"future_field": 1, "stage"'), encoding="utf-8")
    assert store.item(good.id).id == good.id


def test_a_redelivered_update_is_recognised(store):
    store.save_item(item(chat_id=42, message_id=7, file_unique_id="U1"))
    assert store.already_seen(42, 7, "U1")
    assert not store.already_seen(42, 8, "U1") and not store.already_seen(42, 7, "U2") and not store.already_seen(43, 7, "U1")


def test_finished_items_are_forgotten_only_after_the_keep_time(store):
    done, open_one = item(stage=DONE, finished_at=100.0), item(stage=ASKING)
    store.save_item(done), store.save_item(open_one)
    store.save_question(Question("old", 42, [done.id], stage=CLOSED, created_at=100.0))
    store.save_question(Question("live", 42, [open_one.id], created_at=100.0))
    assert store.sweep_done(now=100.0 + 3600) == 0
    assert store.sweep_done(now=100.0 + 49 * 3600) == 1
    assert store.item(done.id) is None and store.item(open_one.id) and store.question("old") is None and store.question("live")


def test_concurrent_saves_do_not_corrupt(store):
    def work(n):
        for _ in range(20):
            store.save_item(item(id=f"i{n}", message_id=n))
    threads = [threading.Thread(target=work, args=(n,)) for n in range(6)]
    [t.start() for t in threads]
    [t.join() for t in threads]
    assert sorted(i.id for i in store.all_items()) == [f"i{n}" for n in range(6)]
