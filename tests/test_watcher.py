import json
import time
from datetime import datetime
from pathlib import Path

import pytest

from ingest_helpers import make_photo
from studybuddy.errors import StudyBuddyError
from studybuddy.ingest.subjects import SubjectError
from studybuddy.ingest.watcher import InboxScanner, InboxWatcher, Group, is_ignored, natural_key
from test_pdf import make_scanned_pdf


class Clock:
    def __init__(self, now=1_000_000.0):
        self.now = now

    def __call__(self):
        return self.now


@pytest.fixture
def inbox(tmp_path):
    d = tmp_path / "inbox"
    d.mkdir()
    return d


@pytest.fixture
def clock():
    return Clock()


@pytest.fixture
def scanner(inbox, clock):
    return InboxScanner(inbox, clock=clock, settle_seconds=3.0)


def settle(scanner, clock, queued=None):
    """First look (starts the quiet period), then wait it out, then look again."""
    scanner.scan(queued)
    clock.now += 3.5
    return scanner.scan(queued)


# ---- readiness -------------------------------------------------------------------------------

def test_a_file_is_ready_only_after_it_stops_changing(scanner, inbox, clock):
    photo = make_photo(inbox / "a.jpg")
    assert scanner.scan() == [] and scanner.unsettled == 1  # first sighting
    clock.now += 2
    assert scanner.scan() == []                              # quiet for 2 s of 3
    clock.now += 1.5
    (group,) = scanner.scan()
    assert group.files == [photo] and group.subject is None and not group.is_album and group.kinds == ["photo-file"]
    assert scanner.unsettled == 0


def test_a_growing_file_restarts_the_quiet_period(scanner, inbox, clock):
    path = inbox / "big.jpg"
    make_photo(path)
    scanner.scan()
    clock.now += 2.9
    with path.open("ab") as f:
        f.write(b"more bytes arrive")  # the copy is still going
    assert scanner.scan() == []
    clock.now += 2.9
    assert scanner.scan() == []  # only 2.9 s since the last change
    clock.now += 0.5
    assert len(scanner.scan()) == 1


def test_a_file_another_program_still_has_locked_is_not_ready(scanner, inbox, clock, monkeypatch):
    path = make_photo(inbox / "a.jpg")
    scanner.scan()
    clock.now += 4
    real_open = Path.open

    def locked(self, *args, **kwargs):
        if self.name == "a.jpg":
            raise PermissionError("being written by another process")
        return real_open(self, *args, **kwargs)

    monkeypatch.setattr(Path, "open", locked)
    assert scanner.scan() == []
    monkeypatch.undo()
    assert len(scanner.scan()) == 1 and path.exists()


@pytest.mark.parametrize("name", ["~$notes.docx", ".hidden.jpg", "Thumbs.db", "desktop.ini", ".DS_Store", "x.tmp",
                                  "photo.jpg.crdownload", "scan.partial", "x.part", "shortcut.lnk"])
def test_temporary_and_system_files_are_ignored(scanner, inbox, clock, name):
    (inbox / name).write_bytes(b"whatever")
    assert settle(scanner, clock) == []
    assert is_ignored(inbox / name)


def test_a_file_that_leaves_the_inbox_is_forgotten(scanner, inbox, clock):
    path = make_photo(inbox / "a.jpg")
    scanner.scan()
    path.unlink()
    clock.now += 4
    assert scanner.scan() == [] and scanner._files == {}
    make_photo(inbox / "a.jpg")  # a new file with the same name starts from scratch
    assert scanner.scan() == []


def test_missing_inbox_is_not_an_error(tmp_path, clock):
    assert InboxScanner(tmp_path / "nope", clock=clock).scan() == []


# ---- layout ----------------------------------------------------------------------------------

def test_a_file_in_a_subject_folder_carries_the_subject(scanner, inbox, clock):
    photo = make_photo(inbox / "Science" / "a.jpg")
    (group,) = settle(scanner, clock)
    assert group.files == [photo] and group.subject == "Science" and not group.is_album


def test_several_loose_files_in_a_subject_folder_are_separate_documents(scanner, inbox, clock):
    for name in ("a.jpg", "b.jpg", "c.jpg"):
        make_photo(inbox / "Math" / name, color=(ord(name[0]), 9, 9))
    groups = settle(scanner, clock)
    assert sorted(g.files[0].name for g in groups) == ["a.jpg", "b.jpg", "c.jpg"]
    assert all(g.subject == "Math" and not g.is_album for g in groups)


def test_a_subfolder_with_several_photos_is_one_album_in_natural_order(scanner, inbox, clock):
    names = ["img10.jpg", "img2.jpg", "img1.jpg"]
    for i, name in enumerate(names):
        make_photo(inbox / "Science" / "field trip" / name, color=(i * 50, 9, 9))
    (group,) = settle(scanner, clock)
    assert group.is_album and group.subject == "Science"
    assert [p.name for p in group.files] == ["img1.jpg", "img2.jpg", "img10.jpg"]
    assert group.kinds == ["photo-file"] * 3


def test_an_album_waits_until_no_more_photos_arrive(scanner, inbox, clock):
    folder = inbox / "Science" / "trip"
    make_photo(folder / "1.jpg", color=(1, 1, 1))
    make_photo(folder / "2.jpg", color=(2, 2, 2))
    scanner.scan()
    clock.now += 3.5
    make_photo(folder / "3.jpg", color=(3, 3, 3))  # the student is still copying: the third photo shows up late
    assert scanner.scan() == []
    clock.now += 3.5
    (group,) = scanner.scan()
    assert [p.name for p in group.files] == ["1.jpg", "2.jpg", "3.jpg"]


def test_a_subfolder_with_one_photo_is_a_single_document(scanner, inbox, clock):
    make_photo(inbox / "Science" / "solo" / "a.jpg")
    (group,) = settle(scanner, clock)
    assert not group.is_album and group.subject == "Science" and len(group.files) == 1


def test_pdfs_in_an_album_folder_are_separate_documents(scanner, inbox, clock):
    make_photo(inbox / "Science" / "mix" / "1.jpg", color=(1, 1, 1))
    make_photo(inbox / "Science" / "mix" / "2.jpg", color=(2, 2, 2))
    make_scanned_pdf(inbox / "Science" / "mix" / "doc.pdf", 1)
    groups = settle(scanner, clock)
    album = [g for g in groups if g.is_album]
    pdf = [g for g in groups if not g.is_album]
    assert len(album) == 1 and len(album[0].files) == 2
    assert len(pdf) == 1 and pdf[0].kinds == ["pdf"]


def test_deeper_nesting_belongs_to_the_album(scanner, inbox, clock):
    make_photo(inbox / "Math" / "set" / "part1" / "a.jpg", color=(1, 1, 1))
    make_photo(inbox / "Math" / "set" / "part2" / "b.jpg", color=(2, 2, 2))
    (group,) = settle(scanner, clock)
    assert group.is_album and {p.name for p in group.files} == {"a.jpg", "b.jpg"}


def test_kinds_are_detected_from_content(scanner, inbox, clock):
    make_scanned_pdf(inbox / "x.bin", 1)           # a PDF with a misleading name
    (inbox / "notes.docx").write_bytes(b"PK\x03\x04 not an image")
    make_photo(inbox / "p.jpg")
    kinds = {g.files[0].name: g.kinds for g in settle(scanner, clock)}
    assert kinds == {"x.bin": ["pdf"], "notes.docx": ["document"], "p.jpg": ["photo-file"]}  # docx still goes on: the processor files it under failed/


def test_empty_folders_do_nothing(scanner, inbox, clock):
    (inbox / "Science" / "empty album").mkdir(parents=True)
    assert settle(scanner, clock) == []


def test_already_queued_files_are_not_queued_again(scanner, inbox, clock):
    photo = make_photo(inbox / "a.jpg")
    other = make_photo(inbox / "b.jpg", color=(3, 3, 3))
    groups = settle(scanner, clock, queued={str(photo.resolve())})
    assert [g.files[0].name for g in groups] == ["b.jpg"] and other.exists()


def test_a_queued_album_is_not_queued_again(scanner, inbox, clock):
    a = make_photo(inbox / "Math" / "set" / "1.jpg", color=(1, 1, 1))
    make_photo(inbox / "Math" / "set" / "2.jpg", color=(2, 2, 2))
    assert settle(scanner, clock, queued={str(a.resolve())}) == []


def test_arrival_time_is_when_it_was_first_seen_not_the_old_mtime(scanner, inbox, clock):
    photo = make_photo(inbox / "a.jpg")
    old = clock.now - 30 * 86400
    import os
    os.utime(photo, (old, old))  # copied from a phone: the file claims to be a month old
    scanner.scan()
    first_seen = datetime.fromtimestamp(clock.now)
    clock.now += 10
    (group,) = scanner.scan()
    assert group.received_at == first_seen


def test_natural_key_orders_numbers_numerically():
    names = [Path(n) for n in ("p10.jpg", "p2.jpg", "p1.jpg", "P3.jpg")]
    assert [p.name for p in sorted(names, key=natural_key)] == ["p1.jpg", "p2.jpg", "P3.jpg", "p10.jpg"]


# ---- watcher ---------------------------------------------------------------------------------

class FakeService:
    def __init__(self, tmp_path, reject=None):
        self.submitted = []
        self.reject = reject

        class S:
            data_dir = tmp_path / "data"

        class J:
            @staticmethod
            def pending_paths():
                return set()

        self.settings, self.jobs, self.inbox_dir = S, J, tmp_path / "data" / "inbox"

    def submit(self, paths, subject=None, source_kind=None, received_at=None, origin="api", is_album=False, **kw):
        if self.reject:
            raise self.reject
        self.submitted.append(dict(paths=list(paths), subject=subject, kinds=source_kind, received_at=received_at,
                                   origin=origin, is_album=is_album))
        return object()


def test_scan_once_submits_ready_groups_with_the_right_arguments(tmp_path, clock):
    service = FakeService(tmp_path)
    inbox = service.inbox_dir
    photo = make_photo(inbox / "Science" / "a.jpg")
    watcher = InboxWatcher(service, scanner=InboxScanner(inbox, clock=clock, settle_seconds=3.0), use_watchdog=False)
    assert watcher.scan_once() == []
    clock.now += 4
    assert len(watcher.scan_once()) == 1
    (call,) = service.submitted
    assert call["paths"] == [photo] and call["subject"] == "Science" and call["origin"] == "inbox"
    assert call["kinds"] == ["photo-file"] and not call["is_album"] and isinstance(call["received_at"], datetime)


def test_an_unusable_subject_folder_is_moved_to_failed_with_a_reason(tmp_path, clock):
    service = FakeService(tmp_path, reject=SubjectError("'CON' is reserved and cannot be used as a subject."))
    inbox = service.inbox_dir
    original = make_photo(inbox / "CON" / "a.jpg")
    data = original.read_bytes()
    watcher = InboxWatcher(service, scanner=InboxScanner(inbox, clock=clock, settle_seconds=3.0), use_watchdog=False)
    watcher.scan_once()
    clock.now += 4
    assert watcher.scan_once() == []
    assert not original.exists()
    (failed,) = list((tmp_path / "data" / "failed").iterdir())
    info = json.loads((failed / "reason.json").read_text())
    assert "reserved" in info["error"] and "rename the folder" in info["error"] and info["kind"] == "permanent"
    assert (failed / "a.jpg").read_bytes() == data and info["files"] == [str(failed / "a.jpg")]


def test_other_queueing_errors_leave_the_file_for_the_next_scan(tmp_path, clock):
    service = FakeService(tmp_path, reject=StudyBuddyError("The ingest service is stopping."))
    inbox = service.inbox_dir
    photo = make_photo(inbox / "a.jpg")
    watcher = InboxWatcher(service, scanner=InboxScanner(inbox, clock=clock, settle_seconds=3.0), use_watchdog=False)
    watcher.scan_once()
    clock.now += 4
    assert watcher.scan_once() == [] and photo.exists()


def test_on_ticket_callback_sees_each_group(tmp_path, clock):
    service = FakeService(tmp_path)
    seen = []
    inbox = service.inbox_dir
    make_photo(inbox / "a.jpg")
    watcher = InboxWatcher(service, scanner=InboxScanner(inbox, clock=clock, settle_seconds=3.0), use_watchdog=False,
                           on_ticket=lambda ticket, group: seen.append(group.files[0].name))
    watcher.scan_once()
    clock.now += 4
    watcher.scan_once()
    assert seen == ["a.jpg"]


def test_watchdog_wakes_the_watcher_without_waiting_for_the_rescan(tmp_path):
    """Real observer, real files, real clock: the periodic rescan is set far too long to explain the result."""
    service = FakeService(tmp_path)
    inbox = service.inbox_dir
    inbox.mkdir(parents=True)
    scanner = InboxScanner(inbox, settle_seconds=0.3)
    watcher = InboxWatcher(service, scanner=scanner, rescan_seconds=60.0, use_watchdog=True).start()
    try:
        time.sleep(0.5)  # let the observer start
        make_photo(inbox / "Science" / "late.jpg")
        deadline = time.time() + 15
        while time.time() < deadline and not service.submitted:
            time.sleep(0.05)
    finally:
        watcher.stop()
    assert len(service.submitted) == 1 and service.submitted[0]["subject"] == "Science"


def test_polling_alone_still_works_when_watching_is_disabled(tmp_path):
    service = FakeService(tmp_path)
    inbox = service.inbox_dir
    make_photo(inbox / "a.jpg")
    scanner = InboxScanner(inbox, settle_seconds=0.2)
    watcher = InboxWatcher(service, scanner=scanner, rescan_seconds=0.1, use_watchdog=False).start()
    try:
        deadline = time.time() + 15
        while time.time() < deadline and not service.submitted:
            time.sleep(0.05)
    finally:
        watcher.stop()
    assert len(service.submitted) == 1


def test_watcher_creates_the_inbox_and_stops_cleanly(tmp_path):
    service = FakeService(tmp_path)
    watcher = InboxWatcher(service, rescan_seconds=0.1, use_watchdog=True).start()
    assert service.inbox_dir.is_dir()
    watcher.stop()
    assert not watcher._thread.is_alive()
