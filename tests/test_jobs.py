import json
import threading
from datetime import datetime

import pytest

from studybuddy.ingest.jobs import BACKOFF_SECONDS, Job, JobStore, backoff_seconds


class Clock:
    def __init__(self, now=1000.0):
        self.now = now

    def __call__(self):
        return self.now


@pytest.fixture
def clock():
    return Clock()


@pytest.fixture
def store(tmp_path, clock):
    return JobStore(tmp_path / "queue", clock=clock)


def make(store, name="a.jpg", **kw):
    return store.create([f"C:/inbox/{name}"], ["photo"], datetime(2026, 10, 17, 14, 25), **kw)


def test_backoff_schedule():
    assert [backoff_seconds(n) for n in range(1, 8)] == [30, 60, 120, 300, 900, 900, 900]
    assert backoff_seconds(0) == 30 and BACKOFF_SECONDS[0] == 30


def test_create_writes_a_json_file_before_anything_else(store):
    job = make(store, subject="Science", origin="bot")
    path = store.dir / f"{job.id}.json"
    assert path.exists()
    data = json.loads(path.read_text())
    assert data["subject"] == "Science" and data["origin"] == "bot" and data["state"] == "pending"
    assert data["received_at"] == "2026-10-17T14:25:00" and data["attempts"] == 0
    assert [p.name for p in store.dir.iterdir()] == [path.name]  # no temp file left behind


def test_jobs_survive_a_restart(store, tmp_path, clock):
    a = make(store, "a.jpg")
    clock.now += 1
    b = make(store, "b.jpg", subject="Math", is_album=True)
    reopened = JobStore(tmp_path / "queue", clock=clock).load_all()
    assert [j.id for j in reopened] == [a.id, b.id]
    assert reopened[1].subject == "Math" and reopened[1].is_album and reopened[0].files == [a.files[0]]


def test_create_validates_input(store):
    with pytest.raises(ValueError):
        store.create([], [])
    with pytest.raises(ValueError):
        store.create(["a.jpg", "b.jpg"], ["photo"])


def test_claim_is_fifo_and_marks_running(store, clock):
    a = make(store, "a.jpg")
    clock.now += 1
    b = make(store, "b.jpg")
    first = store.claim_next()
    assert first.id == a.id and first.state == "running"
    assert store.claim_next().id == b.id
    assert store.claim_next() is None  # both running now
    assert json.loads((store.dir / f"{a.id}.json").read_text())["state"] == "running"


def test_deferred_job_is_not_claimable_until_its_retry_time(store, clock):
    job = make(store)
    claimed = store.claim_next()
    delay = store.defer(claimed, "LM Studio is off")
    assert delay == 30 and claimed.attempts == 1 and claimed.state == "pending"
    assert claimed.last_error == "LM Studio is off"
    assert store.claim_next() is None and store.seconds_until_next() == 30
    clock.now += 29
    assert store.claim_next() is None and store.seconds_until_next() == pytest.approx(1)
    clock.now += 1
    again = store.claim_next()
    assert again.id == job.id and again.attempts == 1


def test_backoff_grows_with_each_failed_attempt(store, clock):
    make(store)
    delays = []
    for _ in range(7):
        job = store.claim_next()
        delays.append(store.defer(job, "busy"))
        clock.now += delays[-1]
    assert delays == [30, 60, 120, 300, 900, 900, 900]


def test_seconds_until_next(store, clock):
    assert store.seconds_until_next() is None
    make(store)
    assert store.seconds_until_next() == 0
    store.claim_next()
    assert store.seconds_until_next() is None  # running jobs do not count


def test_a_ready_job_is_picked_before_a_deferred_older_one(store, clock):
    old = make(store, "old.jpg")
    store.defer(store.claim_next(), "busy")
    clock.now += 1
    new = make(store, "new.jpg")
    assert store.claim_next().id == new.id and old.id != new.id


def test_recover_returns_running_jobs_to_pending(store, tmp_path, clock):
    make(store)
    store.claim_next()  # the process "crashes" while it is running
    fresh = JobStore(tmp_path / "queue", clock=clock)
    jobs = fresh.recover()
    assert [j.state for j in jobs] == ["pending"]
    assert fresh.claim_next() is not None


def test_delete_is_idempotent(store):
    job = make(store)
    store.delete(job.id)
    store.delete(job.id)
    assert store.load_all() == []


def test_corrupt_job_files_are_set_aside_not_fatal(store):
    good = make(store)
    (store.dir / "broken.json").write_text("{nope")
    (store.dir / "wrongshape.json").write_text('{"unexpected": 1}')
    jobs = store.load_all()
    assert [j.id for j in jobs] == [good.id]
    assert sorted(p.name for p in (store.dir / "corrupt").iterdir()) == ["broken.json", "wrongshape.json"]


def test_pending_paths_and_counts(store):
    a = make(store, "a.jpg")
    make(store, "b.jpg")
    assert store.pending_paths() == {a.files[0], store.load_all()[1].files[0]}
    assert store.counts() == {"queued": 2, "deferred": 0, "running": 0}
    store.defer(store.claim_next(), "busy")
    store.claim_next()  # not ready (deferred) -> claims b
    assert store.counts() == {"queued": 0, "deferred": 1, "running": 1}


def test_staged_paths_are_persisted(store, tmp_path, clock):
    job = make(store)
    job.staged = ["C:/lib/Science/x.jpg"]
    store.save(job)
    assert JobStore(tmp_path / "queue", clock=clock).load_all()[0].staged == ["C:/lib/Science/x.jpg"]


def test_concurrent_creates_get_unique_ids_and_valid_files(store):
    ids, lock = [], threading.Lock()

    def worker(i):
        job = store.create([f"C:/inbox/{i}.jpg"], ["photo"])
        with lock:
            ids.append(job.id)

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(40)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert len(set(ids)) == 40 and len(store.load_all()) == 40
    assert not list(store.dir.glob("*.tmp"))


def test_jobs_created_at_the_same_instant_keep_creation_order(store):
    ids = [make(store, f"{i}.jpg").id for i in range(12)]  # the fake clock never moves: identical created_at
    assert [j.id for j in store.load_all()] == ids
    assert [store.claim_next().id for _ in range(12)] == ids
