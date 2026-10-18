import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

from conftest import FakeEmbedder
from ingest_helpers import PAGE_TEXT, FakeChat, FakeReader, make_env, make_photo
from studybuddy.errors import StudyBuddyError, TransientLLMError
from studybuddy.ingest.jobs import JobStore
from studybuddy.ingest.results import DEFERRED, DUPLICATE, FAILED, INDEXED
from studybuddy.ingest.service import IngestService
from studybuddy.ingest.subjects import SubjectError


class Clock:
    def __init__(self, now=10_000.0):
        self.now = now

    def __call__(self):
        return self.now


def build(tmp_path, settings, reader=None, chat=None, embedder=None, clock=None, results=None):
    env = make_env(tmp_path, settings, reader=reader, chat=chat, embedder=embedder)
    clock = clock or Clock()
    service = IngestService(env.settings, env.index, env.reader, None, env.chat, env.subjects,
                            on_result=(results.append if results is not None else None), clock=clock, poll_interval=0.02)
    return env, service, clock


@pytest.fixture
def running(tmp_path, settings):
    results = []
    env, service, clock = build(tmp_path, settings, results=results)
    service.start()
    yield env, service, clock, results
    service.stop()


def photo(env, name="a.jpg", **kw):
    return make_photo(env.inbox / name, **kw)


def test_submit_processes_in_the_background_and_resolves_the_ticket(running):
    env, service, _, results = running
    ticket = service.submit([photo(env)], subject="Science", source_kind="photo")
    result = ticket.result(timeout=20)
    assert result.status == INDEXED and result.subject == "Science" and ticket.reported.result() is result
    assert Path(result.final_path).exists() and env.counts()["documents"] == 1
    assert results == [result] and service.jobs.load_all() == []  # the job file is gone after the final outcome


def test_typed_subject_is_added_when_filed_and_invalid_ones_are_refused_at_once(running):
    env, service, *_ = running
    ticket = service.submit([photo(env)], subject="  earth   science ")
    result = ticket.result(timeout=20)
    assert result.status == INDEXED and result.subject == "earth science"
    assert "earth science" in env.subjects.names()  # added now that a document was really filed under it
    with pytest.raises(SubjectError, match="reserved"):
        service.submit([photo(env, "b.jpg")], subject="CON")
    assert service.jobs.load_all() == []  # nothing was queued


def test_a_known_subject_is_canonicalised_at_submit_and_a_new_one_is_not_added_yet(tmp_path, settings):
    env, service, _ = build(tmp_path, settings)  # the worker is not started: the job just sits in the queue
    service.submit([photo(env, "a.jpg")], subject="science")
    service.submit([photo(env, "b.jpg", color=(9, 9, 9))], subject="Brand New")
    assert [j.subject for j in service.jobs.load_all()] == ["Science", "Brand New"]
    assert env.subjects.names() == ["Math", "Science", "English"]


def test_a_duplicate_does_not_add_its_typed_subject_to_the_list(running):
    env, service, *_ = running
    service.submit([photo(env, "a.jpg")], subject="Science").result(timeout=20)
    result = service.submit([photo(env, "again.jpg")], subject="Brand New Subject").result(timeout=20)
    assert result.status == DUPLICATE
    assert "Brand New Subject" not in env.subjects.names()
    assert not (env.library / "Brand New Subject").exists()


def test_submit_validates_input(running):
    _, service, *_ = running
    with pytest.raises(ValueError):
        service.submit([])


def test_several_paths_need_an_album_and_are_refused_otherwise(running):
    env, service, *_ = running
    pages = [photo(env, f"{i}.jpg", color=(i * 20, 9, 9)) for i in (1, 2)]
    with pytest.raises(StudyBuddyError, match="only as an album"):
        service.submit(pages, subject="Math")
    assert service.status()["queued"] == 0 and all(p.exists() for p in pages)       # nothing was queued, nothing moved
    assert service.submit(pages, subject="Math", is_album=True).result(timeout=20).page_count == 2


def test_separate_documents_are_submitted_one_at_a_time(running):
    env, service, *_ = running
    first, second = (photo(env, f"{n}.jpg", color=c) for n, c in (("a", (10, 10, 10)), ("b", (200, 30, 30))))
    results = [service.submit([p], subject="Math").result(timeout=20) for p in (first, second)]
    assert [r.status for r in results] == ["indexed", "indexed"] and env.counts()["documents"] == 2


def test_default_kind_is_detected_later_and_kinds_can_be_given_per_file(running):
    env, service, *_ = running
    pages = [photo(env, f"{i}.jpg", color=(i * 20, 9, 9)) for i in (1, 2)]
    job_ticket = service.submit(pages, subject="Math", source_kind=["photo", "photo"], is_album=True)
    assert job_ticket.result(timeout=20).source_kinds == ["photo", "photo"]


def test_a_failed_document_reports_failed_through_the_ticket_and_callback(running):
    env, service, _, results = running
    bad = env.inbox / "x.gif"
    bad.write_bytes(b"GIF89a\x01\x00")
    result = service.submit([bad], subject="Math").result(timeout=20)
    assert result.status == FAILED and results[-1] is result and Path(result.failed_dir).exists()
    assert service.jobs.load_all() == []


def test_duplicates_are_reported_not_dropped(running):
    env, service, _, results = running
    service.submit([photo(env, "a.jpg")], subject="Science").result(timeout=20)
    second = service.submit([photo(env, "again.jpg")], subject="Science").result(timeout=20)
    assert second.status == DUPLICATE and [r.status for r in results] == [INDEXED, DUPLICATE]


# ---- transient failures and retries ----------------------------------------------------------

def test_a_transient_failure_is_reported_once_then_retried_to_success(tmp_path, settings):
    results = []
    reader = FakeReader([TransientLLMError("Cannot reach LM Studio")])
    env, service, clock = build(tmp_path, settings, reader=reader, results=results)
    service.start()
    try:
        src = photo(env)
        ticket = service.submit([src], subject="Math")
        first = ticket.reported.result(timeout=20)
        assert first.status == DEFERRED and first.transient and first.retry_in_seconds == 30 and "LM Studio" in first.error
        assert not ticket.final.done() and src.exists()
        assert len(service.jobs.load_all()) == 1 and service.status()["deferred"] == 1
        clock.now += 29
        service.wake()
        time.sleep(0.2)
        assert not ticket.final.done()  # not yet: backoff has not elapsed
        clock.now += 2
        service.wake()
        final = ticket.result(timeout=20)
        assert final.status == INDEXED and final.attempts == 2 and not src.exists()
        assert [r.status for r in results] == [DEFERRED, INDEXED]  # one "deferred" message, then the real outcome
        assert service.jobs.load_all() == []
    finally:
        service.stop()


def test_repeated_failures_keep_backing_off_and_report_deferred_only_once(tmp_path, settings):
    results = []
    reader = FakeReader([TransientLLMError("off")] * 3)
    env, service, clock = build(tmp_path, settings, reader=reader, results=results)
    service.start()
    try:
        ticket = service.submit([photo(env)], subject="Math")
        ticket.reported.result(timeout=20)
        for delay in (30, 60):
            clock.now += delay + 1
            service.wake()
            for _ in range(200):
                job = service.jobs.load_all()[0]
                if job.attempts == {30: 2, 60: 3}[delay]:
                    break
                time.sleep(0.02)
        assert [r.status for r in results] == [DEFERRED]  # no message storm while LM Studio stays off
        clock.now += 200
        service.wake()
        assert ticket.result(timeout=20).status == INDEXED
    finally:
        service.stop()


def test_a_deferred_job_does_not_block_the_ones_behind_it(tmp_path, settings):
    reader = FakeReader([TransientLLMError("busy")])  # only the first read fails
    env, service, clock = build(tmp_path, settings, reader=reader)
    service.start()
    try:
        first = service.submit([photo(env, "a.jpg", color=(1, 1, 1))], subject="Math")
        second = service.submit([photo(env, "b.jpg", color=(2, 2, 2))], subject="Math")
        assert first.reported.result(timeout=20).status == DEFERRED
        assert second.result(timeout=20).status == INDEXED
        clock.now += 31
        service.wake()
        assert first.result(timeout=20).status == INDEXED
    finally:
        service.stop()


def test_queued_jobs_survive_a_restart(tmp_path, settings):
    env, service, clock = build(tmp_path, settings)
    src = photo(env)
    service.submit([src], subject="Math")  # queued but the worker never started: "the app crashed"
    assert src.exists() and len(service.jobs.load_all()) == 1 and env.counts()["documents"] == 0

    results = []
    again = IngestService(env.settings, env.index, env.reader, None, env.chat, env.subjects,
                          on_result=results.append, clock=clock, poll_interval=0.02)
    again.start()
    try:
        for _ in range(500):
            if results:
                break
            time.sleep(0.02)
        assert results and results[0].status == INDEXED and not src.exists()
    finally:
        again.stop()


def test_a_job_left_running_by_a_crash_is_picked_up_again(tmp_path, settings):
    env, service, clock = build(tmp_path, settings)
    src = photo(env)
    service.submit([src], subject="Math")
    service.jobs.claim_next()  # marked running, then the process died
    results = []
    again = IngestService(env.settings, env.index, env.reader, None, env.chat, env.subjects,
                          on_result=results.append, clock=clock, poll_interval=0.02).start()
    try:
        for _ in range(500):
            if results:
                break
            time.sleep(0.02)
        assert results and results[0].status == INDEXED
    finally:
        again.stop()


# ---- single writer ---------------------------------------------------------------------------

class Gate:
    """Counts how many DIFFERENT threads are inside at once (nested entry by one thread is fine), to prove writes
    never overlap."""

    def __init__(self):
        self.lock = threading.Lock()
        self.depth: dict[int, int] = {}
        self.max_threads = 0
        self.entries = 0

    def __enter__(self):
        me = threading.get_ident()
        with self.lock:
            self.depth[me] = self.depth.get(me, 0) + 1
            self.entries += 1
            self.max_threads = max(self.max_threads, len(self.depth))
        time.sleep(0.005)  # widen the window in which an overlap would show up

    def __exit__(self, *exc):
        me = threading.get_ident()
        with self.lock:
            self.depth[me] -= 1
            if not self.depth[me]:
                del self.depth[me]


def test_twenty_concurrent_submissions_are_each_indexed_exactly_once_one_at_a_time(tmp_path, settings):
    env, service, _ = build(tmp_path, settings)
    gate = Gate()
    real_add, real_process = env.index.add_document, service.processor.process

    def gated_add(*a, **k):
        with gate:
            return real_add(*a, **k)

    def gated_process(job):
        with gate:
            return real_process(job)

    env.index.add_document = gated_add
    service.processor.process = gated_process
    service.start()
    try:
        paths = [photo(env, f"p{i}.jpg", color=(i * 11 % 256, i * 7 % 256, i * 3 % 256)) for i in range(20)]
        barrier = threading.Barrier(20)

        def submit(path):
            barrier.wait()  # all 20 threads hit submit() at the same instant
            return service.submit([path], subject="Math")

        with ThreadPoolExecutor(20) as pool:
            tickets = list(pool.map(submit, paths))
        results = [t.result(timeout=60) for t in tickets]
    finally:
        service.stop()
    assert all(r.status == INDEXED for r in results)
    assert len({r.job_id for r in results}) == 20 and len({r.final_path for r in results}) == 20
    assert env.counts() == {"documents": 20, "pages": 20, "chunks": 20} and env.index.vectors.count() == 20
    assert env.index.check().ok
    assert gate.entries == 40 and gate.max_threads == 1  # processing and index writes never overlapped
    assert len(list((env.library / "Math").iterdir())) == 20 and not any(p.exists() for p in paths)


def test_the_same_file_submitted_many_times_at_once_is_indexed_once(tmp_path, settings):
    env, service, _ = build(tmp_path, settings)
    service.start()
    try:
        copies = [photo(env, f"copy{i}.jpg") for i in range(8)]  # identical bytes
        with ThreadPoolExecutor(8) as pool:
            tickets = list(pool.map(lambda p: service.submit([p], subject="Math"), copies))
        statuses = sorted(t.result(timeout=60).status for t in tickets)
    finally:
        service.stop()
    assert statuses == [DUPLICATE] * 7 + [INDEXED] and env.counts()["documents"] == 1
    assert len(list((env.data / "duplicates").iterdir())) == 7


def test_run_exclusive_runs_between_jobs_on_the_worker_thread(tmp_path, settings):
    env, service, _ = build(tmp_path, settings)
    gate = Gate()
    real_process = service.processor.process

    def gated_process(job):
        with gate:
            return real_process(job)

    service.processor.process = gated_process
    service.start()
    try:
        tickets = [service.submit([photo(env, f"{i}.jpg", color=(i * 40, 3, 3))], subject="Math") for i in range(4)]
        threads_seen = []

        def action(index):
            with gate:
                threads_seen.append(threading.current_thread().name)
                return index.store.counts()["documents"]

        futures = [service.run_exclusive(action) for _ in range(5)]
        counts = [f.result(timeout=60) for f in futures]
        [t.result(timeout=60) for t in tickets]
    finally:
        service.stop()
    assert gate.max_threads == 1 and set(threads_seen) == {"ingest-worker"}
    assert counts == sorted(counts)  # the document count only grows between actions: nothing interleaved


def test_run_exclusive_returns_values_and_propagates_exceptions(running):
    env, service, *_ = running
    assert service.run_exclusive(lambda index: 42).result(timeout=10) == 42

    def boom(index):
        raise RuntimeError("ui action failed")

    with pytest.raises(RuntimeError, match="ui action failed"):
        service.run_exclusive(boom).result(timeout=10)
    assert service.run_exclusive(lambda index: "still alive").result(timeout=10) == "still alive"


def test_ui_actions_can_change_the_index_safely(running):
    env, service, *_ = running
    result = service.submit([photo(env)], subject="Math").result(timeout=20)
    service.run_exclusive(lambda index: index.set_subject(result.doc_id, "Science")).result(timeout=10)
    assert env.index.documents()[0].subject == "Science"
    assert service.run_exclusive(lambda index: index.delete_document(result.doc_id)).result(timeout=10) == 1
    assert env.counts()["documents"] == 0


def test_a_failing_callback_never_stops_the_worker(tmp_path, settings):
    env, service, _ = build(tmp_path, settings)

    def bad(result):
        raise RuntimeError("callback bug")

    service.add_callback(bad)
    service.start()
    try:
        assert service.submit([photo(env, "a.jpg", color=(1, 1, 1))], subject="Math").result(timeout=20).status == INDEXED
        assert service.submit([photo(env, "b.jpg", color=(2, 2, 2))], subject="Math").result(timeout=20).status == INDEXED
    finally:
        service.stop()


def test_a_processor_crash_becomes_a_failed_result_not_a_dead_worker(tmp_path, settings):
    env, service, _ = build(tmp_path, settings)
    calls = {"n": 0}
    real = service.processor.process

    def flaky(job):
        calls["n"] += 1
        if calls["n"] == 1:
            raise RuntimeError("processor exploded")
        return real(job)

    service.processor.process = flaky
    service.start()
    try:
        first = service.submit([photo(env, "a.jpg", color=(1, 1, 1))], subject="Math").result(timeout=20)
        second = service.submit([photo(env, "b.jpg", color=(2, 2, 2))], subject="Math").result(timeout=20)
    finally:
        service.stop()
    assert first.status == FAILED and "exploded" in first.error and second.status == INDEXED


def test_stop_finishes_the_job_in_flight_and_keeps_the_rest_queued(tmp_path, settings):
    started, release = threading.Event(), threading.Event()
    reader = FakeReader()
    real = reader.transcribe

    def slow(image):
        started.set()
        release.wait(10)
        return real(image)

    reader.transcribe = slow
    env, service, _ = build(tmp_path, settings, reader=reader)
    service.start()
    first = service.submit([photo(env, "a.jpg", color=(1, 1, 1))], subject="Math")
    second = service.submit([photo(env, "b.jpg", color=(2, 2, 2))], subject="Math")
    assert started.wait(10)
    stopper = threading.Thread(target=service.stop)
    stopper.start()
    time.sleep(0.1)
    release.set()
    stopper.join(20)
    assert first.result(timeout=5).status == INDEXED  # the job in flight was finished, not abandoned
    assert not second.final.done() and len(service.jobs.load_all()) == 1  # the next one waits on disk for next start
    with pytest.raises(StudyBuddyError, match="stopping"):
        service.submit([photo(env, "c.jpg")], subject="Math")


def test_retry_failed_requeues_what_failed_before(tmp_path, settings):
    env, service, _ = build(tmp_path, settings)
    service.start()
    try:
        bad = env.inbox / "later-fixed.jpg"
        bad.write_bytes(b"not an image yet")
        assert service.submit([bad], subject="Math").result(timeout=20).status == FAILED
        failed_files = list((env.data / "failed").glob("*/later-fixed.jpg"))
        assert len(failed_files) == 1
        make_photo(failed_files[0])  # the student replaces it with a good file in failed/
        tickets = service.retry_failed()
        assert len(tickets) == 1
        result = tickets[0].result(timeout=20)
        assert result.status == INDEXED and result.subject == "Math"
        assert service.retry_failed() == []  # reason.json was marked as retried
    finally:
        service.stop()


def test_status_reports_queue_counts(tmp_path, settings):
    env, service, _ = build(tmp_path, settings)
    service.submit([photo(env)], subject="Math")
    assert service.status() == {"queued": 1, "deferred": 0, "running": 0, "busy": False, "exclusive_waiting": 0}


def test_wait_idle(running):
    env, service, *_ = running
    service.submit([photo(env)], subject="Math")
    assert service.wait_idle(timeout=20) and env.counts()["documents"] == 1


def test_the_overlap_detector_really_detects_overlap():
    """Guards the concurrency tests above: without the single writer, the same gate must see several threads inside."""
    gate = Gate()
    barrier = threading.Barrier(4)

    def work():
        barrier.wait()
        with gate:
            time.sleep(0.05)

    threads = [threading.Thread(target=work) for _ in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert gate.max_threads > 1


def test_try_exclusive_runs_when_idle_and_returns_values(running):
    _, service, *_ = running
    future = service.try_exclusive(lambda index: 7)
    assert future is not None and future.result(timeout=10) == 7


def test_try_exclusive_refuses_while_a_job_runs_and_queues_nothing(tmp_path, settings):
    env, service, _ = build(tmp_path, settings)
    started, release = threading.Event(), threading.Event()
    real_process = service.processor.process

    def slow_process(job):
        started.set()
        assert release.wait(30)
        return real_process(job)

    service.processor.process = slow_process
    service.start()
    try:
        ticket = service.submit([photo(env)], subject="Math")
        assert started.wait(10)
        assert service.try_exclusive(lambda index: "late") is None
        assert service.status()["exclusive_waiting"] == 0  # refused, not queued: it must not run later
        release.set()
        assert ticket.result(timeout=30).status == INDEXED
        assert service.wait_idle(10)
        assert service.try_exclusive(lambda index: "now").result(timeout=10) == "now"
    finally:
        release.set()
        service.stop()


def test_try_exclusive_refuses_while_another_action_is_waiting(tmp_path, settings):
    _, service, _ = build(tmp_path, settings)  # not started: the first action stays queued
    first = service.try_exclusive(lambda index: 1)
    assert first is not None and service.try_exclusive(lambda index: 2) is None
    service.start()
    try:
        assert first.result(timeout=10) == 1
    finally:
        service.stop()


def test_try_exclusive_after_stop_raises(tmp_path, settings):
    _, service, _ = build(tmp_path, settings)
    service.start()
    service.stop()
    with pytest.raises(StudyBuddyError, match="stopping"):
        service.try_exclusive(lambda index: 1)
