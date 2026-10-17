import dataclasses
import io
import json
import subprocess
import sys
from pathlib import Path

import pytest

from ingest_helpers import PAGE_TEXT, FakeChat, FakeReader, make_env, make_photo
from studybuddy.errors import TransientLLMError
from studybuddy.ingest.cli import format_result, ingest_main
from studybuddy.ingest.jobs import JobStore
from studybuddy.ingest.results import DEFERRED, DUPLICATE, FAILED, INDEXED, FlaggedPage, IngestResult
from studybuddy.ingest.service import IngestService

ROOT = Path(__file__).resolve().parents[1]


def run(argv, env, reader=None, chat=None):
    """Run the CLI against the test environment's settings with fake models."""
    env.reader.default = PAGE_TEXT
    holder = {}

    def factory(settings):
        service = IngestService(env.settings, env.index, reader or env.reader, None, chat or env.chat, env.subjects,
                                poll_interval=0.02)
        holder["service"] = service
        return service, service.stop

    out = io.StringIO()
    code = ingest_main(argv, env.settings, factory, out)
    return code, out.getvalue()


@pytest.fixture
def env(tmp_path, settings):
    return make_env(tmp_path, settings)


def test_once_ingests_a_photo_and_prints_the_result(env):
    src = make_photo(env.inbox / "a.jpg")
    code, text = run(["once", str(src), "--subject", "Science"], env)
    assert code == 0 and text.startswith("INDEXED   Science | Photosynthesis Worksheet | 1 page | subject user")
    assert "saved:" in text and "summary:" in text and not src.exists()
    assert env.counts()["documents"] == 1


def test_once_keep_leaves_the_original_in_place(env):
    src = make_photo(env.inbox / "a.jpg")
    code, text = run(["once", str(src), "--subject", "Math", "--keep"], env)
    assert code == 0 and src.exists() and "INDEXED" in text


def test_once_album(env):
    pages = [make_photo(env.inbox / f"{i}.jpg", color=(i * 40, 5, 5)) for i in (1, 2, 3)]
    code, text = run(["once", *map(str, pages), "--album", "--subject", "Science"], env)
    assert code == 0 and "3 pages" in text and "pdf:" in text and env.counts()["pages"] == 3


def test_once_auto_detects_the_subject(env):
    code, text = run(["once", str(make_photo(env.inbox / "a.jpg"))], env)
    assert code == 0 and "Science | Photosynthesis Worksheet | 1 page | subject auto" in text


def test_once_reports_a_duplicate(env):
    first, second = make_photo(env.inbox / "a.jpg"), make_photo(env.inbox / "b.jpg")
    run(["once", str(first), "--subject", "Science"], env)
    code, text = run(["once", str(second), "--subject", "Science"], env)
    assert code == 0 and text.startswith("DUPLICATE") and "already indexed as 'Photosynthesis Worksheet'" in text
    assert "kept in:" in text


def test_once_reports_a_failure_with_exit_code_1(env):
    bad = env.inbox / "x.gif"
    bad.write_bytes(b"GIF89a\x01\x00")
    code, text = run(["once", str(bad), "--subject", "Math"], env)
    assert code == 1 and text.startswith("FAILED") and "GIF" in text and "files kept in:" in text


def test_once_with_lm_studio_off_defers_with_exit_code_3(env):
    src = make_photo(env.inbox / "a.jpg")
    code, text = run(["once", str(src), "--subject", "Math"], env, reader=FakeReader([TransientLLMError("Cannot reach LM Studio")]))
    assert code == 3 and text.startswith("DEFERRED") and "will retry in about 30 s" in text
    assert "stays queued" in text and src.exists() and len(JobStore(env.data / "queue").load_all()) == 1


def test_once_with_a_missing_file(env):
    code, text = run(["once", str(env.inbox / "nope.jpg")], env)
    assert code == 1 and "not found" in text


def test_once_with_an_invalid_subject_is_a_clean_failure(env):
    code, text = run(["once", str(make_photo(env.inbox / "a.jpg")), "--subject", "CON"], env)
    assert code == 1 and "[FAIL]" in text and "reserved" in text


def test_run_until_idle_processes_the_inbox_and_exits(env):
    make_photo(env.inbox / "Science" / "a.jpg", color=(1, 1, 1))
    make_photo(env.inbox / "b.jpg", color=(2, 2, 2))
    code, text = run(["run", "--until-idle"], env)
    assert code == 0 and text.count("INDEXED") == 2 and "Watching" in text
    assert env.counts()["documents"] == 2 and not any(env.inbox.rglob("*.jpg"))


def test_run_until_idle_with_nothing_to_do_exits_promptly(env):
    code, text = run(["run", "--until-idle"], env)
    assert code == 0 and "Watching" in text and "INDEXED" not in text


def test_run_until_idle_reports_deferred_documents(env):
    make_photo(env.inbox / "a.jpg")
    code, text = run(["run", "--until-idle"], env, reader=FakeReader([TransientLLMError("off")]))
    assert code == 3 and "DEFERRED" in text and "1 document(s) are waiting to be retried" in text


def test_status_lists_queue_failures_duplicates_and_subjects(env):
    env.jobs.create([str(env.inbox / "q.jpg")], ["photo"], None)
    failed = env.data / "failed" / "job1"
    failed.mkdir(parents=True)
    (failed / "reason.json").write_text(json.dumps({"files": [str(failed / "bad.gif")], "error": "GIF images are not supported"}))
    (env.data / "duplicates").mkdir()
    (env.data / "duplicates" / "abc_dup.jpg").write_bytes(b"x")
    make_photo(env.inbox / "waiting.jpg")
    out = io.StringIO()
    assert ingest_main(["status"], env.settings, None, out) == 0
    text = out.getvalue()
    assert "Queue: 1 waiting, 0 deferred" in text and "Failed: 1" in text and "bad.gif - GIF images are not supported" in text
    assert "Duplicates kept: 1" in text and "Inbox: 1 file(s) waiting" in text and "Subjects: Math, Science, English" in text


def test_retry_failed_requeues_and_processes(env):
    bad = env.inbox / "bad.jpg"
    bad.write_bytes(b"not an image yet")
    assert run(["once", str(bad), "--subject", "Math"], env)[0] == 1
    (failed,) = list((env.data / "failed").glob("*/bad.jpg"))
    make_photo(failed, color=(7, 7, 7))
    code, text = run(["retry-failed"], env)
    assert code == 0 and "Requeued 1 failed document(s)" in text and "INDEXED" in text


def test_format_result_for_each_status():
    flagged = FlaggedPage(2, 0.3, ["too_short"], ["very little text was found for this image size"], True)
    indexed = format_result(IngestResult(INDEXED, "j", files=["a.jpg"], subject="Science", subject_source="auto",
                                         title="T", summary="S", page_count=2, final_path="C:/lib/x.jpg",
                                         flagged_pages=[flagged], warnings=["something small"]))
    assert "2 pages" in indexed and "page 2: very little text" in indexed and "(resend as a file?)" in indexed
    assert "note: something small" in indexed
    deferred = format_result(IngestResult(DEFERRED, "j", files=["a.jpg"], error="LM Studio is off", retry_in_seconds=60, attempts=2))
    assert "attempt 2" in deferred and "60 s" in deferred
    assert format_result(IngestResult(FAILED, "j", files=["a.jpg"], error="bad")).startswith("FAILED    a.jpg")


def test_missing_settings_is_a_clean_failure(monkeypatch):
    for name in ("SB_READER_MODEL", "SB_CHAT_MODEL", "SB_EMBED_MODEL", "SB_LIBRARY_DIR", "SB_DATA_DIR"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr("studybuddy.config.load_dotenv", lambda *a, **k: None)
    out = io.StringIO()
    assert ingest_main(["status"], None, None, out) == 1 and "Missing required settings" in out.getvalue()


def test_script_help_runs():
    out = subprocess.run([sys.executable, str(ROOT / "scripts" / "ingest.py"), "--help"],
                         capture_output=True, text=True, timeout=60)
    assert out.returncode == 0 and "retry-failed" in out.stdout and "once" in out.stdout
    run_help = subprocess.run([sys.executable, str(ROOT / "scripts" / "ingest.py"), "run", "--help"],
                              capture_output=True, text=True, timeout=60)
    assert run_help.returncode == 0 and "--until-idle" in run_help.stdout
