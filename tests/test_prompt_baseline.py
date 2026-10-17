import json
import shutil
import subprocess
from pathlib import Path

import pytest

from studybuddy.reader import prompt_baseline as pb

ROOT = Path(__file__).resolve().parents[1]
GOOD = {"prompt_sha256": "abc", "recorded": "2026-10-17", "reader_model": "m", "pages": 2,
        "reader_cer": 0.03, "reader_wer": 0.06}


def write(path, data):
    path.write_text(json.dumps(data), encoding="utf-8")
    return path


def test_current_prompt_matches_the_recorded_bakeoff():
    """FAILS when the READER prompt is edited without re-running the bake-off.

    Fix: python scripts/evaluate_reader.py --compare-raw --record-prompt-baseline
    then commit docs/reader-prompt-baseline.json with a 'Bake-off: ...' line in the message.
    """
    assert pb.check_baseline() == []


def test_valid_baseline_passes(tmp_path):
    assert pb.check_baseline(write(tmp_path / "b.json", GOOD), current_hash="abc") == []


def test_changed_prompt_is_reported_with_instructions(tmp_path):
    problems = pb.check_baseline(write(tmp_path / "b.json", GOOD), current_hash="different")
    assert len(problems) == 1 and "--record-prompt-baseline" in problems[0] and "Bake-off:" in problems[0]


def test_editing_the_prompt_changes_the_hash(monkeypatch):
    before = pb.prompt_hash()
    monkeypatch.setattr(pb, "USER_PROMPT", pb.USER_PROMPT + " Be creative.")
    assert pb.prompt_hash() != before


def test_missing_invalid_and_incomplete_baselines(tmp_path):
    assert "missing" in pb.check_baseline(tmp_path / "none.json")[0]
    bad = tmp_path / "bad.json"
    bad.write_text("{nope")
    assert "not valid JSON" in pb.check_baseline(bad)[0]
    partial = write(tmp_path / "p.json", {"prompt_sha256": "abc"})
    assert any("reader_cer" in p for p in pb.check_baseline(partial, current_hash="abc"))


def test_baseline_needs_a_scored_page(tmp_path):
    problems = pb.check_baseline(write(tmp_path / "b.json", {**GOOD, "pages": 0}), current_hash="abc")
    assert any("at least 1 scored page" in p for p in problems)


def test_record_baseline_writes_summary_for_current_prompt(tmp_path):
    summary = {"pages": 3, "errors": 1, "reader_cer": 0.04321, "reader_wer": 0.08, "ocr_cer": 0.2, "agreement": 0.9}
    path = tmp_path / "b.json"
    rec = pb.record_baseline(summary, "qwen", "samples", "real pages", path=path, current_hash="h")
    assert rec["pages"] == 2 and rec["reader_cer"] == 0.0432 and rec["note"] == "real pages"
    assert pb.check_baseline(path, current_hash="h") == []
    assert pb.check_baseline(path, current_hash="other") != []


def test_record_baseline_refuses_when_nothing_scored(tmp_path):
    with pytest.raises(ValueError, match="no page was scored"):
        pb.record_baseline({"pages": 2, "errors": 2, "reader_cer": None}, "m", "s", path=tmp_path / "b.json")
    assert not (tmp_path / "b.json").exists()


@pytest.mark.skipif(shutil.which("sh") is None or shutil.which("git") is None, reason="needs sh and git")
@pytest.mark.parametrize("message,ok", [("feat: tune prompt\n", False),
                                        ("feat: tune prompt\n\nBake-off: qwen, 3 pages, CER 0.05 (was 0.07)\n", True),
                                        ("feat: tune prompt\n\nBake-off:\n", False)])
def test_commit_msg_hook_requires_bakeoff_line_when_baseline_changes(tmp_path, message, ok):
    def git(*args):
        subprocess.run(["git", *args], cwd=tmp_path, check=True, capture_output=True)

    git("init", "-q")
    (tmp_path / "docs").mkdir()
    (tmp_path / "docs" / "reader-prompt-baseline.json").write_text("{}")
    git("add", "docs/reader-prompt-baseline.json")
    msg = tmp_path / "MSG"
    msg.write_text(message)
    hook = ROOT / "scripts" / "git-hooks" / "commit-msg"
    run = subprocess.run(["sh", hook.as_posix(), "MSG"], cwd=tmp_path, capture_output=True, text=True)
    assert (run.returncode == 0) == ok, run.stderr


@pytest.mark.skipif(shutil.which("sh") is None or shutil.which("git") is None, reason="needs sh and git")
def test_commit_msg_hook_ignores_commits_that_do_not_touch_the_baseline(tmp_path):
    subprocess.run(["git", "init", "-q"], cwd=tmp_path, check=True, capture_output=True)
    (tmp_path / "x.txt").write_text("x")
    subprocess.run(["git", "add", "x.txt"], cwd=tmp_path, check=True, capture_output=True)
    (tmp_path / "MSG").write_text("chore: unrelated\n")
    hook = ROOT / "scripts" / "git-hooks" / "commit-msg"
    assert subprocess.run(["sh", hook.as_posix(), "MSG"], cwd=tmp_path).returncode == 0
