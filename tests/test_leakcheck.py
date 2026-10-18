import shutil
import subprocess

import pytest

from studybuddy.leakcheck import (check_history, check_working_tree, compile_names, format_report, mask, read_names_file, real_findings,
                                  scan_text)

pytestmark = pytest.mark.skipif(shutil.which("git") is None, reason="needs git")

TOKEN = "123456789" + ":AAHrealLooking_token-value0123456789abc"      # built in pieces so this file itself is clean
AT = "@"


def git(repo, *args):
    subprocess.run(["git", *args], cwd=repo, check=True, capture_output=True, text=True)


@pytest.fixture
def repo(tmp_path):
    git(tmp_path, "init", "-q", "-b", "main")
    git(tmp_path, "config", "user.name", "Some Student")
    git(tmp_path, "config", "user.email", f"student{AT}school.example.org")
    git(tmp_path, "config", "commit.gpgsign", "false")
    return tmp_path


def commit(repo, files, message="chore: add"):
    for name, content in files.items():
        path = repo / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content if isinstance(content, bytes) else content.encode())
    git(repo, "add", "-A")
    git(repo, "commit", "-q", "-m", message)


def kinds(findings):
    return sorted({f.kind for f in findings})


def test_a_clean_history_has_no_findings(repo):
    commit(repo, {"a.py": "print('hello')\n", "README.md": "# Notes\n"})
    findings, stats = check_history(repo, [])
    assert findings == [] and stats["commits"] == 1


def test_a_token_in_an_old_commit_is_found_even_after_it_was_removed(repo):
    commit(repo, {"config.txt": f"bot = {TOKEN}\n"})
    commit(repo, {"config.txt": "bot = removed\n"}, "fix: remove")
    findings, _ = check_history(repo, [])
    assert "token" in kinds(findings) and all(TOKEN not in f.line for f in findings)


def test_obvious_test_values_are_flagged_as_such_not_hidden(repo):
    commit(repo, {"t.py": 'TOKEN = "123456789' + ':AAHfake_token-value0123456789abcdefg"\n'})
    findings, _ = check_history(repo, [])
    assert kinds(findings) == ["test-token"] and "fine in tests" in findings[0].note


def test_your_own_words_are_found_in_files_and_messages(repo):
    commit(repo, {"notes.txt": "Written by Maria Santos of Section Rizal\n"}, "docs: notes for Maria Santos")
    findings, _ = check_history(repo, compile_names(["maria santos", "section rizal"]))
    where = " ".join(f.where for f in findings if f.kind == "name")
    assert "message" in where and "notes.txt" in where


def test_photos_databases_and_env_files_are_found_by_name_and_when_binary(repo):
    commit(repo, {"img/page.jpg": b"\xff\xd8\xff\x00\x01", ".env": "X=1\n", "data/index.db": b"SQLite\x00"})
    findings, _ = check_history(repo, [])
    paths = " ".join(f.line for f in findings if f.kind == "file")
    assert "page.jpg" in paths and ".env" in paths and "index.db" in paths


def test_a_real_looking_email_and_user_path_are_reported_but_noreply_is_not(repo):
    home = "C:" + "\\Users\\Kid2008\\Documents\\x"
    commit(repo, {"x.md": f"mail me at kid{AT}gmail.com or noreply{AT}anthropic.com, file {home}\n"})
    findings, _ = check_history(repo, [])
    assert kinds(findings) == ["email", "user-path"] and not any("anthropic" in f.line for f in findings)


def test_every_branch_is_scanned(repo):
    commit(repo, {"a.txt": "fine\n"})
    git(repo, "checkout", "-q", "-b", "feature")
    commit(repo, {"b.txt": f"{TOKEN}\n"})
    git(repo, "checkout", "-q", "main")
    findings, _ = check_history(repo, [])
    assert "token" in kinds(findings)


def test_the_working_tree_includes_untracked_files_but_not_ignored_ones(repo):
    commit(repo, {".gitignore": ".env\n", "a.txt": "ok\n"})
    (repo / ".env").write_text(f"T={TOKEN}\n")
    (repo / "scratch.txt").write_text(f"T={TOKEN}\n")
    findings = check_working_tree(repo, [])
    assert [f.where for f in findings if f.kind == "token"] == ["working tree scratch.txt"]


def test_secret_assignments_ignore_placeholders():
    seen = set()
    assert scan_text("api_key" + ' = "abcdef0123456789abcdef"', "x", [], seen)[0].kind == "secret"
    assert scan_text("api_key" + " = your-key-here-please", "x", [], set()) == []
    assert scan_text("TELEGRAM_BOT_TOKEN=", "x", [], set()) == []


def test_the_report_lists_authors_and_masks_values(repo):
    commit(repo, {"a.txt": f"{TOKEN}\n"})
    findings, stats = check_history(repo, [])
    text = format_report(findings, stats, ["main"])
    assert f"Some Student <student{AT}school.example.org>" in text and TOKEN not in text and "1 finding" in text
    assert format_report([], stats, ["main"]).endswith("No findings.")
    assert real_findings(scan_text("123456789" + ":AAHfake_token-value0123456789abcdefg", "x", [], set())) == []


def test_names_file_ignores_comments_and_a_missing_file(tmp_path):
    (tmp_path / ".leakcheck").write_text("# my words\nMaria Santos\n\n", encoding="utf-8")
    assert [p.pattern for p in compile_names(read_names_file(tmp_path / ".leakcheck"))] == ["Maria\\ Santos"]
    assert read_names_file(tmp_path / "nope") == [] and mask("abcdefghijkl") == "abcd...kl"
