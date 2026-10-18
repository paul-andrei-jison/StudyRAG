"""Look for things that must not be in the repository: tokens, personal names, e-mail addresses, photos, databases, `.env`.

  python scripts/leak_check.py [--names-file .leakcheck] [--name TEXT ...] [--path DIR]

Scans EVERY commit on EVERY branch (added and removed lines, commit messages, file names) and the working tree. Your own words (full name,
school, section, user id ...) are not written in this program: give them with --name or in `.leakcheck`, one per line (that file is
ignored by git, so it never leaves your computer). Only reports; it never changes history. Matches are masked in the output.
"""
from __future__ import annotations

import re
import subprocess
from dataclasses import dataclass
from pathlib import Path

TOKEN = re.compile(r"\b\d{6,}:[A-Za-z0-9_-]{30,}")
EMAIL = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9-]+(?:\.[A-Za-z0-9-]+)+")
SECRET_ASSIGNMENT = re.compile(r"(?i)\b(?:api[_-]?key|secret|passw(?:or)?d|token)\b\s*[=:]\s*[\"']?([A-Za-z0-9_\-./+]{12,})")
USER_PATH = re.compile(r"(?i)\b[A-Z]:[\\/]+Users[\\/]+([^\\/\s\"']+)")
LOCAL_DEFAULT_USERS = {"user", "users", "public", "default", "<name>", "you", "name", "yourname"}
HARMLESS_EMAILS = ("noreply@anthropic.com", "example.com", "@example.", "users.noreply.github.com")
BAD_EXTENSIONS = {".jpg", ".jpeg", ".png", ".heic", ".heif", ".gif", ".bmp", ".tif", ".tiff", ".webp", ".pdf", ".db", ".sqlite", ".sqlite3",
                  ".lance", ".bak", ".log", ".onnx", ".zip"}
BAD_NAMES = {".env"}
SAFE_PLACEHOLDER = re.compile(r"(?i)fake|example|dummy|placeholder|test|xxxx|your[-_ ]|<")


@dataclass(frozen=True)
class Finding:
    kind: str            # token | secret | email | name | user-path | file
    where: str           # commit (short) + file, or "working tree" + file
    line: str            # masked
    note: str = ""


def mask(text: str, keep: int = 4) -> str:
    return text if len(text) <= keep * 2 else text[:keep] + "..." + text[-2:]


def _run(args: list[str], cwd: Path) -> str:
    done = subprocess.run(["git", *args], cwd=cwd, capture_output=True, text=True, encoding="utf-8", errors="replace")
    if done.returncode != 0:
        raise RuntimeError(f"git {' '.join(args[:2])} failed: {done.stderr.strip()[:200]}")
    return done.stdout


def scan_text(text: str, where: str, names: list[re.Pattern], seen: set) -> list[Finding]:
    out: list[Finding] = []

    def add(kind, matched, note=""):
        key = (kind, where.split(" ")[0], matched)
        if key not in seen:
            seen.add(key)
            out.append(Finding(kind, where, mask(matched), note))

    for m in TOKEN.finditer(text):
        fake = bool(SAFE_PLACEHOLDER.search(m.group()))
        add("test-token" if fake else "token", m.group(), "token-shaped but marked fake: fine in tests" if fake else "looks like a Telegram bot token")
    for m in SECRET_ASSIGNMENT.finditer(text):
        value = m.group(1)
        if not SAFE_PLACEHOLDER.search(value) and not SAFE_PLACEHOLDER.search(m.group(0)):
            add("secret", value, "value assigned to a secret-looking name")
    for m in EMAIL.finditer(text):
        if not any(h in m.group().lower() for h in HARMLESS_EMAILS):
            add("email", m.group())
    for m in USER_PATH.finditer(text):
        if m.group(1).lower() not in LOCAL_DEFAULT_USERS:
            add("user-path", m.group(), "a path with a real Windows user name")
    for pattern in names:
        for m in pattern.finditer(text):
            add("name", m.group(), "matches one of your words")
    return out


def compile_names(words: list[str]) -> list[re.Pattern]:
    return [re.compile(re.escape(w.strip()), re.IGNORECASE) for w in words if w.strip() and not w.strip().startswith("#")]


def read_names_file(path: Path) -> list[str]:
    return path.read_text(encoding="utf-8").splitlines() if path.is_file() else []


def file_findings(paths: set[str], where: str) -> list[Finding]:
    out = []
    for p in sorted(paths):
        pure = Path(p)
        if pure.name in BAD_NAMES or pure.suffix.lower() in BAD_EXTENSIONS or pure.parts[:1] in (("data",), ("samples",)) or ".lance" in p:
            out.append(Finding("file", f"{where}: {p}", p, "a photo, database, log, backup or secret file must never be committed"))
    return out


def check_history(repo: Path, names: list[re.Pattern]) -> tuple[list[Finding], dict]:
    """One pass over `git log --all -p`: commit messages, added/removed lines, and every file name ever committed."""
    findings: list[Finding] = []
    seen: set = set()
    all_paths: dict[str, str] = {}
    stats = {"commits": 0, "authors": set()}
    log = _run(["log", "--all", "-p", "-U0", "--no-color", "--no-ext-diff", "--format=@@@%h|%an|%ae|%s"], repo)
    commit, current_file = "?", ""
    for line in log.splitlines():
        if line.startswith("@@@"):
            commit, author, email, subject = line[3:].split("|", 3)
            stats["commits"] += 1
            stats["authors"].add(f"{author} <{email}>")
            findings += scan_text(subject, f"commit {commit} message", names, seen)
            current_file = ""
        elif line.startswith("+++ b/"):
            current_file = line[6:]
            all_paths.setdefault(current_file, commit)
        elif line.startswith("--- a/") or line.startswith("diff --git") or line.startswith("@@") or line.startswith(("index ", "new file", "deleted file", "similarity", "rename ")):
            continue
        elif line.startswith(("+", "-")):
            findings += scan_text(line[1:], f"commit {commit} {current_file}", names, seen)
        elif line.strip():
            findings += scan_text(line, f"commit {commit} message", names, seen)
    for line in _run(["log", "--all", "--numstat", "--format=@@@%h", "--no-color"], repo).splitlines():
        if line.startswith("@@@"):
            commit = line[3:]
        elif line.count("\t") == 2:
            all_paths.setdefault(line.split("\t")[2], commit)
            if line.startswith("-\t-\t"):
                findings.append(Finding("file", f"commit {commit}: {line.split(chr(9))[2]}", line.split("\t")[2], "binary file committed"))
    for path, first in all_paths.items():
        findings += file_findings({path}, f"commit {first}")
    return findings, stats


def check_working_tree(repo: Path, names: list[re.Pattern]) -> list[Finding]:
    """Tracked files and untracked files that git does not ignore (those could be committed by accident)."""
    findings: list[Finding] = []
    seen: set = set()
    listed = _run(["ls-files", "-co", "--exclude-standard", "-z"], repo).split("\0")
    findings += file_findings({p for p in listed if p}, "working tree")
    for rel in filter(None, listed):
        path = repo / rel
        try:
            if path.stat().st_size > 2_000_000:
                continue
            text = path.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError):
            continue
        findings += scan_text(text, f"working tree {rel}", names, seen)
    return findings


def real_findings(findings: list[Finding]) -> list[Finding]:
    return [f for f in findings if f.kind != "test-token"]


def format_report(findings: list[Finding], stats: dict, branches: list[str]) -> str:
    lines = [f"Scanned {stats['commits']} commit(s) on {len(branches)} branch(es): {', '.join(branches)}",
             "Commit authors (public in every clone): " + "; ".join(sorted(stats["authors"]))]
    if not findings:
        lines.append("\nNo findings.")
        return "\n".join(lines)
    order = ["token", "secret", "file", "name", "user-path", "email", "test-token"]
    real = real_findings(findings)
    lines.append("")
    lines.append(f"{len(real)} finding(s) to look at, {len(findings) - len(real)} harmless test value(s) (listed last):")
    for kind in order:
        group = [f for f in findings if f.kind == kind]
        for f in group[:40]:
            lines.append(f"  [{kind}] {f.where}: {f.line}" + (f"   ({f.note})" if f.note else ""))
        if len(group) > 40:
            lines.append(f"  [{kind}] ... and {len(group) - 40} more")
    return "\n".join(lines)
