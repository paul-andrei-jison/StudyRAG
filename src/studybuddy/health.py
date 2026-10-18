"""`python -m studybuddy --check`: one screen that says whether everything the app needs is in place.

Quick (default): LM Studio and the three models, folders, disk space, the Telegram setting, the index, the web port. Only reads.
Deep (`--check --deep`): also asks each model for a tiny answer, so a model that is listed but cannot answer shows up.
The Telegram token is never printed: only whether one is set.
"""
from __future__ import annotations

import shutil
import socket
import time
from pathlib import Path
from typing import Callable

from .bot.state import DONE, StateStore
from .checks import FAIL, PASS, WARN, CheckResult, exit_code, run_checks
from .config import Settings, bot_plan, validate_dirs
from .errors import StudyBuddyError
from .llm import LMStudioClient
from .logging_setup import log_path

GB = 1024 ** 3
DISK_WARN_BYTES = 5 * GB
DISK_FAIL_BYTES = 1 * GB


def port_in_use(host: str, port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.settimeout(1.0)
        return s.connect_ex(("127.0.0.1" if host in ("localhost", "") else host, port)) == 0


def _disk(label: str, path: Path, usage: Callable) -> CheckResult:
    probe = path
    while not probe.exists() and probe != probe.parent:
        probe = probe.parent
    try:
        free = usage(probe).free
    except OSError as exc:
        return CheckResult(f"disk space ({label})", WARN, f"cannot read free space: {exc}")
    text = f"{free / GB:.1f} GB free on the drive holding {path}"
    status = FAIL if free < DISK_FAIL_BYTES else WARN if free < DISK_WARN_BYTES else PASS
    return CheckResult(f"disk space ({label})", status, text + ("" if status == PASS else " - free some space; photos and the index keep growing"))


def _models(settings: Settings, client: LMStudioClient) -> list[CheckResult]:
    try:
        available = client.list_models()
    except StudyBuddyError as exc:
        return [CheckResult("LM Studio", FAIL, f"{exc} - open LM Studio and start its local server ({settings.base_url})")]
    results = [CheckResult("LM Studio", PASS, f"reachable at {settings.base_url}, {len(available)} model(s) loaded")]
    for role, model in (("READER", settings.reader_model), ("CHAT", settings.chat_model), ("EMBED", settings.embed_model)):
        if model in available:
            results.append(CheckResult(f"model {role}", PASS, model))
        else:
            results.append(CheckResult(f"model {role}", FAIL, f"{model!r} is not loaded - load it in LM Studio (loaded: {', '.join(available) or 'none'})"))
    return results


def _telegram(settings: Settings) -> CheckResult:
    try:
        plan = bot_plan(settings)
    except StudyBuddyError as exc:
        return CheckResult("telegram", FAIL, str(exc))
    if not plan.enabled:
        return CheckResult("telegram", PASS, f"off: {plan.reason}")
    return CheckResult("telegram", PASS, f"on; token set (not shown); answers only user id {settings.telegram_user_id}")


def _index(settings: Settings) -> list[CheckResult]:
    from .index.index import Index, index_dir

    if not index_dir(settings).exists():
        return [CheckResult("index", PASS, "empty - nothing has been added yet")]
    try:
        index = Index.open(settings, None, check_guard=False)     # status never embeds anything
    except Exception as exc:                                      # a broken index must be reported, not crash the report
        return [CheckResult("index", FAIL, f"cannot open the index: {exc}")]
    try:
        s = index.status()
    except Exception as exc:
        return [CheckResult("index", FAIL, f"cannot read the index: {exc}")]
    finally:
        index.close()
    results = [CheckResult("index", PASS, f"{s.documents} document(s), {s.pages} page(s), {s.chunks} chunk(s), {s.vectors} vector(s)")]
    if not s.consistency.ok:
        results.append(CheckResult("index consistency", FAIL, "search data and text data disagree - run: python scripts/index_admin.py repair"))
    if s.guard_problem:
        results.append(CheckResult("embedding model", FAIL, s.guard_problem))
    elif s.recorded_model:
        results.append(CheckResult("embedding model", PASS, f"index built with {s.recorded_model!r}"))
    return results


def _pending(settings: Settings) -> CheckResult:
    root = settings.data_dir / "bot"
    waiting = [i for i in StateStore(root).all_items() if i.stage != DONE] if root.exists() else []
    if not waiting:
        return CheckResult("bot queue", PASS, "nothing waiting")
    return CheckResult("bot queue", WARN, f"{len(waiting)} file(s) waiting for a subject or for reading")


def _log(settings: Settings, now: float) -> CheckResult:
    path = log_path(settings)
    if not path.exists():
        return CheckResult("log file", PASS, f"none yet ({path})")
    age = now - path.stat().st_mtime
    when = f"{age / 60:.0f} min ago" if age < 3600 else f"{age / 3600:.1f} h ago" if age < 172800 else f"{age / 86400:.0f} days ago"
    return CheckResult("log file", PASS, f"{path}, last written {when}")


def run_health(settings: Settings, client: LMStudioClient, deep: bool = False, *, usage: Callable = shutil.disk_usage,
               port_busy: Callable[[str, int], bool] = port_in_use, now: Callable[[], float] = time.time) -> list[CheckResult]:
    results: list[CheckResult] = []
    try:
        validate_dirs(settings)
        results.append(CheckResult("folders", PASS, f"library {settings.library_dir}; data {settings.data_dir}"))
        results.append(_disk("library", settings.library_dir, usage))
        results.append(_disk("data", settings.data_dir, usage))
    except StudyBuddyError as exc:
        results.append(CheckResult("folders", FAIL, str(exc)))
    models = _models(settings, client)
    results += models
    if deep and all(r.status != FAIL for r in models):
        results += [r for r in run_checks(settings, client) if r.name in ("CHAT", "READER", "EMBED", "embedding guard")]
    results.append(_telegram(settings))
    if settings.data_dir.exists():
        results += _index(settings)
        results.append(_pending(settings))
        results.append(_log(settings, now()))
    busy = port_busy(settings.host, settings.port)
    results.append(CheckResult("web port", WARN if busy else PASS,
                               f"port {settings.port} is in use - Study Buddy may already be running" if busy else f"port {settings.port} is free"))
    return results


def format_results(results: list[CheckResult]) -> str:
    lines = [f"[{r.status}] {r.name}: {r.detail}" for r in results]
    bad = sum(r.status == FAIL for r in results)
    warn = sum(r.status == WARN for r in results)
    lines.append("")
    lines.append(f"{bad} problem(s) to fix, {warn} warning(s)." if bad else
                 f"No problems; {warn} warning(s)." if warn else "Everything needed is in place.")
    return "\n".join(lines)


__all__ = ["run_health", "format_results", "exit_code"]
