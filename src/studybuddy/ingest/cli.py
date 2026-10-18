"""Command-line front end for ingest: `scripts/ingest.py`.

  ingest.py run [--until-idle]      watch the inbox and process documents (Ctrl+C to stop)
  ingest.py once FILE... --subject S [--album] [--keep]
  ingest.py status                  queue, deferred, failed, duplicates, subjects
  ingest.py retry-failed            put permanently failed documents back in the queue

Exit code: 0 everything indexed (or a duplicate), 1 something failed, 3 something is waiting for a retry.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Callable, TextIO

from ..config import Settings, load_settings, validate_dirs
from ..errors import StudyBuddyError
from ..index.index import Index
from ..llm import LMStudioClient
from ..reader.ocr import RapidOcrEngine
from ..reader.transcribe import Reader
from .results import DEFERRED, DUPLICATE, FAILED, INDEXED, IngestResult
from .service import IngestService
from .subjects import SubjectList
from .watcher import InboxWatcher

ServiceFactory = Callable[[Settings], tuple[IngestService, Callable[[], None]]]


def format_result(r: IngestResult) -> str:
    names = ", ".join(Path(f).name for f in r.files) or r.job_id
    if r.status == INDEXED:
        head = (f"INDEXED   {r.subject} | {r.title} | {r.page_count} page{'s' if r.page_count != 1 else ''}"
                f" | subject {r.subject_source}")
        lines = [head, f"  saved: {r.final_path}"]
        if r.pdf_path:
            lines.append(f"  pdf:   {r.pdf_path}")
        lines.append(f"  summary: {r.summary}")
        for page in r.flagged_pages:
            mark = " (resend as a file?)" if page.needs_attention else ""
            lines.append(f"  page {page.page_number}: {'; '.join(page.reasons)}{mark}")
    elif r.status == DUPLICATE:
        e = r.existing or {}
        lines = [f"DUPLICATE {names}", f"  already indexed as '{e.get('title', r.title)}' ({e.get('subject', r.subject)})",
                 f"  existing: {e.get('file_path', r.final_path)}"]
        if r.duplicate_dir:
            lines.append(f"  the new copy was kept in: {r.duplicate_dir}")
    elif r.status == DEFERRED:
        lines = [f"DEFERRED  {names}", f"  {r.error}", f"  will retry in about {int(r.retry_in_seconds or 0)} s (attempt {r.attempts})"]
    else:
        lines = [f"FAILED    {names}", f"  {r.error}"]
        if r.failed_dir:
            lines.append(f"  files kept in: {r.failed_dir}")
    lines += [f"  note: {w}" for w in r.warnings]
    return "\n".join(lines)


def default_factory(settings: Settings, use_ocr: bool = True) -> tuple[IngestService, Callable[[], None]]:
    validate_dirs(settings)
    client = LMStudioClient(settings)
    index = Index.open(settings, client)
    subjects = SubjectList.from_settings(settings)
    service = IngestService(settings, index, Reader(client), RapidOcrEngine() if use_ocr else None, client, subjects)

    def cleanup() -> None:
        service.stop()
        index.close()
        client.close()

    return service, cleanup


def _exit_code(results: list[IngestResult]) -> int:
    if any(r.status == FAILED for r in results):
        return 1
    if any(r.status == DEFERRED for r in results):
        return 3
    return 0


def ingest_main(argv: list[str], settings: Settings | None = None, factory: ServiceFactory | None = None,
                out: TextIO | None = None) -> int:
    out = out or sys.stdout
    parser = argparse.ArgumentParser(prog="ingest.py", description="Ingest documents into the study index.")
    sub = parser.add_subparsers(dest="command", required=True)
    run = sub.add_parser("run", help="watch the inbox and process documents")
    run.add_argument("--until-idle", action="store_true", help="process what is there, then exit")
    run.add_argument("--no-ocr", action="store_true")
    once = sub.add_parser("once", help="ingest specific files and wait for the result")
    once.add_argument("files", nargs="+", type=Path)
    once.add_argument("--subject", help="subject (default: auto-detect from the subject list)")
    once.add_argument("--album", action="store_true", help="several photos that form one document")
    once.add_argument("--keep", action="store_true", help="copy into the library but leave the originals where they are")
    once.add_argument("--no-ocr", action="store_true")
    sub.add_parser("status", help="show the queue, failures, duplicates and subjects")
    retry = sub.add_parser("retry-failed", help="requeue documents from failed/")
    retry.add_argument("--no-ocr", action="store_true")
    args = parser.parse_args(argv)

    try:
        settings = settings or load_settings()
    except StudyBuddyError as exc:
        print(f"[FAIL] {exc}", file=out)
        return 1
    if args.command == "status":
        return _status(settings, out)

    try:
        if factory is not None:
            service, cleanup = factory(settings)
        else:
            service, cleanup = default_factory(settings, use_ocr=not getattr(args, "no_ocr", False))
    except StudyBuddyError as exc:
        print(f"[FAIL] {exc}", file=out)
        return 1

    results: list[IngestResult] = []

    def show(result: IngestResult) -> None:
        results.append(result)
        print(format_result(result), file=out, flush=True)

    service.add_callback(show)
    try:
        service.start()
        if args.command == "once":
            return _once(service, args, results, out)
        if args.command == "retry-failed":
            tickets = service.retry_failed()
            print(f"Requeued {len(tickets)} failed document(s).", file=out)
            service.wait_idle(timeout=3600)
            return _exit_code(results)
        return _run(service, args.until_idle, results, out)
    except KeyboardInterrupt:
        print("Stopping after the current document...", file=out)
        return _exit_code(results)
    except StudyBuddyError as exc:
        print(f"[FAIL] {exc}", file=out)
        return 1
    finally:
        cleanup()


def _once(service: IngestService, args: argparse.Namespace, results: list[IngestResult], out: TextIO) -> int:
    missing = [str(p) for p in args.files if not p.exists()]
    if missing:
        print(f"[FAIL] not found: {', '.join(missing)}", file=out)
        return 1
    # several files are one document only with --album; otherwise each file is its own document (its own job)
    groups = [args.files] if args.album or len(args.files) == 1 else [[f] for f in args.files]
    waiting = False
    for files in groups:
        ticket = service.submit(files, subject=args.subject, is_album=args.album, cleanup_sources=not args.keep, origin="cli")
        ticket.reported.result()  # the callback prints it; a deferred job is reported here and not waited for
        waiting = waiting or not ticket.final.done()
    if waiting:
        print("LM Studio is not available; the document stays queued and is retried when the service runs "
              "(ingest.py run).", file=out)
    return _exit_code(results)


def _run(service: IngestService, until_idle: bool, results: list[IngestResult], out: TextIO) -> int:
    watcher = InboxWatcher(service).start()
    print(f"Watching {service.inbox_dir} (Ctrl+C to stop)", file=out, flush=True)
    try:
        idle_since_scan = None
        while True:
            time.sleep(0.2)
            if not until_idle:
                continue
            watcher.poke()  # look at the inbox often instead of waiting for the periodic rescan
            quiet = watcher.scans >= 1 and not watcher.scanner.unsettled and service.wait_idle(timeout=0.05)
            if not quiet:
                idle_since_scan = None
            elif idle_since_scan is None:
                idle_since_scan = watcher.scans
            elif watcher.scans > idle_since_scan:  # a whole scan ran while idle and found nothing new
                break
    finally:
        watcher.stop()
    deferred = service.jobs.counts()["deferred"]
    if deferred:
        print(f"{deferred} document(s) are waiting to be retried.", file=out)
    return _exit_code(results)


def _status(settings: Settings, out: TextIO) -> int:
    from .jobs import JobStore

    counts = JobStore(settings.data_dir / "queue").counts()
    print(f"Queue: {counts['queued']} waiting, {counts['deferred']} deferred (retrying), {counts['running']} running", file=out)
    failed_root = settings.data_dir / "failed"
    failed = sorted(failed_root.glob("*/reason.json")) if failed_root.exists() else []
    print(f"Failed: {len(failed)}", file=out)
    for path in failed:
        try:
            info = json.loads(path.read_text(encoding="utf-8"))
            names = ", ".join(Path(f).name for f in info.get("files", []))
            print(f"  {path.parent.name}: {names} - {info.get('error')}", file=out)
        except (OSError, ValueError):
            print(f"  {path.parent.name}: (unreadable reason.json)", file=out)
    dupes = settings.data_dir / "duplicates"
    print(f"Duplicates kept: {len(list(dupes.iterdir())) if dupes.exists() else 0}", file=out)
    inbox = settings.data_dir / "inbox"
    waiting = [p for p in inbox.rglob("*") if p.is_file()] if inbox.exists() else []
    print(f"Inbox: {len(waiting)} file(s) waiting", file=out)
    try:
        print(f"Subjects: {', '.join(SubjectList.from_settings(settings).names())}", file=out)
    except StudyBuddyError as exc:
        print(f"Subjects: [problem] {exc}", file=out)
    return 0
