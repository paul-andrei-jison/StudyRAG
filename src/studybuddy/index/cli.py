"""Command-line front end for index maintenance: `scripts/index_admin.py`.

The logic lives here (not in scripts/) so tests can inject settings, a fake embedder and an output buffer.
"""
from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path
from typing import TextIO

from ..config import Settings, load_settings
from ..errors import StudyBuddyError
from ..llm import LMStudioClient
from ..reader.quality import PageRecord
from .index import Embedder, Index, index_dir, reset_index
from .store import DocumentMeta

TEXT_SUFFIXES = {".txt", ".md"}


def _setup(settings: Settings | None, embedder: Embedder | None) -> tuple[Settings, Embedder, LMStudioClient | None]:
    settings = settings or load_settings()
    client = None
    if embedder is None:
        client = embedder = LMStudioClient(settings)
    return settings, embedder, client


# ---- index_admin ---------------------------------------------------------------------------

def admin_main(argv: list[str], settings: Settings | None = None, embedder: Embedder | None = None,
               out: TextIO | None = None) -> int:
    out = out or sys.stdout
    parser = argparse.ArgumentParser(prog="index_admin.py", description="Inspect and maintain the search index.")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("status", help="counts, recorded vs configured embedding model, consistency")
    sub.add_parser("check", help="find vectors without chunks and chunks without vectors")
    sub.add_parser("repair", help="delete orphan vectors and embed chunks that lack one")
    sub.add_parser("reindex", help="re-embed every chunk with the CURRENT model and document prefix")
    reset = sub.add_parser("reset", help="delete the whole index (never the library folder)")
    reset.add_argument("--yes", action="store_true", help="really delete")
    add = sub.add_parser("add-text", help="index a .txt/.md file or folder as documents (one page each)")
    add.add_argument("path", type=Path)
    add.add_argument("--subject", required=True)
    add.add_argument("--title", help="title for a single file (default: the file name)")
    args = parser.parse_args(argv)

    try:
        settings, embedder, client = _setup(settings, embedder)
    except StudyBuddyError as exc:
        print(f"[FAIL] {exc}", file=out)
        return 1
    try:
        return _run_admin(args, settings, embedder, out)
    except StudyBuddyError as exc:
        print(f"[FAIL] {exc}", file=out)
        return 1
    finally:
        if client is not None:
            client.close()


def _run_admin(args: argparse.Namespace, settings: Settings, embedder: Embedder, out: TextIO) -> int:
    if args.command == "reset":
        targets = [index_dir(settings), settings.data_dir / "embed_meta.json"]
        existing = [t for t in targets if t.exists()]
        if not args.yes:
            print("Would delete (nothing was changed; add --yes to proceed):", file=out)
            for t in existing or ["(nothing to delete)"]:
                print(f"  {t}", file=out)
            print(f"The library folder {settings.library_dir} and your original photos are never touched.", file=out)
            return 2
        removed = reset_index(settings)
        for t in removed:
            print(f"Deleted {t}", file=out)
        print("Index reset. Re-ingest your documents to rebuild it.", file=out)
        return 0

    guarded = args.command == "add-text"
    index = Index.open(settings, embedder, check_guard=guarded)
    try:
        if args.command == "status":
            return _status(index, out)
        if args.command == "check":
            found = index.check()
            _print_consistency(found, out)
            return 0 if found.ok else 1
        if args.command == "repair":
            found = index.repair()
            _print_consistency(found, out, fixed=True)
            return 0
        if args.command == "reindex":
            def progress(done: int, total: int) -> None:
                print(f"  embedded {done}/{total} chunks", file=out)
            n = index.reindex(progress=progress)
            print(f"Re-indexed {n} chunks with {settings.embed_model!r} (document prefix {settings.embed_doc_prefix!r}).",
                  file=out)
            return 0
        if args.command == "add-text":
            return _add_text(index, args, out)
    finally:
        index.close()
    return 1


def _status(index: Index, out: TextIO) -> int:
    s = index.status()
    print(f"Documents {s.documents}, pages {s.pages}, chunks {s.chunks}, vectors {s.vectors} "
          f"(dimension {s.dim if s.dim is not None else 'n/a'})", file=out)
    print(f"Embedding model recorded: {s.recorded_model!r}  configured: {s.configured_model!r}", file=out)
    print(f"Document prefix recorded: {s.recorded_doc_prefix!r}  configured: {s.configured_doc_prefix!r}", file=out)
    _print_consistency(s.consistency, out)
    if s.guard_problem:
        print(f"[PROBLEM] {s.guard_problem}", file=out)
    return 0 if (s.consistency.ok and not s.guard_problem) else 1


def _print_consistency(found, out: TextIO, fixed: bool = False) -> None:
    if found.ok:
        print("Consistent: every chunk has exactly one vector.", file=out)
        return
    verb = "Fixed" if fixed else "Found"
    if found.missing_vectors:
        print(f"{verb} {len(found.missing_vectors)} chunk(s) without a vector.", file=out)
    if found.orphan_vectors:
        print(f"{verb} {len(found.orphan_vectors)} vector(s) without a chunk.", file=out)
    if not fixed:
        print("Run: python scripts/index_admin.py repair", file=out)


def _add_text(index: Index, args: argparse.Namespace, out: TextIO) -> int:
    path: Path = args.path
    files = sorted(p for p in path.rglob("*") if p.suffix.lower() in TEXT_SUFFIXES) if path.is_dir() else [path]
    if not files or not all(p.exists() for p in files):
        print(f"[FAIL] no .txt/.md files found at {path}", file=out)
        return 1
    failed = 0
    for file in files:
        try:
            raw = file.read_bytes()
            text = raw.decode("utf-8")
        except (OSError, UnicodeDecodeError) as exc:
            print(f"[FAIL] {file.name}: {exc}", file=out)
            failed += 1
            continue
        meta = DocumentMeta(hashlib.sha256(raw).hexdigest(), str(file.resolve()), "text-file", args.subject,
                            args.title if (args.title and len(files) == 1) else file.stem)
        record = PageRecord(1, str(file.resolve()), "text-file", text=text, quality=1.0)
        result = index.add_document(meta, [record], expected_pages=1)
        state = f"indexed, {result.chunk_count} chunk(s)" if result.created else "already indexed, skipped"
        print(f"{file.name}: {state}", file=out)
    return 1 if failed else 0
