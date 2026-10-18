"""Command-line chat: `scripts/chat.py`.

  chat.py [--subject Science] [--ask "one question"] [--no-stream] [--show-scores]

In the chat:  /subject [name|none]   /sources   /new   /help   /quit
After every answer the sources are printed with their file paths.
"""
from __future__ import annotations

import argparse
import sys
from typing import Callable, TextIO

from ..config import Settings, load_settings, validate_dirs
from ..errors import ChatContextError, StudyBuddyError, TransientLLMError
from ..index.index import Index
from ..ingest.subjects import SubjectError, SubjectList
from ..llm import LMStudioClient
from .answers import NEEDS_CHOICE, ChatAnswer, NOT_IN_PAPERS_STATUS
from .session import ChatConfig, ChatSession

HELP = """Commands:
  /subject            show the subject filter and the known subjects
  /subject Science    only answer from that subject
  /subject none       answer from everything again
  /sources            show the sources of the last answer again
  /new                forget the conversation (the subject filter stays)
  /help               this help
  /quit               leave"""

SessionFactory = Callable[[Settings, argparse.Namespace], tuple[ChatSession, Callable[[], None]]]


def default_factory(settings: Settings, args: argparse.Namespace) -> tuple[ChatSession, Callable[[], None]]:
    validate_dirs(settings)
    client = LMStudioClient(settings)
    index = Index.open(settings, client)
    subjects = SubjectList.from_settings(settings)
    config = ChatConfig.from_settings(settings)
    if args.k:
        config.source_count = args.k
    if args.no_expand:
        config.expand_queries = False
    session = ChatSession(index, client, settings, subjects, config)

    def cleanup() -> None:
        index.close()
        client.close()

    return session, cleanup


def format_sources(answer: ChatAnswer, show_scores: bool = False) -> str:
    if not answer.sources:
        return "Sources: none found."
    refused = answer.status == NOT_IN_PAPERS_STATUS
    lines = ["Closest pages (probably not relevant):" if refused else "Sources:"]
    for s in answer.sources:
        mark = " *cited*" if s.cited else ""
        lines.append(f"  [{s.number}]{mark} {s.subject or 'no subject'} | {s.title or '(untitled)'} | {s.file_path}, page {s.page_number}")
        detail = [f"image: {s.image_path}"]
        if show_scores:
            if s.vector_score is not None:
                detail.append(f"cosine {s.vector_score:.3f}")
            if s.keyword_score is not None:
                detail.append(f"bm25 {s.keyword_score:.2f}")
        if s.hard_to_read:
            detail.append("this page was hard to read: check the photo")
        lines.append("      " + "   ".join(detail))
    return "\n".join(lines)


def _show(answer: ChatAnswer, out: TextIO, show_scores: bool, streamed: bool) -> None:
    if not streamed:
        print(answer.text, file=out)
    else:
        print(file=out)
    for warning in answer.warnings:
        print(f"[note] {warning}", file=out)
    if answer.rewritten:
        print(f"[searched for] {answer.query}", file=out)
    if answer.status == NOT_IN_PAPERS_STATUS and answer.refusal_reason:
        print(f"[why] {answer.refusal_reason}", file=out)
    if answer.status != NEEDS_CHOICE:
        print(format_sources(answer, show_scores), file=out)


def _ask(session: ChatSession, question: str, out: TextIO, stream: bool, show_scores: bool) -> bool:
    """Ask one question and print everything. Returns False if it failed."""
    try:
        if stream:
            final = session.ask(question, on_token=lambda piece: print(piece, end="", file=out, flush=True),
                                on_progress=lambda note: print(f"[{note}]", file=out, flush=True))
        else:
            final = session.ask(question, on_progress=lambda note: print(f"[{note}]", file=out, flush=True))
    except ChatContextError as exc:
        print(f"\n[FAIL] {exc}", file=out)
        return False
    except TransientLLMError as exc:
        print(f"\n[FAIL] LM Studio is not available right now: {exc}", file=out)
        return False
    except StudyBuddyError as exc:
        print(f"\n[FAIL] {exc}", file=out)
        return False
    _show(final, out, show_scores, streamed=stream)
    return True


def _command(line: str, session: ChatSession, subjects: SubjectList | None, out: TextIO, show_scores: bool) -> bool:
    """Handle a /command. Returns False when the chat should end."""
    word, _, rest = line.strip().partition(" ")
    rest = rest.strip()
    if word in ("/quit", "/exit", "/q"):
        return False
    if word == "/help":
        print(HELP, file=out)
    elif word == "/new":
        session.new()
        print("Started a new conversation.", file=out)
    elif word == "/sources":
        if session.last_answer is None:
            print("No answer yet.", file=out)
        else:
            print(format_sources(session.last_answer, show_scores), file=out)
    elif word == "/subject":
        if not rest:
            print(f"Subject filter: {session.subject or 'none (all subjects)'}", file=out)
            if subjects is not None:
                print(f"Known subjects: {', '.join(subjects.names())}", file=out)
        elif rest.lower() in ("none", "all", "-"):
            session.set_subject(None)
            print("Answering from all subjects.", file=out)
        else:
            try:
                print(f"Only answering from {session.set_subject(rest)}.", file=out)
            except SubjectError as exc:
                print(f"[FAIL] {exc}", file=out)
    else:
        print(f"Unknown command {word!r}. Type /help.", file=out)
    return True


def chat_main(argv: list[str], settings: Settings | None = None, factory: SessionFactory | None = None,
              input_fn: Callable[[str], str] = input, out: TextIO | None = None) -> int:
    out = out or sys.stdout
    parser = argparse.ArgumentParser(prog="chat.py", description="Ask questions about your own papers.")
    parser.add_argument("--subject", help="only answer from this subject")
    parser.add_argument("--k", type=int, help="number of sources per answer (default 5)")
    parser.add_argument("--ask", help="ask one question and exit")
    parser.add_argument("--no-stream", action="store_true", help="print the answer when it is complete")
    parser.add_argument("--no-expand", action="store_true", help="do not search with rephrasings of the question")
    parser.add_argument("--show-scores", action="store_true", help="show cosine and bm25 scores in the sources")
    args = parser.parse_args(argv)

    try:
        settings = settings or load_settings()
        session, cleanup = (factory or default_factory)(settings, args)
    except StudyBuddyError as exc:
        print(f"[FAIL] {exc}", file=out)
        return 1
    try:
        if args.subject:
            try:
                session.set_subject(args.subject)
            except SubjectError as exc:
                print(f"[FAIL] {exc}", file=out)
                return 1
        stream = not args.no_stream
        if args.ask:
            return 0 if _ask(session, args.ask, out, stream, args.show_scores) else 1
        print("Ask a question about your papers. /help for commands, /quit to leave.", file=out)
        if session.subject:
            print(f"Subject filter: {session.subject}", file=out)
        while True:
            try:
                line = input_fn("\nYou> ").strip()
            except (EOFError, KeyboardInterrupt):
                print(file=out)
                break
            if not line:
                continue
            if line.startswith("/"):
                if not _command(line, session, session.subjects, out, args.show_scores):
                    break
                continue
            print("Study Buddy> ", end="", file=out, flush=True)
            _ask(session, line, out, stream, args.show_scores)
        return 0
    finally:
        cleanup()
