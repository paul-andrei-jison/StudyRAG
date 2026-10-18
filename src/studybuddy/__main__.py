"""`python -m studybuddy`: the web UI, the ingest service and the inbox watcher in one process.

  python -m studybuddy [--no-ocr] [--no-bot]
  python -m studybuddy --check [--deep]     report what is fine and what is not, then exit

The address comes from SB_HOST / SB_PORT (default 127.0.0.1:7860). There is no login, so any other host is warned about.
"""
from __future__ import annotations

import argparse
import os
import sys
from typing import Sequence


def bot_line(plan, settings) -> str:
    """One line for the console: whether the Telegram bot runs and for whom (never the token)."""
    if not plan.enabled:
        return f"  telegram: off ({plan.reason})"
    return f"  telegram: on (long polling; answers only user id {settings.telegram_user_id})"


def run_check_command(deep: bool) -> int:
    from .config import load_settings
    from .errors import StudyBuddyError
    from .health import exit_code, format_results, run_health
    from .llm import LMStudioClient
    from .logsafe import install_log_safety

    try:
        settings = load_settings()
    except StudyBuddyError as exc:
        print(f"[FAIL] settings: {exc}")
        return 1
    install_log_safety([settings.telegram_token])
    client = LMStudioClient(settings, timeout=300.0 if deep else 20.0)
    try:
        results = run_health(settings, client, deep=deep)
    finally:
        client.close()
    print(format_results(results))
    return exit_code(results)


def main(argv: Sequence[str] | None = None) -> int:
    os.environ.setdefault("GRADIO_ANALYTICS_ENABLED", "False")   # before gradio is imported
    parser = argparse.ArgumentParser(prog="python -m studybuddy", description="Run the Study Buddy web app.")
    parser.add_argument("--no-ocr", action="store_true", help="skip the classic OCR cross-check while reading pages")
    parser.add_argument("--no-bot", action="store_true", help="do not run the Telegram bot even if a token is set")
    parser.add_argument("--check", action="store_true", help="report whether LM Studio, the models, folders, disk, Telegram and the index are fine, then exit")
    parser.add_argument("--deep", action="store_true", help="with --check: also ask each model for a tiny answer (slower)")
    args = parser.parse_args(argv)
    if args.check:
        return run_check_command(args.deep)

    from .config import bot_plan, load_settings, network_exposure
    from .errors import StudyBuddyError
    from .logging_setup import setup_logging
    from .logsafe import install_log_safety

    try:
        settings = load_settings()
        install_log_safety([settings.telegram_token])      # the bot token never reaches a log line
        log_file = setup_logging(settings)
        plan = bot_plan(settings, args.no_bot)             # a token without SB_TELEGRAM_USER_ID stops the start here
        warning = network_exposure(settings.host)
        from .ui.runtime import build_runtime

        runtime = build_runtime(settings, use_ocr=not args.no_ocr, with_bot=plan.enabled)
    except StudyBuddyError as exc:
        print(f"[FAIL] {exc}", file=sys.stderr)
        return 1
    if warning:
        bar = "!" * 78
        print(f"\n{bar}\nWARNING: {warning}\n{bar}\n", file=sys.stderr, flush=True)

    from .ui.app import build_app, launch

    try:
        runtime.start()
        print(f"Study Buddy is starting at http://{settings.host}:{settings.port}  (Ctrl+C to stop)", flush=True)
        print(f"  library: {settings.library_dir}\n  data:    {settings.data_dir}\n  log:     {log_file}", flush=True)
        print(bot_line(plan, settings), flush=True)
        launch(build_app(runtime), settings.host, settings.port, prevent_thread_lock=False)
    except OSError as exc:
        print(f"[FAIL] cannot listen on {settings.host}:{settings.port} ({exc}). Is another copy running? "
              f"Change SB_PORT to use another port.", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print("Stopping after the document being read (if any)...", flush=True)
    finally:
        runtime.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
