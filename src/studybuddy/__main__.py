"""`python -m studybuddy`: the web UI, the ingest service and the inbox watcher in one process.

  python -m studybuddy [--no-ocr]

The address comes from SB_HOST / SB_PORT (default 127.0.0.1:7860). There is no login, so any other host is warned about.
"""
from __future__ import annotations

import argparse
import os
import sys
from typing import Sequence


def main(argv: Sequence[str] | None = None) -> int:
    os.environ.setdefault("GRADIO_ANALYTICS_ENABLED", "False")   # before gradio is imported
    parser = argparse.ArgumentParser(prog="python -m studybuddy", description="Run the Study Buddy web app.")
    parser.add_argument("--no-ocr", action="store_true", help="skip the classic OCR cross-check while reading pages")
    parser.add_argument("--no-bot", action="store_true", help="do not run the Telegram bot even if a token is set")
    args = parser.parse_args(argv)

    from .config import bot_plan, load_settings, network_exposure
    from .errors import StudyBuddyError
    from .logsafe import install_log_safety

    try:
        settings = load_settings()
        install_log_safety([settings.telegram_token])      # the bot token never reaches a log line
        bot_plan(settings, args.no_bot)                    # a token without SB_TELEGRAM_USER_ID stops the start here
        warning = network_exposure(settings.host)
        from .ui.runtime import build_runtime

        runtime = build_runtime(settings, use_ocr=not args.no_ocr)
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
        print(f"  library: {settings.library_dir}\n  data:    {settings.data_dir}", flush=True)
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
