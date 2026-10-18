"""Keep the Telegram bot token out of every log line.

httpx logs each request's URL at INFO, and a bot's URL contains its token (`/bot<token>/getUpdates`); exceptions from the network
layer can quote it too. `install_log_safety()` therefore (1) quiets the HTTP libraries, and (2) wraps the log-record factory so
every message and every traceback is cleaned before ANY handler (console, file, test capture) sees it.
"""
from __future__ import annotations

import logging
import re
from typing import Iterable

TOKEN_SHAPE = re.compile(r"\b\d{6,}:[A-Za-z0-9_-]{30,}")
REDACTED = "<telegram-token>"
QUIET_LOGGERS = ("httpx", "httpcore", "telegram", "telegram.ext", "hpack")

_secrets: set[str] = set()
_installed = False


def redact(text: str) -> str:
    for secret in _secrets:
        text = text.replace(secret, REDACTED)
    return TOKEN_SHAPE.sub(REDACTED, text)


def install_log_safety(secrets: Iterable[str | None] = ()) -> None:
    """Idempotent. `secrets` are extra strings to hide even if they do not look like a token."""
    global _installed
    _secrets.update(s for s in secrets if s and len(s) >= 8)
    for name in QUIET_LOGGERS:
        logging.getLogger(name).setLevel(logging.WARNING)
    if _installed:
        return
    _installed = True
    previous = logging.getLogRecordFactory()

    def factory(*args, **kwargs) -> logging.LogRecord:
        record = previous(*args, **kwargs)
        try:
            record.msg = redact(record.getMessage())
            record.args = ()
            if record.exc_info and not record.exc_text:
                record.exc_text = redact(logging.Formatter().formatException(record.exc_info))
        except Exception:                      # logging must never break the program
            pass
        return record

    logging.setLogRecordFactory(factory)
