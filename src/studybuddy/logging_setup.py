"""Log files with rotation: `SB_DATA_DIR/logs/studybuddy.log`, 1 MB each, the last 5 kept.

What goes in: times, levels, which part spoke, and short messages (job ids, counts, error types, file paths).
What never does: the Telegram bot token (`logsafe.install_log_safety` cleans every record before any handler sees it), photo contents
(nothing logs image bytes), message or caption text. Call `install_log_safety` first, then `setup_logging`.
"""
from __future__ import annotations

import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path

from .config import Settings

LOG_DIRNAME = "logs"
LOG_FILENAME = "studybuddy.log"
MAX_BYTES = 1_000_000
BACKUPS = 5
FORMAT = "%(asctime)s %(levelname)-7s %(name)s: %(message)s"
_TAG = "_studybuddy_handler"


def log_path(settings: Settings) -> Path:
    return settings.data_dir / LOG_DIRNAME / LOG_FILENAME


def setup_logging(settings: Settings, max_bytes: int = MAX_BYTES, backups: int = BACKUPS) -> Path:
    """Idempotent: replaces the handlers it added earlier. Returns the log file's path."""
    path = log_path(settings)
    path.parent.mkdir(parents=True, exist_ok=True)
    root = logging.getLogger()
    for handler in [h for h in root.handlers if getattr(h, _TAG, False)]:
        root.removeHandler(handler)
        handler.close()
    level = getattr(logging, settings.log_level)
    file_handler = RotatingFileHandler(path, maxBytes=max_bytes, backupCount=backups, encoding="utf-8")
    file_handler.setLevel(level)
    console = logging.StreamHandler()          # warnings and errors still show in the window, as before
    console.setLevel(logging.WARNING)
    for handler in (file_handler, console):
        handler.setFormatter(logging.Formatter(FORMAT))
        setattr(handler, _TAG, True)
        root.addHandler(handler)
    root.setLevel(min(level, logging.WARNING))
    return path
