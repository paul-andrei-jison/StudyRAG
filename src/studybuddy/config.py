"""Settings loaded from environment / .env. The only place model names come from."""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

from dotenv import load_dotenv

from .errors import ConfigError

REPO_ROOT = Path(__file__).resolve().parents[2]

DEFAULT_BASE_URL = "http://localhost:1234/v1"
DEFAULT_TAG_TIMEOUT = 300


@dataclass(frozen=True)
class Settings:
    base_url: str
    api_key: str
    reader_model: str
    chat_model: str
    embed_model: str
    library_dir: Path
    data_dir: Path
    subjects_file: Path
    tag_timeout_seconds: int
    telegram_token: str | None
    telegram_user_id: int | None
    # Task prefixes some embedding models need (e.g. nomic: "search_document: "/"search_query: ").
    embed_doc_prefix: str = ""
    embed_query_prefix: str = ""


def _path(value: str) -> Path:
    return Path(value).expanduser().resolve()


def _is_within(child: Path, parent: Path) -> bool:
    return child == parent or parent in child.parents


def load_settings(env: Mapping[str, str] | None = None, repo_root: Path = REPO_ROOT) -> Settings:
    """Build Settings. With env=None, reads the repo's .env and os.environ."""
    if env is None:
        load_dotenv(repo_root / ".env")
        env = os.environ

    def get(name: str, default: str | None = None) -> str | None:
        value = (env.get(name) or "").strip()
        return value or default

    missing = [
        n
        for n in ("SB_READER_MODEL", "SB_CHAT_MODEL", "SB_EMBED_MODEL", "SB_LIBRARY_DIR", "SB_DATA_DIR")
        if not get(n)
    ]
    if missing:
        raise ConfigError(f"Missing required settings: {', '.join(missing)} (see .env.example)")

    try:
        timeout = int(get("SB_TAG_TIMEOUT_SECONDS", str(DEFAULT_TAG_TIMEOUT)))
    except ValueError as exc:
        raise ConfigError("SB_TAG_TIMEOUT_SECONDS must be an integer") from exc

    user_id = get("TELEGRAM_USER_ID")
    try:
        telegram_user_id = int(user_id) if user_id else None
    except ValueError as exc:
        raise ConfigError("TELEGRAM_USER_ID must be a number") from exc

    subjects = Path(get("SB_SUBJECTS_FILE", "subjects.json")).expanduser()
    if not subjects.is_absolute():
        subjects = repo_root / subjects

    return Settings(
        base_url=get("SB_LMSTUDIO_BASE_URL", DEFAULT_BASE_URL).rstrip("/"),
        api_key=get("SB_LMSTUDIO_API_KEY", "lm-studio"),
        reader_model=get("SB_READER_MODEL"),
        chat_model=get("SB_CHAT_MODEL"),
        embed_model=get("SB_EMBED_MODEL"),
        library_dir=_path(get("SB_LIBRARY_DIR")),
        data_dir=_path(get("SB_DATA_DIR")),
        subjects_file=subjects,
        tag_timeout_seconds=timeout,
        telegram_token=get("TELEGRAM_BOT_TOKEN"),
        telegram_user_id=telegram_user_id,
        embed_doc_prefix=env.get("EMBED_DOC_PREFIX", ""),
        embed_query_prefix=env.get("EMBED_QUERY_PREFIX", ""),
    )


def validate_dirs(settings: Settings, repo_root: Path = REPO_ROOT) -> None:
    """Startup checks: dirs outside the repo, not nested in each other, creatable."""
    repo_root = repo_root.resolve()
    lib, data = settings.library_dir, settings.data_dir

    for label, d in (("SB_LIBRARY_DIR", lib), ("SB_DATA_DIR", data)):
        if _is_within(d, repo_root) or _is_within(repo_root, d):
            raise ConfigError(f"{label} ({d}) must be outside the repo ({repo_root})")
    if _is_within(lib, data) or _is_within(data, lib):
        raise ConfigError(f"SB_LIBRARY_DIR ({lib}) and SB_DATA_DIR ({data}) must not be nested")

    for label, d in (("SB_LIBRARY_DIR", lib), ("SB_DATA_DIR", data)):
        try:
            d.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            raise ConfigError(f"Cannot create {label} ({d}): {exc}") from exc
        if not d.is_dir():
            raise ConfigError(f"{label} ({d}) is not a directory")
