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
DEFAULT_READER_MAX_EDGE = 2048  # see docs/reader-image-policy.md
DEFAULT_CHAT_ANSWER_TOKENS = 700


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
    reader_max_edge: int = DEFAULT_READER_MAX_EDGE
    samples_dir: Path | None = None  # image + .txt reference pairs for the evaluation script
    chat_context_tokens: int | None = None   # override when LM Studio cannot report the loaded context size
    chat_max_answer_tokens: int = DEFAULT_CHAT_ANSWER_TOKENS
    chat_expand_queries: bool = True   # search with two rephrasings too (measured to help; see docs/chat.md)


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

    try:
        max_edge = int(get("SB_READER_MAX_EDGE", str(DEFAULT_READER_MAX_EDGE)))
    except ValueError as exc:
        raise ConfigError("SB_READER_MAX_EDGE must be an integer") from exc
    if max_edge < 256:
        raise ConfigError("SB_READER_MAX_EDGE must be at least 256 pixels")

    samples = get("SB_SAMPLES_DIR")

    def positive_int(name: str, default: int | None, minimum: int) -> int | None:
        raw = get(name)
        if raw is None:
            return default
        try:
            value = int(raw)
        except ValueError as exc:
            raise ConfigError(f"{name} must be an integer") from exc
        if value < minimum:
            raise ConfigError(f"{name} must be at least {minimum}")
        return value

    expand_raw = (get("SB_CHAT_EXPAND_QUERIES", "1") or "1").lower()
    if expand_raw not in ("1", "0", "true", "false", "yes", "no", "on", "off"):
        raise ConfigError("SB_CHAT_EXPAND_QUERIES must be 1/0, true/false, yes/no or on/off")
    chat_context = positive_int("SB_CHAT_CONTEXT_TOKENS", None, 1024)
    chat_answer = positive_int("SB_CHAT_MAX_ANSWER_TOKENS", DEFAULT_CHAT_ANSWER_TOKENS, 64)

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
        reader_max_edge=max_edge,
        samples_dir=_path(samples) if samples else None,
        chat_context_tokens=chat_context,
        chat_max_answer_tokens=chat_answer,
        chat_expand_queries=expand_raw in ("1", "true", "yes", "on"),
    )


def validate_dirs(settings: Settings, repo_root: Path = REPO_ROOT) -> None:
    """Startup checks: dirs outside the repo, not nested in each other, creatable."""
    repo_root = repo_root.resolve()
    lib, data = settings.library_dir, settings.data_dir

    checked = [("SB_LIBRARY_DIR", lib), ("SB_DATA_DIR", data)]
    if settings.samples_dir:
        checked.append(("SB_SAMPLES_DIR", settings.samples_dir))
    for label, d in checked:
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
