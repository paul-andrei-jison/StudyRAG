"""Detect an embedding model change: vectors from different models are incompatible."""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

META_FILENAME = "embed_meta.json"


@dataclass(frozen=True)
class GuardResult:
    ok: bool
    recorded: bool  # True when this call wrote the first record
    message: str


def check_embedding(data_dir: Path, model: str, dim: int) -> GuardResult:
    """Compare model/dim with what the index was built with; record them on first use."""
    meta_path = data_dir / META_FILENAME
    if not meta_path.exists():
        data_dir.mkdir(parents=True, exist_ok=True)
        meta_path.write_text(json.dumps({"model": model, "dim": dim}), encoding="utf-8")
        return GuardResult(True, True, f"Recorded embedding model {model!r} ({dim} dims).")

    try:
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        old_model, old_dim = meta["model"], int(meta["dim"])
    except (ValueError, KeyError, TypeError):
        return GuardResult(False, False, f"{meta_path} is unreadable; delete it only if you rebuild the index.")

    if old_model == model and old_dim == dim:
        return GuardResult(True, False, f"Embedding model unchanged ({model!r}, {dim} dims).")
    return GuardResult(
        False, False,
        f"EMBEDDING MODEL CHANGED: index was built with {old_model!r} ({old_dim} dims), "
        f"now {model!r} ({dim} dims). Existing vectors are incompatible: rebuild the index "
        f"or switch SB_EMBED_MODEL back.",
    )
