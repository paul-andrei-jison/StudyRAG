"""Detect an embedding model change: vectors from different models are incompatible."""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

META_FILENAME = "embed_meta.json"


@dataclass(frozen=True)
class GuardResult:
    ok: bool
    recorded: bool  # True when this call wrote (or completed) the record
    message: str


def read_meta(data_dir: Path) -> dict | None:
    """The recorded {model, dim[, doc_prefix]}, or None when nothing is recorded yet."""
    meta_path = data_dir / META_FILENAME
    if not meta_path.exists():
        return None
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    return {"model": meta["model"], "dim": int(meta["dim"]), **({"doc_prefix": meta["doc_prefix"]} if "doc_prefix" in meta else {})}


def write_meta(data_dir: Path, model: str, dim: int, doc_prefix: str | None = None) -> None:
    data_dir.mkdir(parents=True, exist_ok=True)
    record: dict = {"model": model, "dim": dim}
    if doc_prefix is not None:
        record["doc_prefix"] = doc_prefix
    (data_dir / META_FILENAME).write_text(json.dumps(record), encoding="utf-8")


def check_embedding(data_dir: Path, model: str, dim: int, doc_prefix: str | None = None) -> GuardResult:
    """Compare model/dim (and the document prefix, when given) with what the index was built with.

    The document prefix changes the stored vectors, so a different one needs a re-index; the query
    prefix does not, which is why it is not recorded. Records everything on first use.
    """
    meta_path = data_dir / META_FILENAME
    if not meta_path.exists():
        write_meta(data_dir, model, dim, doc_prefix)
        return GuardResult(True, True, f"Recorded embedding model {model!r} ({dim} dims).")

    try:
        meta = read_meta(data_dir)
        assert meta is not None
    except (ValueError, KeyError, TypeError):
        return GuardResult(False, False, f"{meta_path} is unreadable; delete it only if you rebuild the index.")

    if meta["model"] != model or meta["dim"] != dim:
        return GuardResult(
            False, False,
            f"EMBEDDING MODEL CHANGED: index was built with {meta['model']!r} ({meta['dim']} dims), "
            f"now {model!r} ({dim} dims). Existing vectors are incompatible: rebuild the index "
            f"or switch SB_EMBED_MODEL back.",
        )
    if doc_prefix is not None:
        if "doc_prefix" not in meta:  # record written before prefixes were tracked: complete it
            write_meta(data_dir, model, dim, doc_prefix)
            return GuardResult(True, True, f"Recorded document prefix {doc_prefix!r}.")
        if meta["doc_prefix"] != doc_prefix:
            return GuardResult(
                False, False,
                f"DOCUMENT PREFIX CHANGED: index was built with {meta['doc_prefix']!r}, now {doc_prefix!r} "
                f"(EMBED_DOC_PREFIX). Stored vectors no longer match: re-index or restore the old prefix.",
            )
    return GuardResult(True, False, f"Embedding model unchanged ({model!r}, {dim} dims).")
