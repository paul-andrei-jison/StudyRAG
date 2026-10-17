"""Tie the READER prompt to a recorded bake-off.

Small prompt edits swing accuracy a lot (a one-sentence change once took a two-column page from CER 0.06
to 0.42), so the prompt's hash is stored with the bake-off numbers it was validated against. A prompt change
fails tests until `scripts/evaluate_reader.py --record-prompt-baseline` has been run on real samples, and the
commit that updates the baseline must carry a "Bake-off:" line (enforced by scripts/git-hooks/commit-msg).
"""
from __future__ import annotations

import hashlib
import json
from datetime import date
from pathlib import Path

from .transcribe import SYSTEM_PROMPT, USER_PROMPT

BASELINE_PATH = Path(__file__).resolve().parents[3] / "docs" / "reader-prompt-baseline.json"
REQUIRED = ("prompt_sha256", "recorded", "reader_model", "pages", "reader_cer", "reader_wer")


def prompt_hash() -> str:
    return hashlib.sha256(f"{SYSTEM_PROMPT}\n---\n{USER_PROMPT}".encode("utf-8")).hexdigest()


def check_baseline(path: Path = BASELINE_PATH, current_hash: str | None = None) -> list[str]:
    """Return a list of problems; empty means the prompt matches a recorded bake-off."""
    current_hash = current_hash or prompt_hash()
    if not path.is_file():
        return [f"{path.name} is missing: run scripts/evaluate_reader.py --record-prompt-baseline"]
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except ValueError as exc:
        return [f"{path.name} is not valid JSON: {exc}"]
    problems = [f"{path.name} lacks '{key}'" for key in REQUIRED if key not in data]
    if problems:
        return problems
    if not isinstance(data["pages"], int) or data["pages"] < 1:
        problems.append("baseline must come from a bake-off over at least 1 scored page")
    for key in ("reader_cer", "reader_wer"):
        if not isinstance(data[key], (int, float)):
            problems.append(f"baseline '{key}' must be a number")
    if data["prompt_sha256"] != current_hash:
        problems.append(
            "The READER prompt changed since the last recorded bake-off. Run "
            "`python scripts/evaluate_reader.py --compare-raw --record-prompt-baseline` on your samples, "
            "commit the updated docs/reader-prompt-baseline.json with the prompt change, and put a "
            "'Bake-off: ...' line in the commit message."
        )
    return problems


def record_baseline(summary: dict, reader_model: str, samples_dir: str, note: str = "",
                    path: Path = BASELINE_PATH, current_hash: str | None = None) -> dict:
    """Write the baseline for the current prompt from one model's bake-off summary."""
    if summary.get("pages", 0) - summary.get("errors", 0) < 1 or summary.get("reader_cer") is None:
        raise ValueError("cannot record a baseline: no page was scored successfully")
    record = {
        "prompt_sha256": current_hash or prompt_hash(),
        "recorded": date.today().isoformat(),
        "reader_model": reader_model,
        "samples_dir": samples_dir,
        "pages": summary["pages"] - summary["errors"],
        "reader_cer": round(summary["reader_cer"], 4),
        "reader_wer": round(summary["reader_wer"], 4),
        "ocr_cer": None if summary.get("ocr_cer") is None else round(summary["ocr_cer"], 4),
        "agreement": None if summary.get("agreement") is None else round(summary["agreement"], 4),
        "note": note,
    }
    path.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    return record
