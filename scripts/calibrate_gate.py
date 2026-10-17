"""Calibrate the "not in your papers" gate against the index named by SB_DATA_DIR / SB_LIBRARY_DIR.

Needs, in SB_SAMPLES_DIR: questions.json (answerable questions about the indexed pages) and unanswerable.json
([{"question": "...", "kind": "far" | "near"}]). Prints every question's signals, the false-refusal and false-answer
rates at the current thresholds and at the best ones found. Results are saved to SB_DATA_DIR/eval/, never the repo.
"""
import json
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from studybuddy.chat.calibrate import POOL, choose, collect, evaluate, format_rates, format_table, load_cases  # noqa: E402
from studybuddy.chat.gate import GateConfig  # noqa: E402
from studybuddy.config import load_settings  # noqa: E402
from studybuddy.errors import StudyBuddyError  # noqa: E402
from studybuddy.index.index import Index  # noqa: E402
from studybuddy.llm import LMStudioClient  # noqa: E402


def main() -> int:
    try:
        settings = load_settings()
        if settings.samples_dir is None or not (settings.samples_dir / "questions.json").is_file():
            print("Need SB_SAMPLES_DIR with questions.json (and unanswerable.json).")
            return 0
        cases = load_cases(settings.samples_dir)
    except (StudyBuddyError, ValueError, KeyError) as exc:
        print(f"[FAIL] {exc}")
        return 1
    client = LMStudioClient(settings)
    try:
        index = Index.open(settings, client)
        if index.store.counts()["chunks"] == 0:
            print("The index is empty: ingest the sample pages first (scripts/ingest.py).")
            return 0
        scored = collect(lambda q: index.search(q, k=POOL, mode="hybrid"), cases)
        index.close()
    except StudyBuddyError as exc:
        print(f"[FAIL] {exc}")
        return 1
    finally:
        client.close()

    current = GateConfig()
    best, best_rates = choose(scored)
    print(format_table(scored, best))
    print()
    print("current defaults :", format_rates(evaluate(scored, current), current).replace("\n", "\n                   "))
    print("best found       :", format_rates(best_rates, best).replace("\n", "\n                   "))
    out_dir = settings.data_dir / "eval"
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"gate-calibration-{datetime.now():%Y%m%d-%H%M%S}.json"
    path.write_text(json.dumps([{"kind": s.case.kind, "question": s.case.question, "top_cosine": s.signals.top_cosine,
                                 "coverage": s.signals.coverage, "max_coverage": s.signals.max_coverage} for s in scored],
                               indent=2), encoding="utf-8")
    print(f"\nSaved: {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
