"""Bake-off: CER/WER of READER model(s) and classic OCR on image + .txt pairs in SB_SAMPLES_DIR.

  python scripts/evaluate_reader.py                       # configured SB_READER_MODEL
  python scripts/evaluate_reader.py --reader-model a --reader-model b   # compare models
  python scripts/evaluate_reader.py --compare-raw --limit 5
"""
import argparse
import dataclasses
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from studybuddy.config import load_settings  # noqa: E402
from studybuddy.errors import ConfigError  # noqa: E402
from studybuddy.llm import LMStudioClient  # noqa: E402
from studybuddy.reader.evaluate import (  # noqa: E402
    Report, evaluate, find_pairs, format_report, summarise, write_report,
)
from studybuddy.reader.ocr import RapidOcrEngine  # noqa: E402
from studybuddy.reader.transcribe import Reader  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--reader-model", action="append", help="READER model id (repeatable); default SB_READER_MODEL")
    parser.add_argument("--compare-raw", action="store_true", help="also read the uncleaned image to measure cleanup")
    parser.add_argument("--limit", type=int, help="only evaluate the first N pairs")
    parser.add_argument("--no-ocr", action="store_true", help="skip the classic OCR engine")
    args = parser.parse_args()

    try:
        settings = load_settings()
    except ConfigError as exc:
        print(f"[FAIL] config: {exc}")
        return 1

    found = find_pairs(settings.samples_dir)
    if not found.pairs:
        print(f"No samples to evaluate. {found.note}")
        print("Put page photos and same-name hand-corrected .txt files there (see docs/reader-image-policy.md).")
        return 0
    pairs = found.pairs[: args.limit] if args.limit else found.pairs

    models = args.reader_model or [settings.reader_model]
    readers, clients = {}, []
    for model in models:
        client = LMStudioClient(dataclasses.replace(settings, reader_model=model))
        clients.append(client)
        readers[model] = Reader(client)

    print(f"Evaluating {len(pairs)} page(s) with: {', '.join(models)}")
    try:
        rows = evaluate(pairs, readers, None if args.no_ocr else RapidOcrEngine(),
                        max_edge=settings.reader_max_edge, compare_raw=args.compare_raw)
    finally:
        for c in clients:
            c.close()

    report = Report(rows=rows, summary=summarise(rows), skipped=found)
    print(format_report(report))
    json_path, csv_path = write_report(report, settings.data_dir / "eval")
    print(f"\nSaved: {json_path}\n       {csv_path}")
    return 1 if any(r.error for r in rows) else 0


if __name__ == "__main__":
    sys.exit(main())
