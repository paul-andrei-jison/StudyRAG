"""Compare vector retrieval with and without the nomic task prefixes on your own pages.

Needs, in SB_SAMPLES_DIR: proofread image + .txt pairs (the same ones the reader bake-off uses) and a
questions.json (see src/studybuddy/index/prefix_compare.py for the format). Needs EMBED_DOC_PREFIX and
EMBED_QUERY_PREFIX set in .env: those are the "with prefix" values. Results go to SB_DATA_DIR/eval/, never the repo.
"""
import json
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from studybuddy.config import load_settings  # noqa: E402
from studybuddy.errors import StudyBuddyError  # noqa: E402
from studybuddy.index.chunking import chunk_page  # noqa: E402
from studybuddy.index.prefix_compare import (  # noqa: E402
    ChunkText, compare, format_results, load_questions,
)
from studybuddy.llm import LMStudioClient  # noqa: E402
from studybuddy.reader.evaluate import find_pairs  # noqa: E402


def main() -> int:
    try:
        settings = load_settings()
    except StudyBuddyError as exc:
        print(f"[FAIL] {exc}")
        return 1
    if not (settings.embed_doc_prefix or settings.embed_query_prefix):
        print("[FAIL] EMBED_DOC_PREFIX and EMBED_QUERY_PREFIX are both empty: set them in .env to compare.")
        return 1
    found = find_pairs(settings.samples_dir)
    questions_path = (settings.samples_dir / "questions.json") if settings.samples_dir else None
    if not found.pairs or questions_path is None or not questions_path.is_file():
        print(f"Nothing to compare: need proofread pairs and {questions_path}. {found.note}")
        return 0
    try:
        questions = load_questions(questions_path)
    except ValueError as exc:
        print(f"[FAIL] {exc}")
        return 1

    chunks = [ChunkText(image.stem, c.text) for image, ref in found.pairs
              for c in chunk_page(ref.read_text(encoding="utf-8"), 1)]
    client = LMStudioClient(settings)
    try:
        results = compare(chunks, questions, client.embed, settings.embed_doc_prefix, settings.embed_query_prefix)
    except StudyBuddyError as exc:
        print(f"[FAIL] {exc}")
        return 1
    finally:
        client.close()

    print(format_results(results, len(chunks)))
    out_dir = settings.data_dir / "eval"
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"prefix-compare-{datetime.now().strftime('%Y%m%d-%H%M%S')}.json"
    path.write_text(json.dumps([{"variant": r.name, "doc_prefix": r.doc_prefix, "query_prefix": r.query_prefix,
                                 "hit@1": r.hit_at(1), "hit@3": r.hit_at(3), "mrr": r.mrr, "margin": r.mean_margin,
                                 "ranks": [q.rank for q in r.results]} for r in results], indent=2), encoding="utf-8")
    print(f"\nSaved: {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
