"""Does query expansion help? Compare plain retrieval with question + two rephrasings on your own pages.

Needs the index named by SB_DATA_DIR / SB_LIBRARY_DIR and, in SB_SAMPLES_DIR, questions.json (answerable) and
unanswerable.json. Reports, for answerable questions, where the page that holds the answer ranks (in the whole
retrieved list and among the 5 sources chat would use), and for unanswerable ones whether the gate would let them
through. Expansion is only worth switching on if the first improves and the second does not get worse.
"""
import hashlib
import json
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from studybuddy.chat.calibrate import POOL, load_cases  # noqa: E402
from studybuddy.chat.expand import expand_query, fuse_searches  # noqa: E402
from studybuddy.chat.gate import GateConfig, assess  # noqa: E402
from studybuddy.chat.select import spread  # noqa: E402
from studybuddy.config import load_settings  # noqa: E402
from studybuddy.errors import StudyBuddyError  # noqa: E402
from studybuddy.index.index import Index  # noqa: E402
from studybuddy.index.prefix_compare import load_questions  # noqa: E402
from studybuddy.llm import LMStudioClient  # noqa: E402


def main() -> int:
    try:
        settings = load_settings()
        if settings.samples_dir is None or not (settings.samples_dir / "questions.json").is_file():
            print("Need SB_SAMPLES_DIR with questions.json (and unanswerable.json).")
            return 0
        cases = load_cases(settings.samples_dir)
        expectations = {q.question: q for q in load_questions(settings.samples_dir / "questions.json")}
    except (StudyBuddyError, ValueError, KeyError) as exc:
        print(f"[FAIL] {exc}")
        return 1
    stem_of = {hashlib.sha256(p.read_bytes()).hexdigest(): p.stem for p in settings.samples_dir.glob("*.jpg")}
    client = LMStudioClient(settings)
    try:
        index = Index.open(settings, client)

        def correct(hit, question) -> bool:
            q = expectations[question]
            stem = stem_of.get(hashlib.sha256(Path(hit.image_path).read_bytes()).hexdigest())
            return stem in q.expect and (not q.expect_text or q.expect_text.casefold() in hit.text.casefold())

        def first_rank(hits, question):
            return next((i for i, h in enumerate(hits, 1) if correct(h, question)), None)

        rows, gate = [], GateConfig()
        for case in cases:
            plain = index.search(case.question, k=POOL, mode="hybrid")
            phrasings = expand_query(client, case.question)
            fused = fuse_searches(index, [case.question, *phrasings], POOL) if phrasings else plain
            row = {"question": case.question, "kind": case.kind, "phrasings": phrasings}
            if case.answerable:
                row["rank_plain"], row["rank_fused"] = first_rank(plain, case.question), first_rank(fused, case.question)
                row["in_sources_plain"] = first_rank(spread(plain), case.question) is not None
                row["in_sources_fused"] = first_rank(spread(fused), case.question) is not None
            else:
                row["gate_plain"] = assess(spread(plain), case.question, gate).answerable
                row["gate_fused"] = assess(spread(fused), case.question, gate).answerable
            rows.append(row)
        index.close()
    except StudyBuddyError as exc:
        print(f"[FAIL] {exc}")
        return 1
    finally:
        client.close()

    answerable = [r for r in rows if r["kind"] == "answerable"]
    unanswerable = [r for r in rows if r["kind"] != "answerable"]

    def stats(key):
        ranks = [r[key] for r in answerable]
        n = len(ranks)
        return (sum(1 for x in ranks if x == 1) / n, sum(1 for x in ranks if x and x <= 3) / n,
                sum(1 for x in ranks if x and x <= 5) / n, sum(1 / x for x in ranks if x) / n)

    print(f"{len(answerable)} answerable questions: rank of the page holding the answer")
    for label, key in (("plain", "rank_plain"), ("expanded", "rank_fused")):
        h1, h3, h5, mrr = stats(key)
        print(f"  {label:9} hit@1 {h1:.2f}  hit@3 {h3:.2f}  hit@5 {h5:.2f}  MRR {mrr:.2f}")
    print(f"  answer page among the 5 sources: plain {sum(r['in_sources_plain'] for r in answerable)}/{len(answerable)}, "
          f"expanded {sum(r['in_sources_fused'] for r in answerable)}/{len(answerable)}")
    changed = [r for r in answerable if r["rank_plain"] != r["rank_fused"]]
    for r in changed:
        print(f"    rank {r['rank_plain']} -> {r['rank_fused']}: {r['question'][:70]}")
    print(f"{len(unanswerable)} unanswerable questions let through by the gate: plain "
          f"{sum(r['gate_plain'] for r in unanswerable)}, expanded {sum(r['gate_fused'] for r in unanswerable)}")
    out_dir = settings.data_dir / "eval"
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"expansion-{datetime.now():%Y%m%d-%H%M%S}.json"
    path.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    print(f"Saved: {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
