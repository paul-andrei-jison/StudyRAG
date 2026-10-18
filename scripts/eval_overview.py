"""Do overview questions work, and do the other questions stay as they were? Full pipeline, real models, your own pages.

Needs the index named by SB_DATA_DIR / SB_LIBRARY_DIR (ingest the sample pages first, ideally into temporary folders) and, in
SB_SAMPLES_DIR: questions.json (with "kind": "overview" items and "expect_docs"), unanswerable.json, optionally nearmiss.json.
Reports, per group, how many came out right; how many questions that are NOT overview requests were handled as one (target 0);
and with --timing the model calls and seconds for a per-document answer, a condensed ~14-page document and the stored-summary
fallback (the last two use SYNTHETIC documents built from your pages in a scratch index). Results go to SB_DATA_DIR/eval/.
"""
import argparse
import dataclasses
import hashlib
import json
import sys
import tempfile
import time
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from studybuddy.chat.budget import TokenEstimator  # noqa: E402
from studybuddy.chat.calibrate import load_cases  # noqa: E402
from studybuddy.chat.condense import plan_document  # noqa: E402
from studybuddy.chat.overview_eval import (  # noqa: E402
    format_report, judge_answerable, judge_near_miss, judge_overview, judge_unanswerable, load_near_misses,
    load_overview_cases, resolve_expectation, wrongly_detected)
from studybuddy.chat.prompt import OVERVIEW_SYSTEM_PROMPT, format_source  # noqa: E402
from studybuddy.chat.session import ChatConfig, ChatSession  # noqa: E402
from studybuddy.config import load_settings  # noqa: E402
from studybuddy.errors import StudyBuddyError  # noqa: E402
from studybuddy.index.index import Index  # noqa: E402
from studybuddy.index.store import DocumentMeta  # noqa: E402
from studybuddy.ingest.subjects import SubjectList  # noqa: E402
from studybuddy.llm import LMStudioClient  # noqa: E402
from studybuddy.reader.quality import PageRecord  # noqa: E402


class CountingClient:
    """Delegates to the real client and counts the model calls."""

    def __init__(self, client):
        self._client, self.chats, self.streams = client, 0, 0

    def chat(self, *args, **kwargs):
        self.chats += 1
        return self._client.chat(*args, **kwargs)

    def chat_stream(self, *args, **kwargs):
        self.streams += 1
        return self._client.chat_stream(*args, **kwargs)

    def __getattr__(self, name):
        return getattr(self._client, name)

    def reset(self):
        self.chats = self.streams = 0


def ask(index, client, settings, subjects, question, subject=None):
    session = ChatSession(index, client, settings, subjects, ChatConfig.from_settings(settings))
    if subject:
        session.set_subject(subject)
    started = time.monotonic()
    answer = session.ask(question)
    return answer, round(time.monotonic() - started, 1)


def add_synthetic(index, key, subject, title, pages, summary):
    path = f"synthetic/{key}.jpg"
    records = [PageRecord(i, path, "image", text=t, quality=0.95) for i, t in enumerate(pages, 1)]
    return index.add_document(DocumentMeta(key, path, "photo", subject, title, summary), records)


def timing(settings, client, real_index, counter):
    """Model calls and seconds for the three expensive paths. The long documents are synthetic."""
    texts = [p["text"] for d in real_index.documents() for p in real_index.store.pages(d.id) if p["text"]]
    block = "\n\n".join(texts)
    results = {}
    with tempfile.TemporaryDirectory(prefix="sb-overview-") as tmp:
        scratch_settings = dataclasses.replace(settings, data_dir=Path(tmp) / "data", library_dir=Path(tmp) / "lib")
        scratch_settings.data_dir.mkdir(parents=True)
        subjects = SubjectList(scratch_settings.data_dir / "subjects.json", ["Science", "Math", "History"])
        scratch = Index.open(scratch_settings, client)
        try:
            est = TokenEstimator()
            ctx = counter.context_length(settings.chat_model) or 0
            available = ctx - settings.chat_max_answer_tokens - est.estimate(OVERVIEW_SYSTEM_PROMPT) - 30
            # the reviewer must NOT fit whole, or condensing is never measured: repeat the sample text until it overflows by half
            repeat = max(1, -(-int(available * 1.5) // (14 * max(1, est.estimate(block)))))
            pages14 = [f"Page {i} of the reviewer.\n" + "\n\n".join([block] * repeat) for i in range(1, 15)]
            pages40 = [f"Page {i} of the book. {texts[i % len(texts)]}" for i in range(1, 41)]
            add_synthetic(scratch, "rev14", "Math", "Synthetic 14-page reviewer", pages14, "A synthetic reviewer built from the sample pages.")
            real_summaries = " ".join(d.summary for d in real_index.documents() if d.summary)
            add_synthetic(scratch, "book40", "History", "Synthetic 40-page book", pages40, real_summaries)
            for name, subject, pages in (("condensed (14 pages)", "Math", pages14), ("stored summary (40 pages)", "History", pages40)):
                doc = next(d for d in scratch.documents() if d.subject == subject)
                costs = [est.estimate(format_source(_source_of(doc, i, t))) for i, t in enumerate(pages, 1)]
                counter.reset()
                answer, seconds = ask(scratch, counter, settings, subjects, "summarize this whole document", subject)
                results[name] = {"seconds": seconds, "condensing_calls": answer.model_calls_condensing,
                                 "chat_calls": counter.chats, "stream_calls": counter.streams, "status": answer.status,
                                 "mode": answer.mode, "tier_expected": plan_document(available, costs, est).tier,
                                 "estimated_page_tokens": costs[0], "context": ctx, "cited": answer.cited,
                                 "notes": answer.overview_notes[1:]}
        finally:
            scratch.close()
    return results


def stems_of_path(stem_of, path):
    try:
        return stem_of.get(hashlib.sha256(Path(path).read_bytes()).hexdigest(), "?")
    except OSError:
        return "?"


def _source_of(doc, number, text):
    from studybuddy.chat.answers import Source
    return Source(number, doc.subject, doc.title, doc.file_path, number, doc.file_path, text, 0.95, [])


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--timing", action="store_true", help="also time the condensed, stored-summary and per-document paths")
    args = parser.parse_args()
    try:
        settings = load_settings()
        if settings.samples_dir is None or not (settings.samples_dir / "questions.json").is_file():
            print("Need SB_SAMPLES_DIR with questions.json.")
            return 0
        overview_cases = load_overview_cases(settings.samples_dir)
        plain_cases = load_cases(settings.samples_dir)
        near_misses = load_near_misses(settings.samples_dir)
    except (StudyBuddyError, ValueError, KeyError) as exc:
        print(f"[FAIL] {exc}")
        return 1
    if not overview_cases:
        print("questions.json has no items with \"kind\": \"overview\".")
        return 0
    stem_of = {hashlib.sha256(p.read_bytes()).hexdigest(): p.stem for p in settings.samples_dir.glob("*.jpg")}
    raw_client = LMStudioClient(settings)
    client = CountingClient(raw_client)
    try:
        index = Index.open(settings, raw_client)
        if index.store.counts()["chunks"] == 0:
            print("The index is empty: ingest the sample pages first (scripts/ingest.py).")
            return 0
        subjects = SubjectList.from_settings(settings)

        def stems(answer):
            found = []
            for s in answer.sources:
                try:
                    found.append(stem_of.get(hashlib.sha256(Path(s.image_path).read_bytes()).hexdigest(), "?"))
                except OSError:
                    found.append("?")
            return found

        library = []
        for doc in index.documents():
            page = index.store.pages(doc.id)[0]
            added = datetime.fromisoformat(doc.created_at)
            library.append((stems_of_path(stem_of, page["image_path"]), (datetime.now(added.tzinfo) - added).total_seconds() / 60))

        outcomes = []
        for case in overview_cases:
            case = resolve_expectation(case, library)
            client.reset()
            answer, seconds = ask(index, client, settings, subjects, case.question)
            outcome = judge_overview(case, answer, stems(answer), seconds)
            outcome.extra.update(model_calls=client.chats + client.streams, text=answer.text[:600], notes=answer.overview_notes[:3])
            outcomes.append(outcome)
        for case in plain_cases:
            answer, seconds = ask(index, client, settings, subjects, case.question)
            outcomes.append((judge_answerable if case.answerable else judge_unanswerable)(case.question, answer, seconds))
        for case in near_misses:
            answer, seconds = ask(index, client, settings, subjects, case.question)
            outcomes.append(judge_near_miss(case, answer, seconds))
        print(format_report(outcomes))
        timings = timing(settings, client, index, client) if args.timing else {}
        for name, t in timings.items():
            print(f"\ntiming - {name}: {t['seconds']} s, {t['condensing_calls']} condensing calls, {t['chat_calls']} chat + {t['stream_calls']} stream calls, "
                  f"tier expected {t['tier_expected']}, status {t['status']}, notes {t['notes']}")
        index.close()
    except StudyBuddyError as exc:
        print(f"[FAIL] {exc}")
        return 1
    finally:
        raw_client.close()
    out_dir = settings.data_dir / "eval"
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"overview-{datetime.now():%Y%m%d-%H%M%S}.json"
    path.write_text(json.dumps({"outcomes": [dataclasses.asdict(o) for o in outcomes], "timings": timings,
                                "wrongly_detected": wrongly_detected(outcomes)}, indent=2, default=str), encoding="utf-8")
    print(f"\nSaved: {path}")
    return 0 if all(o.ok for o in outcomes) else 2


if __name__ == "__main__":
    sys.exit(main())
