"""End to end with the REAL models, in temporary folders: ingest the sample pages, ask questions, then do it again through the bot.

  python scripts/e2e_live.py [--keep] [--no-ocr] [--questions N]

Needs LM Studio running with the three models loaded, and SB_SAMPLES_DIR with the sample pages (page .jpg + reference .txt) and
questions.json. Your real library, index and Telegram bot are never touched: everything happens under a temporary folder, and the
Telegram side is the offline fake from tests/ (no network, no token). Prints the time of every step and saves the numbers to
SB_DATA_DIR/eval/e2e-<time>.json. The sample photos are read from your folder and are never copied into the repository.

Stage 1  ingest each sample page through the real IngestService (reader, OCR cross-check, tagging, summary, embeddings)
Stage 2  ask the questions from questions.json (answerable, one overview, a few that must be refused)
Stage 3  the same pages as Telegram messages: caption, buttons, a duplicate, a group of two with Auto-detect
"""
import argparse
import dataclasses
import json
import re
import shutil
import sys
import tempfile
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tests"))

from studybuddy.chat.calibrate import FAR, load_cases  # noqa: E402
from studybuddy.chat.overview_eval import judge_answerable, judge_unanswerable, load_overview_cases  # noqa: E402
from studybuddy.chat.session import ChatConfig, ChatSession  # noqa: E402
from studybuddy.checks import FAIL  # noqa: E402
from studybuddy.config import load_settings  # noqa: E402
from studybuddy.errors import StudyBuddyError  # noqa: E402
from studybuddy.health import run_health  # noqa: E402
from studybuddy.llm import LMStudioClient  # noqa: E402
from studybuddy.logsafe import install_log_safety  # noqa: E402
from studybuddy.reader.evaluate import error_rates  # noqa: E402
from studybuddy.ui.runtime import build_runtime  # noqa: E402

FAKE_TOKEN = "123456789:AAHfake_token-value0123456789abcdefg"
STUDENT = 42


def mentions(expected, text):
    """The answer may paraphrase ("about 78%" vs "78%"): it must have the numbers and the first real word of the expected text."""
    parts = re.findall(r"[a-z]+|[0-9]+", expected.casefold())
    words = [w for w in parts if w.isalpha() and len(w) >= 4]
    low = text.casefold()
    return all(n in low for n in parts if n.isdigit()) and (not words or words[0] in low)


def temp_settings(base, root, name):
    library, data = root / name / "library", root / name / "data"
    library.mkdir(parents=True)
    data.mkdir(parents=True)
    subjects = data / "subjects.json"
    shutil.copy(base.subjects_file, subjects)
    return dataclasses.replace(base, library_dir=library, data_dir=data, subjects_file=subjects, telegram_token=FAKE_TOKEN,
                               telegram_user_id=STUDENT, telegram_enabled=True)


def sample_pages(samples_dir):
    pages = [(p, p.with_suffix(".txt")) for p in sorted(samples_dir.glob("*.jpg")) if p.with_suffix(".txt").is_file()]
    if not pages:
        raise StudyBuddyError(f"No sample pages (page .jpg with a same-name .txt) in {samples_dir}.")
    return pages


def stage_ingest(runtime, pages, scratch, report):
    print("\nStage 1: ingest through the real service", flush=True)
    out = []
    for image, reference in pages:
        copy = scratch / image.name
        shutil.copy(image, copy)
        started = time.monotonic()
        ticket = runtime.service.submit([copy], subject=None, source_kind="photo", origin="e2e")
        result = ticket.result(timeout=1800)
        seconds = round(time.monotonic() - started, 1)
        row = {"page": image.stem, "status": result.status, "seconds": seconds, "subject": result.subject, "pages": result.page_count,
               "flagged": len(result.flagged_pages), "title": result.title}
        if result.ok:
            text = " ".join(p["text"] or "" for p in runtime.index.store.pages(result.doc_id))
            cer, wer = error_rates(reference.read_text(encoding="utf-8"), text)
            row.update(cer=round(cer, 3), wer=round(wer, 3))
        elif result.error:
            row["error"] = result.error
        out.append(row)
        print(f"  {image.stem}: {result.status} in {seconds} s, subject {result.subject}, {result.page_count} page(s), "
              f"{len(result.flagged_pages)} flagged" + (f", CER {row['cer']}" if "cer" in row else ""), flush=True)
    report["ingest"] = out
    return out


def stage_chat(runtime, settings, limit, report):
    print("\nStage 2: questions", flush=True)
    items = [q for q in json.loads((settings.samples_dir / "questions.json").read_text(encoding="utf-8")) if q.get("kind") != "overview"]
    rows = []

    def ask(question):
        session = ChatSession(runtime.index, runtime.client, settings, runtime.subjects, ChatConfig.from_settings(settings))
        started = time.monotonic()
        answer = session.ask(question)
        return answer, round(time.monotonic() - started, 1)

    for item in items[:limit]:
        answer, seconds = ask(item["question"])
        outcome = judge_answerable(item["question"], answer, seconds)
        has_text = mentions(item["expect_text"], answer.text) if item.get("expect_text") else None
        ok = outcome.ok and bool(answer.cited) and has_text is not False
        rows.append({"question": item["question"], "kind": "answerable", "ok": ok, "seconds": seconds, "cited": bool(answer.cited),
                     "has_expected_text": has_text, "why": outcome.why})
        print(f"  [{'ok' if ok else 'NOT OK'}] {seconds} s  {item['question']}", flush=True)
    overview = load_overview_cases(settings.samples_dir)
    if overview:
        answer, seconds = ask(overview[0].question)
        ok = answer.status == "answered" and answer.mode == "overview" and bool(answer.cited)
        rows.append({"question": overview[0].question, "kind": "overview", "ok": ok, "seconds": seconds, "status": answer.status})
        print(f"  [{'ok' if ok else 'NOT OK'}] {seconds} s  (overview) {overview[0].question}", flush=True)
    for case in [c for c in load_cases(settings.samples_dir) if c.kind == FAR][:2]:
        answer, seconds = ask(case.question)
        outcome = judge_unanswerable(case.question, answer, seconds)
        rows.append({"question": case.question, "kind": "must-refuse", "ok": outcome.ok, "seconds": seconds, "why": outcome.why})
        print(f"  [{'ok' if outcome.ok else 'NOT OK'}] {seconds} s  (must refuse) {case.question}", flush=True)
    report["questions"] = rows
    return rows


def stage_bot(base, pages, root, use_ocr, report):
    from studybuddy.bot.telegram_app import TelegramBot
    from telegram_helpers import FakeTelegram

    print("\nStage 3: the bot flow (fake Telegram, real service and models)", flush=True)
    settings = temp_settings(base, root, "bot")
    runtime = build_runtime(settings, use_ocr=use_ocr, with_bot=False)
    api = FakeTelegram()
    runtime.bot = TelegramBot(settings, runtime.service, runtime.subjects, request=api, tick_seconds=0.2)
    rows = []

    def replies(prefix):
        return [m for m in api.messages if m["text"].startswith(prefix)]

    def wait_reply(prefix, count, label, seconds=900):
        started = time.monotonic()
        while time.monotonic() - started < seconds:
            if len(replies(prefix)) >= count:
                took = round(time.monotonic() - started, 1)
                rows.append({"step": label, "seconds": took, "ok": True})
                print(f"  [ok] {label}: {took} s", flush=True)
                return True
            time.sleep(0.3)
        rows.append({"step": label, "seconds": seconds, "ok": False})
        print(f"  [NOT OK] {label}: no reply in {seconds} s", flush=True)
        return False

    try:
        runtime.start()
        if not runtime.bot.ready.wait(30):
            raise StudyBuddyError("the bot did not start")
        first, second, third = (p.read_bytes() for p, _ in (pages * 3)[:3])
        api.updates.append(api.photo(content=first, caption="Science"))
        wait_reply("Filed:", 1, "photo with caption 'Science' -> filed")
        filed = replies("Filed:")[0]["text"] if replies("Filed:") else ""
        rows.append({"step": "reply carries the handwriting warning", "ok": "Handwritten names, dates, scores and answer letters may be read wrongly" in filed})
        api.updates.append(api.photo(content=first, caption="Science"))
        wait_reply("I already have this one", 1, "same photo again -> duplicate message")
        if len({p for p, _ in pages}) >= 3:
            api.updates.append(api.photo(content=second))
            deadline = time.monotonic() + 30
            while not api.questions() and time.monotonic() < deadline:
                time.sleep(0.2)
            if api.questions():
                api.updates.append(api.tap(api.questions()[-1], "Auto-detect"))
                wait_reply("Filed:", 2, "photo with no caption -> button -> filed (auto-detect)")
            before = len(api.messages)
            api.updates.append(api.photo(content=third, group="e2e-group"))
            api.updates.append(api.photo(content=second + b"\x00", group="e2e-group"))   # different bytes: not a duplicate
            deadline = time.monotonic() + 30
            asked = []
            while not asked and time.monotonic() < deadline:
                asked = [m for m in api.messages[before:] if m.get("reply_markup", {}).get("inline_keyboard")]
                time.sleep(0.2)
            one = len(asked) == 1 and "2 photos" in asked[0]["text"]
            rows.append({"step": "two photos together -> ONE question", "ok": one})
            print(f"  [{'ok' if one else 'NOT OK'}] two photos together -> one question", flush=True)
            if asked:
                api.updates.append(api.tap(asked[0], "Auto-detect"))
                started = time.monotonic()

                def finished():
                    return [m for m in api.messages[before:] if m["text"].startswith(("Filed:", "I already have", "I couldn't read"))]

                while len(finished()) < 2 and time.monotonic() - started < 900:
                    time.sleep(0.3)
                took = round(time.monotonic() - started, 1)
                rows.append({"step": "tap Auto-detect -> two documents answered", "seconds": took, "ok": len(finished()) == 2})
                print(f"  [{'ok' if len(finished()) == 2 else 'NOT OK'}] group of two -> two replies: {took} s", flush=True)
        spool = settings.library_dir / "_Unsorted"
        time.sleep(1)
        stray = list(spool.glob("*")) if spool.exists() else []
        rows.append({"step": "spool folder empty after filing", "ok": not stray})
        print(f"  [{'ok' if not stray else 'NOT OK'}] spool folder empty after filing ({len(stray)} left)", flush=True)
    finally:
        runtime.close()
    report["bot"] = rows
    return rows


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--keep", action="store_true", help="keep the temporary folders (they hold copies of your sample pages)")
    parser.add_argument("--no-ocr", action="store_true")
    parser.add_argument("--questions", type=int, default=5, help="how many answerable questions to ask (default 5)")
    args = parser.parse_args(argv)
    try:
        base = load_settings()
        install_log_safety([base.telegram_token])
        if base.samples_dir is None or not (base.samples_dir / "questions.json").is_file():
            print("[FAIL] set SB_SAMPLES_DIR to the folder with the sample pages and questions.json")
            return 1
        pages = sample_pages(base.samples_dir)
        client = LMStudioClient(base, timeout=20.0)
        try:
            blocking = [r for r in run_health(base, client) if r.status == FAIL and r.name.startswith(("LM Studio", "model"))]
        finally:
            client.close()
        if blocking:
            print("[FAIL] LM Studio is not ready:\n  " + "\n  ".join(f"{r.name}: {r.detail}" for r in blocking))
            return 1
    except StudyBuddyError as exc:
        print(f"[FAIL] {exc}")
        return 1

    report: dict = {"started": datetime.now().isoformat(timespec="seconds"), "models": [base.reader_model, base.chat_model, base.embed_model]}
    root = Path(tempfile.mkdtemp(prefix="sb-e2e-"))
    started = time.monotonic()
    ok = True
    try:
        settings = dataclasses.replace(temp_settings(base, root, "direct"), samples_dir=base.samples_dir, telegram_token=None, telegram_user_id=None)
        scratch = root / "incoming"
        scratch.mkdir()
        runtime = build_runtime(settings, use_ocr=not args.no_ocr, with_bot=False).start()
        try:
            ingested = stage_ingest(runtime, pages, scratch, report)
            ok &= all(r["status"] == "indexed" for r in ingested)
            asked = stage_chat(runtime, settings, args.questions, report)
            ok &= all(r["ok"] for r in asked)
        finally:
            runtime.close()
        botted = stage_bot(base, pages, root, not args.no_ocr, report)
        ok &= all(r["ok"] for r in botted if "ok" in r)
    except StudyBuddyError as exc:
        print(f"[FAIL] {exc}")
        ok = False
    finally:
        report["total_seconds"] = round(time.monotonic() - started, 1)
        report["passed"] = bool(ok)
        eval_dir = base.data_dir / "eval"
        eval_dir.mkdir(parents=True, exist_ok=True)
        out = eval_dir / f"e2e-{datetime.now():%Y%m%d-%H%M%S}.json"
        out.write_text(json.dumps(report, indent=1), encoding="utf-8")
        if args.keep:
            print(f"\nTemporary folders kept: {root}")
        else:
            shutil.rmtree(root, ignore_errors=True)
    print(f"\n{'PASSED' if ok else 'FAILED'} in {report['total_seconds']} s. Numbers saved to {out}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
