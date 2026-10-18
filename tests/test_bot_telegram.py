"""The Telegram adapter against a fake Bot API: who is allowed, what each kind of message becomes, polling, start-up trouble, results."""
import asyncio
import dataclasses
import logging
import time
from pathlib import Path

import pytest
from telegram import Update

from bot_helpers import Clock, StubService
from ingest_helpers import make_env, make_photo
from studybuddy.bot.flow import ANSWER_SECONDS, SETTLE_SECONDS
from studybuddy.bot.telegram_app import TelegramBot
from studybuddy.ingest.results import INDEXED, IngestResult
from studybuddy.ingest.service import IngestService
from studybuddy.ingest.subjects import SPOOL_FOLDER, SubjectList
from studybuddy.logsafe import install_log_safety
from telegram_helpers import STUDENT, FakeTelegram

TOKEN = "123456789:AAHfake_token-value0123456789abcdefg"
STRANGER = 99


@pytest.fixture
def world(tmp_path, settings):
    install_log_safety([TOKEN])
    s = dataclasses.replace(settings, telegram_token=TOKEN, telegram_user_id=STUDENT, library_dir=tmp_path / "library",
                            data_dir=tmp_path / "data", subjects_file=tmp_path / "data" / "subjects.json")
    s.library_dir.mkdir(), s.data_dir.mkdir()
    return s


class Rig:
    """A bot wired to the fake API and a stub ingest service, driven on one event loop."""

    def __init__(self, settings, service=None):
        self.api, self.clock = FakeTelegram(), Clock()
        self.service = service or StubService()
        self.subjects = SubjectList(settings.data_dir / "subjects.json", ["Math", "Science", "English"])
        self.bot = TelegramBot(settings, self.service, self.subjects, request=self.api, clock=self.clock, tick_seconds=0.05,
                               connect_retry_seconds=0.05)
        self.loop = asyncio.new_event_loop()
        self.app = self.bot.build_application()
        self.run(self.app.initialize())

    def run(self, coro):
        return self.loop.run_until_complete(coro)

    def feed(self, update_dict):
        self.run(self.app.process_update(Update.de_json(update_dict, self.app.bot)))

    def settle(self):
        self.clock.advance(SETTLE_SECONDS + 0.1)
        self.run(self.bot.flow.tick())

    def spool(self):
        folder = self.bot.settings.library_dir / SPOOL_FOLDER
        return sorted(folder.iterdir()) if folder.exists() else []

    def close(self):
        self.run(self.app.shutdown())
        self.loop.close()


@pytest.fixture
def rig(world):
    r = Rig(world)
    yield r
    r.close()


# ---- who may talk to the bot ------------------------------------------------------------------------------

def test_the_student_sends_a_photo_and_gets_a_subject_question(rig):
    rig.feed(rig.api.photo())
    (question,) = rig.api.questions()
    assert question["text"].startswith("Which subject is this photo for?")
    assert rig.api.labels(question) == ["Math", "Science", "English", "Other...", "Auto-detect"]
    assert question["reply_to"] is not None and len(rig.spool()) == 1


def test_tapping_a_subject_queues_the_photo_and_answers_the_tap(rig):
    rig.feed(rig.api.photo())
    rig.feed(rig.api.tap(rig.api.questions()[0], "Science"))
    (call,) = rig.service.calls
    assert call["subject"] == "Science" and call["source_kind"] == ["photo"] and call["origin"] == "bot"
    assert "Queued 1 document (subject: Science)" in rig.api.messages[0]["text"] and rig.api.labels(rig.api.messages[0]) == []
    assert rig.api.methods("answerCallbackQuery")[-1]["text"] == "Got it."


@pytest.mark.parametrize("make", [
    lambda api: api.photo(user_id=STRANGER, caption="secret homework"),
    lambda api: api.text("hello bot", user_id=STRANGER),
    lambda api: api.command("/start", user_id=STRANGER),
    lambda api: api.document(user_id=STRANGER, caption="secret homework"),
    lambda api: api.photo(chat_type="group", caption="secret homework"),               # the right person, but not a private chat
])
def test_anyone_else_gets_no_reply_and_nothing_is_stored_or_logged_with_content(rig, caplog, make):
    with caplog.at_level(logging.DEBUG):
        rig.feed(make(rig.api))
    assert rig.api.messages == [] and rig.api.methods("sendMessage") == [] and rig.api.methods("getFile") == []
    assert rig.spool() == [] and rig.bot.state.all_items() == [] and rig.service.calls == []
    assert "unauthorised sender" in caplog.text and "secret homework" not in caplog.text and "hello bot" not in caplog.text


def test_a_strangers_button_tap_is_ignored_too(rig, caplog):
    rig.feed(rig.api.photo())
    question = rig.api.questions()[0]
    with caplog.at_level(logging.WARNING):
        rig.feed(rig.api.tap(question, "Math", user_id=STRANGER))
    assert rig.service.calls == [] and rig.api.methods("answerCallbackQuery") == [] and "unauthorised sender (id 99)" in caplog.text


# ---- kinds of message -------------------------------------------------------------------------------------

@pytest.mark.parametrize("name,mime,kind", [
    ("scan.pdf", "application/pdf", "pdf"), ("page.PDF", "application/octet-stream", "pdf"),
    ("pic.jpg", "image/jpeg", "photo-file"), ("pic.png", "image/png", "photo-file"), ("IMG_1.HEIC", "image/heic", "photo-file"),
    ("pic.jpeg", None, "photo-file"),
])
def test_documents_are_sorted_into_pdf_and_image_files(rig, name, mime, kind):
    rig.feed(rig.api.document(name=name, mime=mime))
    rig.feed(rig.api.tap(rig.api.questions()[0], "Math"))
    assert rig.service.calls[0]["source_kind"] == [kind]


@pytest.mark.parametrize("name,mime,what", [("essay.docx", "application/vnd.openxmlformats-officedocument.wordprocessingml.document", "a DOCX file"),
                                            ("notes.txt", "text/plain", "a TXT file"), ("clip.mp4", "video/mp4", "a MP4 file")])
def test_other_documents_are_declined_in_plain_words(rig, name, mime, what):
    rig.feed(rig.api.document(name=name, mime=mime, content=b"data"))
    text = rig.api.messages[-1]["text"]
    assert "photos, images sent as a file, and PDFs" in text and what in text and "Nothing was saved" in text
    assert rig.spool() == [] and rig.api.methods("getFile") == []


def test_other_kinds_of_message_are_declined(rig):
    rig.feed(rig.api.video())
    assert "not a video" in rig.api.messages[-1]["text"] and rig.spool() == []


def test_a_file_over_twenty_megabytes_is_refused_before_any_download(rig):
    rig.feed(rig.api.document(size=25_000_000))
    text = rig.api.messages[-1]["text"]
    assert "too big" in text and "20 MB" in text and "25.0 MB" in text
    assert rig.api.methods("getFile") == [] and rig.spool() == []


def test_a_file_telegram_says_is_too_big_is_refused(rig):
    update = rig.api.photo()
    rig.api.too_big.add(update["message"]["photo"][-1]["file_id"])
    rig.feed(update)
    assert "too big" in rig.api.messages[-1]["text"] and rig.spool() == []


def test_a_network_failure_during_the_download_is_retried_later(rig):
    update = rig.api.photo()
    file_id = update["message"]["photo"][-1]["file_id"]
    rig.api.fail_downloads.add(file_id)
    rig.feed(update)
    assert "couldn't download" in rig.api.messages[-1]["text"] and rig.spool() == []
    rig.api.fail_downloads.clear()
    rig.clock.advance(61)
    rig.run(rig.bot.flow.tick())
    assert len(rig.spool()) == 1 and len(rig.api.questions()) == 1


def test_text_gets_a_hint_and_help_too(rig):
    rig.feed(rig.api.text("hi"))
    rig.feed(rig.api.command("/start"))
    rig.feed(rig.api.command("/help"))
    assert all("Send me photos or PDFs" in m["text"] for m in rig.api.messages) and len(rig.api.messages) == 3


def test_other_then_a_typed_name(rig):
    rig.feed(rig.api.photo())
    rig.feed(rig.api.tap(rig.api.questions()[0], "Other..."))
    assert rig.api.messages[0]["text"].startswith("Type the name of the new subject")
    rig.feed(rig.api.text("Astronomy"))
    assert rig.service.calls[0]["subject"] == "Astronomy"


def test_three_photos_sent_together_get_one_question_and_three_jobs(rig):
    for _ in range(3):
        rig.feed(rig.api.photo(group="album-1"))
    assert rig.api.questions() == [] and len(rig.spool()) == 3
    rig.settle()
    (question,) = rig.api.questions()
    assert question["text"].startswith("3 photos: same subject for all?") and "Make these one document" in rig.api.labels(question)
    rig.feed(rig.api.tap(question, "Math"))
    assert len(rig.service.calls) == 3 and not any(c["is_album"] for c in rig.service.calls)


def test_make_these_one_document_is_the_only_way_to_an_album(rig):
    for _ in range(2):
        rig.feed(rig.api.photo(group="g"))
    rig.settle()
    rig.feed(rig.api.tap(rig.api.questions()[0], "Make these one document"))
    assert rig.service.calls == []
    rig.feed(rig.api.tap(rig.api.questions()[0], "English"))
    (call,) = rig.service.calls
    assert call["is_album"] is True and len(call["paths"]) == 2


def test_the_photo_caption_with_a_known_subject_skips_the_question(rig):
    rig.feed(rig.api.photo(caption="Science"))
    assert rig.api.questions() == [] and rig.service.calls[0]["subject"] == "Science"


def test_no_answer_in_five_minutes_is_announced(rig):
    rig.feed(rig.api.photo())
    rig.clock.advance(ANSWER_SECONDS + 1)
    rig.run(rig.bot.flow.tick())
    assert rig.service.calls[0]["subject"] is None and "No answer in 5 minutes" in rig.api.messages[0]["text"]


def test_the_token_is_never_written_to_the_log(rig, caplog):
    with caplog.at_level(logging.DEBUG):
        rig.feed(rig.api.photo())
        rig.feed(rig.api.tap(rig.api.questions()[0], "Math"))
        rig.feed(rig.api.photo(user_id=STRANGER))
        logging.getLogger("httpx").info("POST https://api.telegram.org/bot%s/getUpdates", TOKEN)
    assert TOKEN not in caplog.text and any(TOKEN in url for url in rig.api.urls)         # it IS in the URLs, just never logged


# ---- the thread, polling and start-up trouble -------------------------------------------------------------

def wait_for(condition, timeout=10.0):
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        if condition():
            return True
        time.sleep(0.02)
    return False


@pytest.fixture
def started(world):
    api, service = FakeTelegram(), StubService()
    subjects = SubjectList(world.data_dir / "subjects.json", ["Math", "Science"])
    bot = TelegramBot(world, service, subjects, request=api, tick_seconds=0.05, connect_retry_seconds=0.05)
    yield bot, api, service
    bot.stop()


def test_the_bot_polls_in_its_own_thread_and_answers(started):
    bot, api, service = started
    api.updates.append(api.photo())
    bot.start()
    assert bot.ready.wait(10) and "connected as @study_buddy_test_bot" in bot.status
    assert wait_for(lambda: api.questions())
    api.updates.append(api.tap(api.questions()[0], "Math"))
    assert wait_for(lambda: service.calls) and service.calls[0]["subject"] == "Math"
    assert any(n == "getUpdates" for n, _ in api.calls)
    bot.stop()
    assert not bot._thread.is_alive() and bot.status == "stopped"


def test_strangers_polled_from_telegram_get_nothing(started):
    bot, api, service = started
    api.updates.append(api.photo(user_id=STRANGER))
    api.updates.append(api.text("hi", user_id=STRANGER))
    bot.start()
    assert bot.ready.wait(10)
    assert wait_for(lambda: not api.updates) and not wait_for(lambda: api.messages, timeout=0.5)
    assert api.methods("sendMessage") == [] and service.calls == []


def test_messages_sent_while_the_app_was_off_are_read_when_it_starts(started):
    bot, api, service = started
    for _ in range(2):
        api.updates.append(api.photo(caption="Science"))                 # waiting at Telegram since before the start
    bot.start()
    assert wait_for(lambda: len(service.calls) == 2)


def test_start_up_without_a_connection_keeps_trying_and_does_not_block(started):
    bot, api, service = started
    api.fail_get_me = 3
    bot.start()
    assert bot.ready.wait(10) and sum(1 for n, _ in api.calls if n == "getMe") >= 1


def test_a_rejected_token_stops_the_bot_quietly(started, caplog):
    bot, api, service = started
    api.invalid_token = True
    with caplog.at_level(logging.ERROR):
        bot.start()
        assert wait_for(lambda: not bot._thread.is_alive())
    assert "rejected" in caplog.text and TOKEN not in caplog.text and not bot.ready.is_set() and "token" in bot.status


def test_results_from_the_ingest_worker_reach_the_chat(started):
    bot, api, service = started
    bot.start()
    assert bot.ready.wait(10)
    api.updates.append(api.photo(caption="Math"))
    assert wait_for(lambda: service.calls)
    job = service.calls[0]["job_id"]
    bot._on_ingest_result(IngestResult(status=INDEXED, job_id=job, subject="Math", title="Algebra", page_count=2, subject_source="user",
                                       final_path="C:/lib/Math/2026-10-18_0930_Math_algebra.jpg"))     # from another thread, as the worker does
    assert wait_for(lambda: any("Filed:" in m["text"] for m in api.messages))
    assert "2 pages." in next(m["text"] for m in api.messages if "Filed:" in m["text"])


def test_a_result_before_the_bot_is_up_is_ignored_safely(world):
    bot = TelegramBot(world, StubService(), SubjectList(world.data_dir / "subjects.json", ["Math"]), request=FakeTelegram())
    bot._on_ingest_result(IngestResult(status=INDEXED, job_id="x"))


def test_the_bot_will_not_be_built_without_a_user_id(world):
    with pytest.raises(ValueError, match="user id"):
        TelegramBot(dataclasses.replace(world, telegram_user_id=None), StubService(), SubjectList(world.data_dir / "subjects.json", []))


# ---- all the way through the real ingest service ----------------------------------------------------------

def test_a_photo_travels_from_telegram_to_the_library_and_the_student_is_told(tmp_path, settings):
    env = make_env(tmp_path, settings)
    s = dataclasses.replace(env.settings, telegram_token=TOKEN, telegram_user_id=STUDENT)
    service = IngestService(s, env.index, env.reader, None, env.chat, env.subjects, poll_interval=0.02)
    api = FakeTelegram()
    bot = TelegramBot(s, service, env.subjects, request=api, tick_seconds=0.05)
    photo_bytes = make_photo(tmp_path / "source.jpg", size=(300, 400)).read_bytes()
    try:
        bot.start()
        service.start()
        assert bot.ready.wait(10)
        api.updates.append(api.photo(content=photo_bytes, caption="Science"))
        assert wait_for(lambda: any(m["text"].startswith("Filed:") for m in api.messages), timeout=30)
        reply = next(m["text"] for m in api.messages if m["text"].startswith("Filed:"))
        assert "Subject: Science" in reply and "Title: Photosynthesis Worksheet" in reply and "1 page." in reply
        assert "Handwritten names, dates, scores and answer letters may be read wrongly" in reply
        filed = list((s.library_dir / "Science").glob("*.jpg"))
        assert len(filed) == 1 and filed[0].read_bytes() == photo_bytes
        assert list((s.library_dir / SPOOL_FOLDER).glob("*")) == []                           # the spool copy was removed after filing
        assert env.index.documents()[0].subject == "Science"
        api.updates.append(api.photo(content=photo_bytes, caption="Science"))                 # the same photo again
        assert wait_for(lambda: any(m["text"].startswith("I already have this one") for m in api.messages), timeout=30)
    finally:
        bot.stop()
        service.stop()
        env.index.close()
