"""The bot's conversation: arrival, the subject question, grouping, timeouts, restarts and the messages about results."""
from pathlib import Path

import pytest

from bot_helpers import JPEG, PDF, WHEN, Clock, FakeMessenger, StubService, incoming, run
from studybuddy.bot.flow import (ANSWER_SECONDS, DOWNLOAD_LIMIT_BYTES, HANDWRITING, MAX_DOWNLOAD_ATTEMPTS, RETRY_SECONDS, SETTLE_SECONDS,
                                 BotFlow, DownloadError, TooBigError)
from studybuddy.bot.spool import Spool
from studybuddy.bot.state import ASKING, CLOSED, DONE, DOWNLOADING, SUBMITTED, StateStore
from studybuddy.errors import StudyBuddyError
from studybuddy.ingest.results import DEFERRED, DUPLICATE, FAILED, INDEXED, FlaggedPage, IngestResult
from studybuddy.ingest.subjects import SPOOL_FOLDER, SubjectList


class Env:
    def __init__(self, tmp_path):
        self.root = tmp_path
        self.library = tmp_path / "library"
        self.clock = Clock()
        self.messenger = FakeMessenger()
        self.service = StubService()
        self.subjects = SubjectList(tmp_path / "subjects.json", ["Math", "Science", "English"])
        self.state = StateStore(tmp_path / "data" / "bot")
        self.flow = self.make_flow()

    def make_flow(self):
        return BotFlow(self.service, self.subjects, Spool(self.library), self.state, self.messenger, self.clock)

    def settle(self):
        self.clock.advance(SETTLE_SECONDS + 0.1)
        run(self.flow.tick())

    @property
    def question_message(self):
        return [m for m in self.messenger.messages if m.buttons][-1]      # the newest question

    def tap(self, label, message=None):
        message = message or self.question_message
        return run(self.flow.on_callback(42, message.id, message.now_data_of(label)))

    def spool_files(self):
        folder = self.library / SPOOL_FOLDER
        return sorted(p for p in folder.iterdir()) if folder.exists() else []


@pytest.fixture
def env(tmp_path):
    return Env(tmp_path)


# ---- one photo --------------------------------------------------------------------------------------------

def test_a_photo_is_saved_at_once_and_the_student_is_asked_for_the_subject(env):
    run(env.flow.on_file(incoming("photo")))
    assert len(env.spool_files()) == 1 and env.spool_files()[0].read_bytes().startswith(JPEG) and env.service.calls == []
    q = env.question_message
    assert q.text.startswith("Which subject is this photo for?") and "5 minutes" in q.text
    assert q.labels == ["Math", "Science", "English", "Other...", "Auto-detect"]       # the live subject list, then the two extras
    assert q.reply_to is not None and "Make these one document" not in q.labels


def test_the_buttons_follow_the_live_subject_list(env):
    env.subjects.add("Astronomy")
    run(env.flow.on_file(incoming("photo")))
    assert "Astronomy" in env.question_message.labels


def test_tapping_a_subject_submits_the_file_as_its_own_job(env):
    inc = incoming("photo")
    run(env.flow.on_file(inc))
    assert env.tap("Science") == "Got it."
    (call,) = env.service.calls
    assert call["subject"] == "Science" and call["origin"] == "bot" and call["is_album"] is False and call["cleanup_sources"] is True
    assert call["source_kind"] == ["photo"] and call["received_at"] == WHEN and call["paths"][0].parent.name == SPOOL_FOLDER
    q = env.question_message
    assert "Queued 1 document (subject: Science)" in q.now_text and q.now_labels == []        # the buttons are gone
    item = env.state.all_items()[0]
    assert item.stage == SUBMITTED and item.job_id == "job-1" and item.subject == "Science"
    stale = "s:" + env.state.all_questions()[0].id + ":0"
    assert run(env.flow.on_callback(42, q.id, stale)) == "That question is already answered."
    assert len(env.service.calls) == 1                                                       # a second tap does nothing


def test_auto_detect_submits_without_a_subject(env):
    run(env.flow.on_file(incoming("photo")))
    env.tap("Auto-detect")
    assert env.service.calls[0]["subject"] is None and "auto-detect" in env.question_message.now_text


@pytest.mark.parametrize("kind,expected", [("photo", "photo"), ("photo-file", "photo-file"), ("pdf", "pdf")])
def test_the_source_kind_follows_how_it_was_sent(env, kind, expected):
    run(env.flow.on_file(incoming(kind, name="scan.pdf" if kind == "pdf" else "pic.jpg")))
    env.messenger.files = {}
    env.tap("Math")
    assert env.service.calls[0]["source_kind"] == [expected]


def test_a_pdf_keeps_its_type_in_the_spool(env):
    inc = incoming("pdf", name="Chapter 3.pdf", file_id="PDFID")
    env.messenger.files["PDFID"] = PDF
    run(env.flow.on_file(inc))
    assert env.spool_files()[0].suffix == ".pdf" and "Chapter 3" in env.spool_files()[0].name


# ---- the caption ------------------------------------------------------------------------------------------

@pytest.mark.parametrize("caption,subject", [("science", "Science"), ("  Math ", "Math"), ("Subject: English", "English"), ("subject:math", "Math")])
def test_a_caption_naming_a_known_subject_skips_the_buttons(env, caption, subject):
    run(env.flow.on_file(incoming("photo", caption=caption)))
    assert [m for m in env.messenger.messages if m.buttons] == []
    assert env.service.calls[0]["subject"] == subject
    assert "Using the subject from your caption" in env.messenger.messages[-1].text


def test_a_caption_with_an_unknown_but_valid_name_offers_to_add_it(env):
    run(env.flow.on_file(incoming("photo", caption="Astronomy")))
    q = env.question_message
    assert "\"Astronomy\" in the caption is not a subject I know" in q.text and env.service.calls == []
    assert "Add 'Astronomy' as a new subject" in q.labels
    env.tap("Add 'Astronomy' as a new subject")
    assert env.service.calls[0]["subject"] == "Astronomy"


@pytest.mark.parametrize("caption", ["CON", "x" * 60, "two\nlines", "Failed", "look at this"])
def test_a_caption_that_cannot_be_a_subject_just_shows_the_buttons(env, caption):
    run(env.flow.on_file(incoming("photo", caption=caption)))
    labels = env.question_message.labels
    assert "Math" in labels and not any(lbl.startswith("Add '") and caption in ("CON", "x" * 60, "two\nlines", "Failed") for lbl in labels)
    assert env.service.calls == []


# ---- several photos sent together -------------------------------------------------------------------------

def three_photos(env, group="G1", caption=None):
    for n in range(3):
        run(env.flow.on_file(incoming("photo", group=group, caption=caption if n == 0 else None)))


def test_photos_sent_together_get_one_question_after_they_settle(env):
    three_photos(env)
    assert [m for m in env.messenger.messages if m.buttons] == [] and len(env.spool_files()) == 3     # saved, not yet asked
    run(env.flow.tick())
    assert [m for m in env.messenger.messages if m.buttons] == []                                   # still settling
    env.settle()
    questions = [m for m in env.messenger.messages if m.buttons]
    assert len(questions) == 1 and questions[0].text.startswith("3 photos: same subject for all?")
    assert "Make these one document" in questions[0].labels


def test_each_photo_of_a_group_becomes_its_own_job_by_default(env):
    three_photos(env)
    env.settle()
    env.tap("English")
    assert len(env.service.calls) == 3 and all(c["subject"] == "English" and c["is_album"] is False and len(c["paths"]) == 1 for c in env.service.calls)
    assert [c["paths"][0] for c in env.service.calls] == [Path(i.spool_path) for i in sorted(env.state.all_items(), key=lambda i: i.message_id)]
    assert "Queued 3 documents (subject: English)" in env.question_message.now_text


def test_make_these_one_document_makes_an_album_only_when_tapped(env):
    three_photos(env)
    env.settle()
    env.tap("Make these one document")
    q = env.question_message
    assert q.now_text.startswith("These 3 photos will be ONE document. Which subject is it?") and "Keep separate" in q.now_labels
    assert "Make these one document" not in q.now_labels and env.service.calls == []
    env.tap("Science")
    (call,) = env.service.calls
    assert call["is_album"] is True and len(call["paths"]) == 3 and call["source_kind"] == ["photo"] * 3 and call["subject"] == "Science"
    assert "Queued 3 photos as one document" in q.now_text


def test_keep_separate_goes_back_to_one_job_per_photo(env):
    three_photos(env)
    env.settle()
    env.tap("Make these one document")
    env.tap("Keep separate")
    assert env.question_message.now_text.startswith("3 photos: same subject for all?")
    env.tap("Math")
    assert len(env.service.calls) == 3 and not any(c["is_album"] for c in env.service.calls)


def test_a_group_with_a_pdf_cannot_be_made_one_document(env):
    run(env.flow.on_file(incoming("photo", group="G2")))
    run(env.flow.on_file(incoming("pdf", group="G2", name="a.pdf", file_id="P")))
    env.messenger.files["P"] = PDF
    env.settle()
    q = env.question_message
    assert q.text.startswith("2 files: same subject for all?") and "Make these one document" not in q.labels


def test_a_caption_on_the_first_photo_of_a_group_applies_to_all(env):
    three_photos(env, caption="Science")
    env.settle()
    assert len(env.service.calls) == 3 and {c["subject"] for c in env.service.calls} == {"Science"}
    assert [m for m in env.messenger.messages if m.buttons] == []


def test_a_late_photo_of_the_same_group_joins_the_open_question(env):
    three_photos(env)
    env.settle()
    run(env.flow.on_file(incoming("photo", group="G1")))
    q = env.question_message
    assert q.now_text.startswith("4 photos: same subject for all?")
    env.tap("Math")
    assert len(env.service.calls) == 4


def test_two_groups_are_asked_separately(env):
    run(env.flow.on_file(incoming("photo", group="A")))
    run(env.flow.on_file(incoming("photo", group="B")))
    env.settle()
    assert len([m for m in env.messenger.messages if m.buttons]) == 2


def test_separate_photos_each_get_their_own_question(env):
    run(env.flow.on_file(incoming("photo")))
    run(env.flow.on_file(incoming("photo")))
    assert len([m for m in env.messenger.messages if m.buttons]) == 2


# ---- "Other..." -------------------------------------------------------------------------------------------

def test_other_asks_for_a_name_and_the_next_text_is_the_subject(env):
    run(env.flow.on_file(incoming("photo")))
    env.tap("Other...")
    q = env.question_message
    assert q.now_text.startswith("Type the name of the new subject") and q.now_labels == ["Back", "Auto-detect"]
    run(env.flow.on_text(42, "astronomy club"))
    assert env.service.calls[0]["subject"] == "astronomy club"


def test_an_existing_subject_typed_in_other_uses_its_proper_spelling(env):
    run(env.flow.on_file(incoming("photo")))
    env.tap("Other...")
    run(env.flow.on_text(42, "SCIENCE"))
    assert env.service.calls[0]["subject"] == "Science"


@pytest.mark.parametrize("bad", ["CON", "x" * 60, "   ", "Failed"])
def test_a_bad_new_subject_is_explained_and_asked_for_again(env, bad):
    run(env.flow.on_file(incoming("photo")))
    env.tap("Other...")
    run(env.flow.on_text(42, bad))
    assert env.service.calls == [] and "Type another name" in env.messenger.messages[-1].text
    run(env.flow.on_text(42, "History"))
    assert env.service.calls[0]["subject"] == "History"


def test_back_returns_to_the_subject_buttons(env):
    run(env.flow.on_file(incoming("photo")))
    env.tap("Other...")
    env.tap("Back")
    q = env.question_message
    assert q.now_text.startswith("Which subject is this photo for?") and "Math" in q.now_labels
    run(env.flow.on_text(42, "Chemistry"))                                  # no longer waiting: only a hint
    assert env.service.calls == [] and "Send me photos or PDFs" in env.messenger.messages[-1].text


def test_text_with_nothing_pending_gets_a_short_hint(env):
    run(env.flow.on_text(42, "hello"))
    assert "photos, images sent as a file, and PDFs" in env.messenger.messages[-1].text and env.service.calls == []


def test_the_latest_other_question_receives_the_typed_name(env):
    run(env.flow.on_file(incoming("photo")))
    first = env.question_message
    run(env.flow.on_file(incoming("photo")))
    second = [m for m in env.messenger.messages if m.buttons][1]
    env.tap("Other...", first)
    env.clock.advance(5)
    env.tap("Other...", second)
    run(env.flow.on_text(42, "Art"))
    stages = {q.message_id: q.stage for q in env.state.all_questions()}
    assert len(env.service.calls) == 1 and stages[second.id] == CLOSED and stages[first.id] == "awaiting_other"


# ---- time and restarts ------------------------------------------------------------------------------------

def test_no_answer_in_five_minutes_falls_back_to_auto_detect_and_says_so(env):
    run(env.flow.on_file(incoming("photo")))
    env.clock.advance(ANSWER_SECONDS - 1)
    run(env.flow.tick())
    assert env.service.calls == []
    env.clock.advance(2)
    run(env.flow.tick())
    (call,) = env.service.calls
    assert call["subject"] is None
    q = env.question_message
    assert "No answer in 5 minutes" in q.now_text and "auto-detect" in q.now_text and q.now_labels == []
    run(env.flow.tick())
    assert len(env.service.calls) == 1                                       # only once


def test_a_pending_question_survives_a_restart_and_can_still_be_answered(env):
    run(env.flow.on_file(incoming("photo")))
    q = env.question_message
    restarted = env.make_flow()
    run(restarted.start())
    assert len([m for m in env.messenger.messages if m.buttons]) == 1          # not asked a second time
    run(restarted.on_callback(42, q.id, q.data_of("Math")))
    assert env.service.calls[0]["subject"] == "Math"


def test_a_question_that_ran_out_of_time_while_the_app_was_off_is_settled_at_start(env):
    three_photos(env)
    env.settle()
    env.clock.advance(ANSWER_SECONDS * 3)
    restarted = env.make_flow()
    run(restarted.start())
    assert len(env.service.calls) == 3 and all(c["subject"] is None for c in env.service.calls)


def test_a_group_that_was_saved_but_never_asked_is_asked_after_a_restart(env):
    three_photos(env)                                                           # saved, the app stops before the settle time passes
    restarted = env.make_flow()
    run(restarted.start())
    env.clock.advance(SETTLE_SECONDS + 1)
    run(restarted.tick())
    assert len([m for m in env.messenger.messages if m.buttons]) == 1


def test_items_that_were_asked_but_whose_question_was_lost_are_asked_again(env):
    run(env.flow.on_file(incoming("photo")))
    (question,) = env.state.all_questions()
    question.stage = CLOSED                                                     # the app died after closing it, before submitting
    env.state.save_question(question)
    run(env.make_flow().start())
    assert len([m for m in env.messenger.messages if m.buttons]) == 2 and env.service.calls == []


def test_a_download_interrupted_by_a_restart_is_retried(env):
    run(env.flow.on_file(incoming("photo")))
    item = env.state.all_items()[0]
    item.stage, item.spool_path, item.question_id, item.attempts = DOWNLOADING, None, None, 0
    env.state.save_item(item)
    for q in env.state.all_questions():
        env.state.delete_question(q.id)
    restarted = env.make_flow()
    run(restarted.start())
    assert env.state.all_items()[0].stage in (ASKING,) and len(env.spool_files()) == 2


def test_start_removes_half_written_spool_files(env):
    folder = env.library / SPOOL_FOLDER
    folder.mkdir(parents=True)
    (folder / ".2026-10-18_x.jpg.partial").write_bytes(b"half")
    run(env.flow.start())
    assert list(folder.iterdir()) == []


# ---- what is refused or goes wrong ------------------------------------------------------------------------

def test_a_file_over_twenty_megabytes_is_refused_clearly_and_nothing_is_saved(env):
    run(env.flow.on_file(incoming("pdf", size=DOWNLOAD_LIMIT_BYTES + 1, name="big.pdf")))
    assert "too big" in env.messenger.messages[-1].text and "20 MB" in env.messenger.messages[-1].text and "(20.0 MB;" in env.messenger.messages[-1].text
    assert env.spool_files() == [] and env.messenger.downloads == [] and env.state.all_items() == []


def test_a_file_exactly_at_the_limit_is_accepted(env):
    run(env.flow.on_file(incoming("pdf", size=DOWNLOAD_LIMIT_BYTES, name="ok.pdf")))
    assert len(env.spool_files()) == 1


def test_an_unsupported_file_is_declined_in_plain_words(env):
    run(env.flow.on_file(incoming("unsupported", what="a Word document")))
    text = env.messenger.messages[-1].text
    assert "photos, images sent as a file, and PDFs" in text and "a Word document" in text and "Nothing was saved" in text
    assert env.spool_files() == [] and env.messenger.downloads == []


def test_telegram_refusing_a_big_download_is_reported_as_too_big(env):
    env.messenger.files["BIG"] = TooBigError("file is too big")
    run(env.flow.on_file(incoming("photo", file_id="BIG", size=None)))
    assert "too big" in env.messenger.messages[-1].text and env.spool_files() == []
    assert env.state.all_items()[0].stage == DONE


def test_a_failed_download_is_retried_and_the_student_told_once(env):
    env.messenger.files["F"] = DownloadError("network down")
    run(env.flow.on_file(incoming("photo", file_id="F")))
    assert env.spool_files() == [] and "couldn't download" in env.messenger.messages[-1].text and len(env.messenger.messages) == 1
    run(env.flow.tick())
    assert len(env.messenger.downloads) == 1                                    # not before the retry time
    env.clock.advance(RETRY_SECONDS + 1)
    run(env.flow.tick())
    assert len(env.messenger.downloads) == 2 and len(env.messenger.messages) == 1  # the second failure is silent
    env.messenger.files["F"] = JPEG + b"ok"
    env.clock.advance(RETRY_SECONDS + 1)
    run(env.flow.tick())
    assert len(env.spool_files()) == 1 and len([m for m in env.messenger.messages if m.buttons]) == 1


def test_a_download_that_keeps_failing_is_given_up_with_a_message(env):
    env.messenger.files["F"] = DownloadError("network down")
    run(env.flow.on_file(incoming("photo", file_id="F")))
    for _ in range(MAX_DOWNLOAD_ATTEMPTS):
        env.clock.advance(RETRY_SECONDS + 1)
        run(env.flow.tick())
    assert "Please send it again" in env.messenger.messages[-1].text and env.state.all_items()[0].stage == DONE
    n = len(env.messenger.downloads)
    env.clock.advance(RETRY_SECONDS * 5)
    run(env.flow.tick())
    assert len(env.messenger.downloads) == n


def test_the_same_update_delivered_twice_is_taken_once(env):
    inc = incoming("photo", message_id=77, unique="UNIQ")
    run(env.flow.on_file(inc))
    run(env.flow.on_file(inc))
    assert len(env.spool_files()) == 1 and len(env.state.all_items()) == 1 and len([m for m in env.messenger.messages if m.buttons]) == 1


def test_buttons_from_an_old_or_foreign_question_are_ignored(env):
    run(env.flow.on_file(incoming("photo")))
    q = env.question_message
    assert run(env.flow.on_callback(42, q.id, "s:nosuch:0")) == "That question is already answered."
    assert run(env.flow.on_callback(99, q.id, q.data_of("Math"))) == "That question is already answered."       # another chat
    assert run(env.flow.on_callback(42, q.id, "s:" + env.state.all_questions()[0].id + ":99")) == "That button is out of date."
    assert env.service.calls == []


def test_when_the_service_refuses_the_files_stay_safe_and_the_student_is_told(env):
    run(env.flow.on_file(incoming("photo")))
    env.service.error = StudyBuddyError("The ingest service is stopping and takes no new documents.")
    env.tap("Math")
    assert "could not be queued" in env.question_message.now_text and "_Unsorted" in env.question_message.now_text
    assert len(env.spool_files()) == 1 and env.state.all_items()[0].stage == ASKING


# ---- the messages about results ---------------------------------------------------------------------------

def submitted(env, **kw):
    run(env.flow.on_file(incoming("photo", **kw)))
    env.tap("Science")
    return env.service.calls[-1]["job_id"]


def result(job_id, **kw):
    base = dict(status=INDEXED, job_id=job_id, subject="Science", subject_source="user", title="Photosynthesis Worksheet", page_count=1,
                final_path="C:/lib/Science/2026-10-18_0930_Science_photosynthesis-worksheet.jpg")
    base.update(kw)
    return IngestResult(**base)


def test_an_indexed_document_is_reported_with_its_name_subject_pages_and_the_handwriting_warning(env):
    job = submitted(env)
    run(env.flow.on_result(result(job)))
    text = env.messenger.messages[-1].text
    assert "Filed: 2026-10-18_0930_Science_photosynthesis-worksheet.jpg" in text and "Subject: Science" in text
    assert "Title: Photosynthesis Worksheet" in text and "1 page." in text and HANDWRITING in text
    assert env.messenger.messages[-1].reply_to is not None and env.state.all_items()[0].stage == DONE


def test_an_auto_detected_subject_is_labelled_and_a_fallback_explained(env):
    run(env.flow.on_result(result(submitted(env), subject_source="auto")))
    assert "Subject: Science (auto-detected)" in env.messenger.messages[-1].text
    run(env.flow.on_result(result(submitted(env), subject="Unsorted", subject_source="fallback")))
    assert "Unsorted (I couldn't tell the subject; change it in the Library tab)" in env.messenger.messages[-1].text


def test_flagged_pages_are_listed_with_reasons_and_a_hint_when_a_resend_would_help(env):
    flagged = [FlaggedPage(2, 0.4, ["many_illegible"], ["many words are unreadable"], True),
               FlaggedPage(3, 0.7, ["glare"], ["there is glare on the photo"], False)]
    run(env.flow.on_result(result(submitted(env), page_count=3, flagged_pages=flagged)))
    text = env.messenger.messages[-1].text
    assert "3 pages." in text and "Check page 2: many words are unreadable." in text and "Note: page 3: there is glare on the photo." in text
    assert "send it as a file (full resolution)" in text and HANDWRITING in text


def test_a_document_with_nothing_flagged_has_no_resend_hint(env):
    run(env.flow.on_result(result(submitted(env))))
    assert "full resolution" not in env.messenger.messages[-1].text and HANDWRITING in env.messenger.messages[-1].text


def test_warnings_from_ingest_are_passed_on(env):
    run(env.flow.on_result(result(submitted(env), warnings=["no text could be read, so the title and summary are generic"])))
    assert "Note: no text could be read, so the title and summary are generic." in env.messenger.messages[-1].text


def test_a_duplicate_is_explained_in_plain_language(env):
    job = submitted(env)
    run(env.flow.on_result(IngestResult(status=DUPLICATE, job_id=job, page_count=2, duplicate_dir="C:/data/duplicates",
                                        existing={"title": "Earth notes", "subject": "Science"})))
    text = env.messenger.messages[-1].text
    assert "I already have this one: 'Earth notes' (Science, 2 pages)" in text and "did not add it again" in text and "C:/data/duplicates" in text


def test_a_failure_says_why_and_where_the_file_is(env):
    job = submitted(env)
    run(env.flow.on_result(IngestResult(status=FAILED, job_id=job, error="the image is damaged or unreadable", failed_dir="C:/data/failed/job-1")))
    text = env.messenger.messages[-1].text
    assert text.startswith("I couldn't read this one: the image is damaged or unreadable.") and "safe in C:/data/failed/job-1" in text
    assert env.state.all_items()[0].stage == DONE


def test_a_deferred_job_tells_the_student_it_will_retry_and_stays_open(env):
    job = submitted(env)
    run(env.flow.on_result(IngestResult(status=DEFERRED, job_id=job, retry_in_seconds=120)))
    assert "isn't reachable" in env.messenger.messages[-1].text and "in about 2 minute(s)" in env.messenger.messages[-1].text
    assert env.state.all_items()[0].stage == SUBMITTED
    run(env.flow.on_result(result(job)))                                         # and the final result still arrives
    assert "Filed:" in env.messenger.messages[-1].text


def test_results_of_jobs_the_bot_did_not_submit_are_ignored(env):
    submitted(env)
    before = len(env.messenger.messages)
    run(env.flow.on_result(result("someone-elses-job")))
    assert len(env.messenger.messages) == before


def test_an_album_gets_one_message_for_all_its_photos(env):
    three_photos(env)
    env.settle()
    env.tap("Make these one document")
    env.tap("Science")
    job = env.service.calls[0]["job_id"]
    run(env.flow.on_result(result(job, page_count=3, final_path="C:/lib/Science/2026-10-18_0930_Science_album")))
    assert len([m for m in env.messenger.messages if m.text.startswith("Filed:")]) == 1
    assert "3 pages." in env.messenger.messages[-1].text and all(i.stage == DONE for i in env.state.all_items())


def test_a_result_that_arrived_while_the_bot_was_off_is_acknowledged_at_start(env):
    job = submitted(env)
    env.service.jobs.ids.remove(job)                                            # the job file is gone: it finished unseen
    run(env.make_flow().start())
    assert "processed while I was switched off" in env.messenger.messages[-1].text and env.state.all_items()[0].stage == DONE


def test_a_job_still_queued_at_start_is_left_alone(env):
    submitted(env)
    before = len(env.messenger.messages)
    run(env.make_flow().start())
    assert len(env.messenger.messages) == before and env.state.all_items()[0].stage == SUBMITTED
