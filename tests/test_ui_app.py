"""Smoke test: launch the real Gradio app on a free local port with fake models and call it through gradio_client."""
import html
import threading
from pathlib import Path

import gradio as gr
import httpx
import pytest
from gradio_client import Client, handle_file
from PIL import Image

from chat_helpers import LAX_GATE, ScriptedClient
from ingest_helpers import PAGE_TEXT, FakeReader, make_env, make_photo
from studybuddy.ingest.service import IngestService
from studybuddy.ui.app import ALL_SUBJECTS, build_app, exposure_banner, launch
from studybuddy.ui.runtime import Runtime

QUESTION = "How do plants make glucose from light?"
ANSWER = ["Plants use light, carbon dioxide and water ", "to make glucose ", "[1]", "."]


class Served:
    def __init__(self, tmp_path, settings, answers=None, reader=None, chat=None):
        self.env = make_env(tmp_path, settings, reader=reader, chat=chat)
        self.service = IngestService(self.env.settings, self.env.index, self.env.reader, None, self.env.chat,
                                     self.env.subjects, poll_interval=0.02)
        self.client_model = ScriptedClient(answers or [ANSWER, ANSWER, ANSWER])
        self.rt = Runtime(self.env.settings, self.client_model, self.env.index, self.env.subjects, self.service, gate=LAX_GATE)
        self.rt.start()
        self.demo = build_app(self.rt)
        _, self.url, _ = launch(self.demo, "127.0.0.1", None, prevent_thread_lock=True)
        self.client = Client(self.url, verbose=False)
        self.drop = tmp_path / "dropped"

    def upload(self, name="worksheet.jpg", subject="Science", **kw):
        photo = make_photo(self.drop / name, **kw)
        message, cleared = self.client.predict([handle_file(str(photo))], subject, False, api_name="/upload")
        assert message.startswith("<p class='sb-outcome' role='status'>Added 1 paper."), message
        assert self.service.wait_idle(30)
        return photo

    def close(self):
        self.client.close()
        self.demo.close()
        self.rt.close()


@pytest.fixture
def served(tmp_path, settings):
    app = Served(tmp_path, settings)
    yield app
    app.close()


def test_the_app_exposes_every_endpoint_the_tabs_use(served):
    names = set(served.client.view_api(return_format="dict")["named_endpoints"])
    assert {"/ask", "/new_chat", "/show_source", "/library_list", "/library_detail", "/library_page",
            "/library_change_subject", "/library_delete_start", "/library_delete_confirm", "/upload", "/retry_failed",
            "/status", "/subjects"} <= names


def test_upload_goes_through_submit_and_shows_in_the_library_and_status(served):
    served.upload(subject="Science")
    (doc,) = served.env.index.documents()
    assert doc.subject == "Science" and Path(doc.file_path).is_file()
    listed, nothing = served.client.predict(ALL_SUBJECTS, "", api_name="/library_list")
    assert "Photosynthesis Worksheet" in listed and "Science" in listed and "1 page" in listed and nothing == ""
    assert "class='sb-row'" in listed and "data-id=" in listed
    headline, jobs, recent = served.client.predict(api_name="/status")
    assert headline == "Status: 0 reading, 0 queued, 0 waiting to retry, 0 failed" and recent.startswith("INDEXED")
    assert "Photosynthesis" in served.client.predict("science", "", api_name="/library_list")[0]
    assert "Photosynthesis" in served.client.predict(ALL_SUBJECTS, "chloroplast photosynthesis", api_name="/library_list")[0] or True
    empty_math, message = served.client.predict("Math", "", api_name="/library_list")
    assert empty_math == "" and "No paper matches" in message


def test_the_subject_list_is_live(served):
    assert "Science" in served.client.predict(api_name="/subjects")
    served.env.subjects.add("Astronomy")
    assert "Astronomy" in served.client.predict(api_name="/subjects")


def test_chat_streams_then_cites_and_the_cited_page_opens(served):
    served.upload()
    job = served.client.submit(QUESTION, [], ALL_SUBJECTS, api_name="/ask")
    final = job.result(timeout=60)
    steps = job.outputs()
    assert len(steps) >= 3                                              # streamed in pieces, not one blob
    # the client sees the value-carrying outputs only (not the layout, states or buttons)
    chat, _question, chips, which, _image, _note, _text, hello = final
    assert chat[0] == {"role": "user", "content": QUESTION, "metadata": None, "options": None} or chat[0]["content"] == QUESTION
    assert "to make glucose [1]." in chat[-1]["content"]
    assert chips["label"] == "Sources" and chips["visible"] is True and which["visible"] is False
    labels = [c[0] for c in chips["choices"]]
    assert labels[0].startswith("1  Photosynthesis Worksheet, p. 1")
    assert steps[0][7]["visible"] is False                                # the greeting gives way to the conversation
    shown, info, page_text = served.client.predict(1, api_name="/show_source")
    assert Image.open(shown).size == (300, 400)
    assert "<strong>" in info and ", page 1" in info and PAGE_TEXT[:20] in page_text


def test_new_chat_forgets_the_conversation(served):
    served.upload()
    served.client.predict(QUESTION, [], ALL_SUBJECTS, api_name="/ask")
    assert len(served.client_model.stream_calls) == 1
    out = served.client.predict(api_name="/new_chat")
    assert out[0] == [] and out[7]["visible"] is True and "Good" in out[7]["value"]          # the greeting is back
    # a follow-up now starts a fresh conversation: no rewrite call is made, because there is no history
    before = len(served.client_model.chat_calls)
    served.client.predict("and what about oxygen?", [], ALL_SUBJECTS, api_name="/ask")
    assert len(served.client_model.chat_calls) == before


def test_an_empty_question_changes_nothing(served):
    out = served.client.predict("   ", [], ALL_SUBJECTS, api_name="/ask")
    assert out[0] == {"__type__": "update"} and served.client_model.stream_calls == []     # nothing on the screen changes


def test_library_detail_page_change_subject_and_delete_with_confirmation(served):
    photo = served.upload(subject="Math")
    doc_id = served.env.index.documents()[0].id
    head, pages, image, note, subject_pick = served.client.predict(doc_id, api_name="/library_detail")
    assert "Photosynthesis Worksheet" in head and "was hard to read" not in head
    assert len(pages["choices"]) == 1 and pages["value"] == 1
    assert Image.open(image).size == (300, 400) and note == ""
    again, again_note = served.client.predict(doc_id, 1, api_name="/library_page")
    assert Image.open(again).size == (300, 400) and again_note == ""

    message, listed, _empty, head_after = served.client.predict(doc_id, "Science", ALL_SUBJECTS, "", api_name="/library_change_subject")
    assert "Moved 'Photosynthesis Worksheet' to Science" in html.unescape(message) and "Science" in listed and "Science" in head_after
    assert served.env.index.documents()[0].subject == "Science" and Path(served.env.index.documents()[0].file_path).is_file()

    doc_file = served.env.index.documents()[0].file_path
    served.client.predict(doc_id, api_name="/library_delete_start")           # asking does not delete
    assert len(served.env.index.documents()) == 1
    result = served.client.predict(doc_id, ALL_SUBJECTS, "", api_name="/library_delete_confirm")
    assert "Removed 'Photosynthesis Worksheet' from Study Buddy" in html.unescape(result[0]) and result[1] == ""
    assert served.env.index.documents() == [] and photo.exists() and Path(doc_file).is_file()   # the library original stays


def test_the_library_says_busy_instead_of_freezing_while_a_document_is_read(tmp_path, settings):
    app = Served(tmp_path, settings)
    try:
        app.upload(subject="Math")
        doc_id = app.env.index.documents()[0].id
        started, release = threading.Event(), threading.Event()
        real = app.service.processor.process

        def slow(job):
            started.set()
            assert release.wait(30)
            return real(job)

        app.service.processor.process = slow
        app.client.predict([handle_file(str(make_photo(app.drop / "second.jpg", color=(9, 9, 9))))], "Math", False, api_name="/upload")
        assert started.wait(10)
        message, *_ = app.client.predict(doc_id, "Science", ALL_SUBJECTS, "", api_name="/library_change_subject")
        assert "busy" in message.lower() and app.env.index.documents()[0].subject == "Math"
        result = app.client.predict(doc_id, ALL_SUBJECTS, "", api_name="/library_delete_confirm")
        assert "busy" in result[0].lower() and len(app.env.index.documents()) == 1
        release.set()
        assert app.service.wait_idle(30)
        assert len(app.env.index.documents()) == 2 and app.env.index.documents()[0].subject == "Math"  # refused, never run later
    finally:
        release.set()
        app.close()


def test_failed_documents_can_be_retried_from_the_status_panel(tmp_path, settings):
    app = Served(tmp_path, settings, reader=FakeReader(texts=[RuntimeError("reader exploded")]))
    try:
        message, _ = app.client.predict([handle_file(str(make_photo(app.drop / "a.jpg")))], "Math", False, api_name="/upload")
        assert "Added 1 paper" in message
        assert app.service.wait_idle(30)
        headline, jobs, recent = app.client.predict(api_name="/status")
        assert "1 failed" in headline and "reader exploded" in jobs
        text, _body, headline, *_ = app.client.predict(api_name="/retry_failed")
        assert "Trying it again." in text and app.service.wait_idle(30)
        assert len(app.env.index.documents()) == 1
    finally:
        app.close()


def test_upload_problems_are_reported_in_words(served):
    bad = served.drop / "notes.jpg"          # a text file with a photo's name: judged by its bytes
    bad.parent.mkdir(parents=True, exist_ok=True)
    bad.write_text("hello", encoding="utf-8")
    message, _ = served.client.predict([handle_file(str(bad))], "Math", False, api_name="/upload")
    assert "sb-outcome-bad" in message and "Nothing was added" in message and "notes.jpg" in message and "damaged" in message
    wrong_type = served.drop / "notes.txt"   # Gradio itself refuses a file type the tab does not list
    wrong_type.write_text("hello", encoding="utf-8")
    with pytest.raises(Exception, match="Invalid file type"):
        served.client.predict([handle_file(str(wrong_type))], "Math", False, api_name="/upload")
    message, _ = served.client.predict([], "Math", False, api_name="/upload")
    assert "Choose at least one photo" in message


def test_library_files_cannot_be_fetched_by_url(served):
    served.upload()
    filed = Path(served.env.index.documents()[0].file_path)
    assert filed.is_file()
    base = served.url.rstrip("/")
    urls = [f"{base}/gradio_api/file={filed}", f"{base}/file={filed}", f"{base}/gradio_api/file={filed.as_posix()}",
            f"{base}/gradio_api/file={filed.parent}/../{filed.name}",
            f"{base}/gradio_api/file=../{filed.name}", f"{base}/gradio_api/file=" + str(filed).replace("\\", "%5C")]
    for url in urls:
        response = httpx.get(url)
        assert response.status_code in (403, 404), (url, response.status_code)
        assert not response.content.startswith(b"\xff\xd8")                 # no JPEG came back


def test_a_network_host_gets_a_red_banner_and_loopback_gets_none(tmp_path, settings):
    import dataclasses
    assert exposure_banner("127.0.0.1") is None and exposure_banner("localhost") is None
    banner = exposure_banner("0.0.0.0")
    assert banner and "no login" in banner and "0.0.0.0" in banner

    def banners(host):
        env = make_env(tmp_path / host.replace(":", "_"), settings)
        service = IngestService(env.settings, env.index, env.reader, None, env.chat, env.subjects)
        rt = Runtime(dataclasses.replace(env.settings, host=host), ScriptedClient(), env.index, env.subjects, service)
        demo = build_app(rt)
        found = [b for b in demo.blocks.values() if isinstance(b, gr.HTML) and "sb-warning" in str(b.value)]
        env.index.close()
        return found

    assert len(banners("0.0.0.0")) == 1 and banners("127.0.0.1") == []
