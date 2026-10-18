"""The Gradio app: layout and wiring only. Every decision lives in the modules next to this one, where it is tested.

Pages are returned to the browser as decoded images (see files.py); no library path is ever given to Gradio, and the
app is launched without `allowed_paths`, so Gradio cannot serve a file from the library by URL.
"""
from __future__ import annotations

import html

import gradio as gr
from starlette.middleware import Middleware

from ..chat.answers import NEEDS_CHOICE, NOT_IN_PAPERS_STATUS
from ..config import network_exposure
from ..errors import StudyBuddyError
from .chat import DONE, source_choices, source_details, sources_caption, stream_turn
from .library import filter_rows
from .markup import activity_html, esc, greeting, library_empty, paper_head_html, papers_html, suggest_questions
from .offline import StripExternalScripts
from .runtime import Runtime
from .theme import THEME_JS, build_theme, css
from .uploads import AUTO_DETECT, submit_uploads

ALL_SUBJECTS = "All subjects"
TIMER_SECONDS = 3.0
MAX_UPLOAD = "100mb"
UPLOAD_TYPES = [".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff", ".heic", ".pdf"]


def exposure_banner(host: str) -> str | None:
    warning = network_exposure(host)
    if warning is None:
        return None
    return (f"<div class='sb-warning' role='alert'>This app can be reached from the network ({html.escape(host)}) and has no login. "
            f"Anyone who can connect can read, change and delete your study library. "
            f"Set SB_HOST=127.0.0.1 and restart unless you meant this.</div>")


def _subject_filter(value: str | None) -> str | None:
    return None if not value or value == ALL_SUBJECTS else value


def build_app(rt: Runtime) -> gr.Blocks:
    def subject_names() -> list[str]:
        try:
            rt.subjects.reload()     # pick up a subject added by hand or by another part of the app
        except StudyBuddyError as exc:
            gr.Warning(str(exc))
        return rt.subjects.choices()

    def refresh_filter(current):
        names = [ALL_SUBJECTS] + subject_names()
        return gr.update(choices=names, value=current if current in names else ALL_SUBJECTS)

    def refresh_choice(current):
        names = subject_names()
        return gr.update(choices=names, value=current if current in names else (names[0] if names else None))

    def refresh_upload_subject(current):
        names = [AUTO_DETECT] + subject_names()
        return gr.update(choices=names, value=current if current in names else AUTO_DETECT)

    try:
        start_names = rt.subjects.choices()
    except StudyBuddyError:
        start_names = []

    with gr.Blocks(title="Study Buddy", theme=build_theme(), css=css(), js=THEME_JS, fill_width=True) as demo, \
            gr.Column(elem_id="sb-root"):
        banner = exposure_banner(rt.settings.host)
        if banner:
            gr.HTML(banner)
        with gr.Row(elem_id="sb-header"):
            gr.HTML("<span id='sb-wordmark'>Study Buddy</span>", padding=False)
            with gr.Row(elem_id="sb-tools"):
                pill = gr.Button("", elem_id="sb-pill", variant="secondary", visible=False)
                gr.Button("◐", elem_id="sb-theme", variant="secondary", min_width=44)

        # ---- activity: a small indicator in the corner that opens a plain panel ----------------------
        activity_open = gr.State(False)
        with gr.Column(elem_id="sb-activity", visible=False) as activity_panel:
            gr.HTML("<h2>Activity</h2>", padding=False)
            activity_body = gr.HTML(padding=False)
            retry_btn = gr.Button("Retry", visible=False, variant="primary", elem_id="sb-retry")
            retry_msg = gr.HTML(elem_id="sb-retry-msg", padding=False)
            close_btn = gr.Button("Close", size="sm", variant="secondary", elem_id="sb-close")
            with gr.Accordion("Technical details", open=False, elem_id="sb-tech"):
                headline = gr.Markdown()
                jobs_box = gr.Textbox(label="Queue", lines=4, interactive=False)
                recent_box = gr.Textbox(label="Latest results", lines=6, interactive=False)
        timer = gr.Timer(TIMER_SECONDS)

        def status():
            snap = rt.board.snapshot()
            return snap.headline(), snap.jobs_text(), snap.recent_text()

        def activity_view(is_open):
            """The corner indicator, the panel and its contents. Nothing to say means no indicator and a closed panel."""
            snap = rt.board.snapshot()
            text = snap.pill()
            show = bool(is_open and text)
            return (gr.update(value=text, visible=bool(text)), gr.update(visible=show), activity_html(snap),
                    gr.update(visible=snap.retry_possible()), snap.headline(), snap.jobs_text(), snap.recent_text(), show)

        def retry_failed(is_open):
            outcome = rt.board.retry_failed()
            return (say(outcome), *activity_view(is_open))

        status_outputs = [headline, jobs_box, recent_box]
        view_outputs = [pill, activity_panel, activity_body, retry_btn, headline, jobs_box, recent_box, activity_open]
        timer.tick(activity_view, [activity_open], view_outputs, api_name=False, show_progress="hidden")
        pill.click(lambda is_open: activity_view(not is_open), [activity_open], view_outputs, api_name=False, show_progress="hidden")
        close_btn.click(lambda: activity_view(False), None, view_outputs, api_name=False, show_progress="hidden")
        retry_btn.click(retry_failed, [activity_open], [retry_msg, *view_outputs], api_name="retry_failed")

        with gr.Tabs(elem_id="sb-tabs") as tabs:
            # ---- ask -------------------------------------------------------------------------------
            with gr.Tab("Ask", id="ask") as chat_tab:
                session_state = gr.State(None)
                answer_state = gr.State(None)
                with gr.Column(elem_id="sb-ask"):
                    hello = gr.HTML(greeting(), elem_id="sb-hello", padding=False)
                    with gr.Column(elem_id="sb-convo", visible=False) as convo:
                        chatbot = gr.Chatbot(type="messages", layout="bubble", show_label=False, container=False, height=None,
                                             max_height="62vh", min_height=0, sanitize_html=True, render_markdown=True,
                                             elem_id="sb-chat", group_consecutive_messages=False, autoscroll=True)
                        chips = gr.Radio([], label="Sources", elem_id="sb-chips", interactive=True, visible=False)
                        closest_btn = gr.Button("Show the closest pages", elem_id="sb-closest", size="sm", visible=False)
                        which = gr.Radio([], label="Pick one", elem_id="sb-which", interactive=True, visible=False)
                    with gr.Column(elem_id="sb-slip-col", visible=False) as slip_col, gr.Group(elem_id="sb-slip"):
                        page_image = gr.Image(type="pil", format="jpeg", interactive=False, show_label=False, container=False,
                                              elem_id="sb-slip-image", show_download_button=False, show_fullscreen_button=True)
                        slip_note = gr.HTML(elem_id="sb-slip-note", padding=False)
                        with gr.Accordion("Text read from this page", open=False, elem_id="sb-slip-text"):
                            page_text = gr.Textbox(show_label=False, lines=8, interactive=False, container=False)
                    with gr.Group(elem_id="sb-composer"):
                        question = gr.Textbox(placeholder="Ask about your papers", show_label=False, container=False, lines=1,
                                              max_lines=5, elem_id="sb-question", scale=5)
                        send_btn = gr.Button("Ask", variant="primary", elem_id="sb-send", scale=0, min_width=88)
                    with gr.Row(elem_id="sb-ask-tools"):
                        chat_subject = gr.Dropdown([ALL_SUBJECTS, *start_names], value=ALL_SUBJECTS, show_label=False, container=False,
                                                   interactive=True, allow_custom_value=True, elem_id="sb-subject", min_width=140)
                        new_btn = gr.Button("New chat", size="sm", elem_id="sb-new", variant="secondary", min_width=100)
                    with gr.Column(elem_id="sb-examples") as examples_box:
                        example_btns = [gr.Button("", elem_classes=["sb-example"], visible=False, variant="secondary") for _ in range(3)]
                        examples_note = gr.HTML(padding=False, elem_id="sb-examples-note")
                        add_first_btn = gr.Button("Add papers", visible=False, variant="primary", elem_id="sb-add-first")

                ASK_KEYS = ["chatbot", "question", "session", "answer", "chips", "which", "closest", "image", "note", "text",
                            "hello", "convo", "slip", "examples"]
                ask_parts = {"chatbot": chatbot, "question": question, "session": session_state, "answer": answer_state,
                             "chips": chips, "which": which, "closest": closest_btn, "image": page_image, "note": slip_note,
                             "text": page_text, "hello": hello, "convo": convo, "slip": slip_col, "examples": examples_box}
                ask_outputs = [ask_parts[k] for k in ASK_KEYS]

                def pack(**changes):
                    """One value per Ask output; anything not named stays as it is."""
                    return tuple(changes.get(key, gr.update()) for key in ASK_KEYS)

                def hidden(**extra):
                    return dict(chips=gr.update(choices=[], value=None, visible=False), which=gr.update(choices=[], value=None, visible=False),
                                closest=gr.update(visible=False), slip=gr.update(visible=False), image=None, note="", text="", **extra)

                def ask(text, history, subject, session, answer):
                    text = (text or "").strip()
                    history = list(history or [])
                    if not text:
                        gr.Warning("Type a question first.")
                        yield pack()
                        return
                    session = session or rt.new_session()
                    shown = history + [{"role": "user", "content": text}]
                    first = True
                    for update in stream_turn(session, text, _subject_filter(subject)):
                        message = {"role": "assistant", "content": update.text}
                        if update.phase == DONE and update.answer is not None:
                            yield pack(chatbot=shown + [message], question="", session=session, answer=update.answer,
                                       **answer_view(update.answer))
                        elif first:
                            yield pack(chatbot=shown + [message], question="", session=session, answer=None, hello=gr.update(visible=False),
                                       convo=gr.update(visible=True), examples=gr.update(visible=False), **hidden())
                        else:
                            yield pack(chatbot=shown + [message])
                        first = False

                def answer_view(ans):
                    """What sits under a finished answer: chips, the 'closest pages' button, or tappable paper titles."""
                    if ans.status == NEEDS_CHOICE and ans.choices:
                        return dict(which=gr.update(choices=[(title, str(n)) for n, title in enumerate(ans.choices, 1)], value=None, visible=True),
                                    chips=gr.update(choices=[], visible=False), closest=gr.update(visible=False))
                    choices = source_choices(ans)
                    if ans.status == NOT_IN_PAPERS_STATUS:
                        return dict(chips=gr.update(choices=choices, value=None, visible=False, label=sources_caption(ans)),
                                    closest=gr.update(visible=bool(choices)), which=gr.update(visible=False))
                    return dict(chips=gr.update(choices=choices, value=None, visible=bool(choices), label=sources_caption(ans)),
                                closest=gr.update(visible=False), which=gr.update(visible=False))

                ask_inputs = [question, chatbot, chat_subject, session_state, answer_state]
                send_btn.click(ask, ask_inputs, ask_outputs, api_name="ask", concurrency_limit=1, concurrency_id="chat",
                               show_progress="hidden")
                question.submit(ask, ask_inputs, ask_outputs, api_name=False, concurrency_limit=1, concurrency_id="chat",
                                show_progress="hidden")
                which.input(lambda number: number, [which], [question], api_name=False).then(
                    ask, ask_inputs, ask_outputs, api_name=False, concurrency_limit=1, concurrency_id="chat", show_progress="hidden")
                for example in example_btns:
                    example.click(lambda label: label, [example], [question], api_name=False).then(
                        ask, ask_inputs, ask_outputs, api_name=False, concurrency_limit=1, concurrency_id="chat", show_progress="hidden")
                closest_btn.click(lambda: (gr.update(visible=True), gr.update(visible=False)), None, [chips, closest_btn], api_name=False)

                def new_chat(session):
                    if session is not None:
                        session.new()
                    return pack(chatbot=[], question="", session=session, answer=None, hello=gr.update(visible=True, value=greeting()),
                                convo=gr.update(visible=False), examples=gr.update(visible=True), **hidden())

                new_btn.click(new_chat, [session_state], ask_outputs, api_name="new_chat")

                def show_source(answer, number):
                    if answer is None or number is None:
                        return None, "", "", gr.update(visible=False)
                    image, caption, text = source_details(answer, int(number), rt.guard)
                    return image, caption, text, gr.update(visible=True)

                chips.input(show_source, [answer_state, chips], [page_image, slip_note, page_text, slip_col], api_name="show_source")

                def example_updates():
                    questions = suggest_questions(rt.library.rows())
                    buttons = [gr.update(value=questions[i], visible=True) if i < len(questions) else gr.update(value="", visible=False)
                               for i in range(3)]
                    note = "" if questions else ("<p class='sb-muted'>You have not added any papers yet. Add a photo or a PDF, "
                                                 "or send a photo to your Telegram bot, then ask about it here.</p>")
                    return (*buttons, note, gr.update(visible=not questions))

                example_outputs = [*example_btns, examples_note, add_first_btn]
                chat_subject.focus(refresh_filter, [chat_subject], [chat_subject], api_name=False)
                chat_tab.select(lambda f: (refresh_filter(f), *example_updates()), [chat_subject], [chat_subject, *example_outputs],
                                api_name=False)

            # ---- library ---------------------------------------------------------------------------
            with gr.Tab("Library", id="library") as library_tab:
                with gr.Column(elem_id="sb-lib"):
                    with gr.Row(elem_id="sb-lib-bar"):
                        search = gr.Textbox(placeholder="Search your papers", show_label=False, container=False, lines=1, max_lines=1,
                                            elem_id="sb-search", scale=3)
                        lib_filter = gr.Dropdown([ALL_SUBJECTS, *start_names], value=ALL_SUBJECTS, show_label=False, container=False,
                                                 allow_custom_value=True, elem_id="sb-lib-subject", scale=1, min_width=160)
                    lib_message = gr.HTML(elem_id="sb-lib-message", padding=False)
                    with gr.Row(elem_id="sb-lib-body"):
                        with gr.Column(elem_id="sb-lib-list"):
                            papers = gr.HTML(elem_id="sb-papers", padding=False)
                            lib_empty = gr.HTML(elem_id="sb-lib-empty", padding=False)
                            add_empty_btn = gr.Button("Add papers", variant="primary", visible=False, elem_id="sb-add-empty")
                        with gr.Column(elem_id="sb-paper", visible=False) as paper_col:
                            back_btn = gr.Button("Back to papers", size="sm", elem_id="sb-back", variant="secondary")
                            paper_head = gr.HTML(elem_id="sb-paper-head", padding=False)
                            with gr.Accordion("More actions", open=False, elem_id="sb-more"):
                                new_subject = gr.Dropdown(start_names, label="Move to subject", allow_custom_value=True)
                                change_btn = gr.Button("Change subject")
                                remove_btn = gr.Button("Remove from Study Buddy", variant="stop")
                            with gr.Group(visible=False, elem_id="sb-confirm") as confirm_group:
                                gr.HTML("<p><strong>Remove from Study Buddy?</strong> Your original file stays in your folder.</p>", padding=False)
                                with gr.Row():
                                    confirm_btn = gr.Button("Remove", variant="stop")
                                    keep_btn = gr.Button("Keep it")
                            page_pick = gr.Radio([], label="Pages", elem_id="sb-pages", interactive=True)
                            lib_image = gr.Image(type="pil", format="jpeg", interactive=False, show_label=False, container=False,
                                                 elem_id="sb-paper-image", show_download_button=False, show_fullscreen_button=True)
                            lib_image_note = gr.HTML(elem_id="sb-paper-note", padding=False)
                selected_id = gr.Number(visible="hidden", elem_id="sb-pick", precision=0)

                def lib_list(filter_value, query="", selected=None):
                    """The list of papers for the current subject and search, plus what to show when there is none."""
                    rows = filter_rows(rt.library.rows(_subject_filter(filter_value)), query)
                    searching = bool((query or "").strip()) or _subject_filter(filter_value) is not None
                    return papers_html(rows, selected), ("" if rows else library_empty(searching)), gr.update(visible=not rows and not searching)

                def say(text, ok=True):
                    return f"<p class='sb-outcome{'' if ok else ' sb-outcome-bad'}' role='status'>{esc(text)}</p>" if text else ""

                def show_page(doc_id, page):
                    if not doc_id or not page:
                        return None, ""
                    try:
                        return rt.library.page_image(int(doc_id), int(page)), ""
                    except StudyBuddyError as exc:
                        return None, f"<p class='sb-note' role='note'>The page image is not available: {esc(exc)}</p>"

                doc_outputs = [paper_col, paper_head, page_pick, lib_image, lib_image_note, new_subject, confirm_group]

                def show_document(doc_id):
                    closed = gr.update(visible=False)
                    if not doc_id:
                        return closed, "", gr.update(choices=[], value=None), None, "", gr.update(), closed
                    detail = rt.library.detail(int(doc_id))
                    if detail is None:
                        return (closed, "", gr.update(choices=[], value=None), None, "", gr.update(), closed)
                    first_flagged = next((p.page_number for p in detail.pages if p.flagged), None)
                    page = first_flagged or (detail.pages[0].page_number if detail.pages else None)
                    image, note = show_page(doc_id, page) if page else (None, "")
                    return (gr.update(visible=True), paper_head_html(detail),
                            gr.update(choices=[(f"Page {p.page_number}", p.page_number) for p in detail.pages], value=page,
                                      visible=len(detail.pages) > 1),
                            image, note, gr.update(choices=subject_names(), value=detail.row.subject), closed)

                def change_subject(doc_id, subject, filter_value, query=""):
                    if not doc_id:
                        return say("Choose a paper first.", False), *lib_list(filter_value, query), gr.update()
                    outcome = rt.library.change_subject(int(doc_id), subject or "")
                    detail = rt.library.detail(int(doc_id))
                    return (say(outcome.message, outcome.ok), *lib_list(filter_value, query, int(doc_id)),
                            paper_head_html(detail) if detail else gr.update())

                def remove_start(doc_id):
                    if not doc_id or rt.library.detail(int(doc_id)) is None:
                        return gr.update(visible=False), say("Choose a paper first.", False)
                    return gr.update(visible=True), ""

                def remove_confirm(doc_id, filter_value, query=""):
                    if not doc_id:
                        return say("Choose a paper first.", False), *lib_list(filter_value, query), gr.update(), gr.update(), gr.update()
                    outcome = rt.library.delete(int(doc_id), confirmed=True)
                    if not outcome.ok:
                        return (say(outcome.message, False), *lib_list(filter_value, query), gr.update(), gr.update(visible=False), gr.update())
                    return (say(outcome.message), *lib_list(filter_value, query), gr.update(visible=False), gr.update(visible=False), None)

                list_outputs = [papers, lib_empty, add_empty_btn]
                lib_inputs = [lib_filter, search]
                lib_filter.focus(refresh_filter, [lib_filter], [lib_filter], api_name=False)
                search.input(lib_list, lib_inputs, list_outputs, api_name=False, show_progress="hidden")
                lib_filter.change(lib_list, lib_inputs, list_outputs, api_name=False, show_progress="hidden")
                gr.Button(visible=False).click(lib_list, lib_inputs, list_outputs, api_name="library_list")
                library_tab.select(lambda f, q: (refresh_filter(f), *lib_list(f, q)), lib_inputs, [lib_filter, *list_outputs], api_name=False)
                selected_id.change(show_document, [selected_id], doc_outputs, api_name="library_detail", show_progress="hidden")
                page_pick.input(show_page, [selected_id, page_pick], [lib_image, lib_image_note], api_name="library_page")
                change_btn.click(change_subject, [selected_id, new_subject, lib_filter, search], [lib_message, *list_outputs, paper_head],
                                 api_name="library_change_subject")
                remove_btn.click(remove_start, [selected_id], [confirm_group, lib_message], api_name="library_delete_start")
                confirm_btn.click(remove_confirm, [selected_id, lib_filter, search],
                                  [lib_message, *list_outputs, confirm_group, paper_col, selected_id], api_name="library_delete_confirm")
                keep_btn.click(lambda: gr.update(visible=False), None, [confirm_group], api_name=False)
                back_btn.click(lambda: (gr.update(visible=False), None), None, [paper_col, selected_id], api_name=False)
                add_empty_btn.click(lambda: gr.Tabs(selected="add"), None, [tabs], api_name=False)

            # ---- add -------------------------------------------------------------------------------
            with gr.Tab("Add", id="add") as upload_tab:
                with gr.Column(elem_id="sb-add"):
                    gr.HTML("<h2>Add papers</h2><p class='sb-muted'>Photos or a PDF. Study Buddy reads them in the background, "
                            "so you can keep studying.</p>", padding=False)
                    files = gr.File(show_label=False, file_count="multiple", file_types=UPLOAD_TYPES, type="filepath", elem_id="sb-drop",
                                    container=False)
                    up_subject = gr.Dropdown([AUTO_DETECT, *start_names], value=AUTO_DETECT, label="Subject", allow_custom_value=True,
                                             info="Pick one, or type a new subject.", elem_id="sb-add-subject")
                    album = gr.Checkbox(label="These pages belong together", value=False, elem_id="sb-album",
                                        info="Turn this on for several photos of one paper. They are kept in the order of their names.")
                    upload_btn = gr.Button("Add papers", variant="primary", elem_id="sb-add-btn")
                    upload_msg = gr.HTML(elem_id="sb-added", padding=False)

                def upload(dropped, subject, as_album):
                    outcome = submit_uploads(rt.service, dropped, subject, bool(as_album))
                    return say(outcome.message, outcome.ok), (gr.update(value=None) if outcome.ok else gr.update())

                upload_btn.click(upload, [files, up_subject, album], [upload_msg, files], api_name="upload")
                up_subject.focus(refresh_upload_subject, [up_subject], [up_subject], api_name=False)
                upload_tab.select(refresh_upload_subject, [up_subject], [up_subject], api_name=False)

        add_first_btn.click(lambda: gr.Tabs(selected="add"), None, [tabs], api_name=False)
        demo.load(lambda: (*activity_view(False), refresh_filter(None), refresh_filter(None), refresh_upload_subject(None),
                           *lib_list(None, ""), greeting(), *example_updates()),
                  outputs=[*view_outputs, chat_subject, lib_filter, up_subject, *list_outputs, hello, *example_outputs], api_name=False)

        # Plain endpoints for scripts and tests (gradio_client); the buttons are never shown.
        status_api = gr.Button(visible=False)
        status_api.click(status, outputs=status_outputs, api_name="status")
        subjects_out = gr.JSON(visible=False)
        subjects_api = gr.Button(visible=False)
        subjects_api.click(subject_names, outputs=subjects_out, api_name="subjects")

    demo.queue(default_concurrency_limit=4)
    return demo


def launch(demo: gr.Blocks, host: str, port: int | None, **kwargs):
    """Start serving. No `allowed_paths`, no sharing link, no analytics."""
    return demo.launch(server_name=host, server_port=port, share=False, max_file_size=MAX_UPLOAD, ssr_mode=False,
                       inbrowser=False, show_error=False, app_kwargs={"middleware": [Middleware(StripExternalScripts)]}, **kwargs)
