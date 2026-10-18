"""The Gradio app: layout and wiring only. Every decision lives in the modules next to this one, where it is tested.

Pages are returned to the browser as decoded images (see files.py); no library path is ever given to Gradio, and the
app is launched without `allowed_paths`, so Gradio cannot serve a file from the library by URL.
"""
from __future__ import annotations

import html

import gradio as gr

from ..config import network_exposure
from ..errors import StudyBuddyError
from .chat import DONE, source_choices, source_details, sources_caption, stream_turn
from .library import TABLE_HEADERS
from .runtime import Runtime
from .uploads import AUTO_DETECT, submit_uploads

ALL_SUBJECTS = "All subjects"
TIMER_SECONDS = 3.0
MAX_UPLOAD = "100mb"
UPLOAD_TYPES = [".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff", ".heic", ".pdf"]
CSS = """
.sb-warning {background:#b00020;color:#fff;padding:12px 16px;border-radius:8px;font-weight:600;margin-bottom:8px}
.sb-warning * {color:#fff}
"""


def exposure_banner(host: str) -> str | None:
    warning = network_exposure(host)
    if warning is None:
        return None
    return (f"<div class='sb-warning'>WARNING: this app is reachable from the network ({html.escape(host)}) and has no login. "
            f"Anyone who can connect can read, change and delete your study library. "
            f"Set SB_HOST=127.0.0.1 and restart unless you really mean this.</div>")


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

    with gr.Blocks(title="Study Buddy", css=CSS) as demo:
        gr.Markdown("# Study Buddy\nAsk questions about your own papers. Answers cite the page they came from.")
        banner = exposure_banner(rt.settings.host)
        if banner:
            gr.HTML(banner)

        # ---- status panel (every tab) ---------------------------------------------------------
        with gr.Accordion("Status", open=True):
            headline = gr.Markdown("Status: ...")
            with gr.Accordion("Details", open=False):
                jobs_box = gr.Textbox(label="Jobs", lines=4, interactive=False)
                recent_box = gr.Textbox(label="Latest results", lines=6, interactive=False)
                retry_btn = gr.Button("Retry failed")
                retry_msg = gr.Markdown()
        timer = gr.Timer(TIMER_SECONDS)

        def status():
            snap = rt.board.snapshot()
            return snap.headline(), snap.jobs_text(), snap.recent_text()

        def retry_failed():
            message = rt.board.retry_failed()
            return (message, *status())

        status_outputs = [headline, jobs_box, recent_box]
        timer.tick(status, outputs=status_outputs, api_name=False, show_progress="hidden")
        retry_btn.click(retry_failed, outputs=[retry_msg, *status_outputs], api_name="retry_failed")

        with gr.Tabs():
            # ---- chat ------------------------------------------------------------------------
            with gr.Tab("Chat") as chat_tab:
                session_state = gr.State(None)
                answer_state = gr.State(None)
                with gr.Row():
                    with gr.Column(scale=3):
                        chatbot = gr.Chatbot(type="messages", height=460, label="Conversation", sanitize_html=True,
                                             render_markdown=True)
                        with gr.Row():
                            question = gr.Textbox(placeholder="Ask about your papers...", show_label=False, scale=5,
                                                  lines=1, max_lines=4)
                            send_btn = gr.Button("Ask", variant="primary", scale=1)
                    with gr.Column(scale=2):
                        chat_subject = gr.Dropdown([ALL_SUBJECTS, *start_names], value=ALL_SUBJECTS, label="Subject", interactive=True,
                                                   allow_custom_value=True)
                        new_btn = gr.Button("New chat")
                        sources_md = gr.Markdown()
                        source_radio = gr.Radio([], label="Pages", interactive=True)
                        page_image = gr.Image(type="pil", format="jpeg", interactive=False, label="Page")
                        page_info = gr.Textbox(label="File and page", lines=4, interactive=False, show_copy_button=True)
                        page_text = gr.Textbox(label="Text read from this page", lines=8, interactive=False,
                                               show_copy_button=True)

                source_outputs = [sources_md, source_radio, page_image, page_info, page_text]

                def blank_sources():
                    return "", gr.update(choices=[], value=None), None, "", ""

                def ask(text, history, subject, session, answer):
                    text = (text or "").strip()
                    history = list(history or [])
                    if not text:
                        gr.Warning("Type a question first.")
                        yield (history, gr.update(), session, answer, *[gr.update()] * 5)
                        return
                    session = session or rt.new_session()
                    shown = history + [{"role": "user", "content": text}]
                    first = True
                    for update in stream_turn(session, text, _subject_filter(subject)):
                        message = {"role": "assistant", "content": update.text}
                        if update.phase == DONE and update.answer is not None:
                            ans = update.answer
                            yield (shown + [message], "", session, ans, sources_caption(ans),
                                   gr.update(choices=source_choices(ans), value=None), None, "", "")
                        else:
                            yield (shown + [message], "" if first else gr.update(), session, None if first else gr.update(),
                                   *(blank_sources() if first else [gr.update()] * 5))
                        first = False

                ask_inputs = [question, chatbot, chat_subject, session_state, answer_state]
                ask_outputs = [chatbot, question, session_state, answer_state, *source_outputs]
                send_btn.click(ask, ask_inputs, ask_outputs, api_name="ask", concurrency_limit=1, concurrency_id="chat",
                               show_progress="hidden")
                question.submit(ask, ask_inputs, ask_outputs, api_name=False, concurrency_limit=1, concurrency_id="chat",
                                show_progress="hidden")

                def new_chat(session):
                    if session is not None:
                        session.new()
                    return ([], "", session, None, *blank_sources())

                new_btn.click(new_chat, [session_state], [chatbot, question, session_state, answer_state, *source_outputs],
                              api_name="new_chat")

                def show_source(answer, number):
                    if answer is None or number is None:
                        return None, "", ""
                    return source_details(answer, int(number), rt.guard)

                source_radio.input(show_source, [answer_state, source_radio], [page_image, page_info, page_text],
                                   api_name="show_source")
                chat_subject.focus(refresh_filter, [chat_subject], [chat_subject], api_name=False)
                chat_tab.select(refresh_filter, [chat_subject], [chat_subject], api_name=False)

            # ---- library ---------------------------------------------------------------------
            with gr.Tab("Library") as library_tab:
                with gr.Row():
                    lib_filter = gr.Dropdown([ALL_SUBJECTS, *start_names], value=ALL_SUBJECTS, label="Show subject", scale=3,
                                             allow_custom_value=True)
                    lib_refresh_btn = gr.Button("Refresh", scale=1)
                table = gr.Dataframe(headers=TABLE_HEADERS, datatype="str", type="array", interactive=False, wrap=True,
                                     max_height=320, label="Documents (click a row)")
                selected_id = gr.Number(label="Selected document ID", precision=0, interactive=True)
                lib_message = gr.Markdown()
                with gr.Row():
                    with gr.Column(scale=2):
                        detail_box = gr.Textbox(label="Document", lines=9, interactive=False, show_copy_button=True)
                        with gr.Row():
                            new_subject = gr.Dropdown(start_names, label="Change subject to", allow_custom_value=True, scale=3)
                            change_btn = gr.Button("Change subject", scale=1)
                        delete_btn = gr.Button("Delete from index...", variant="stop")
                        with gr.Group(visible=False) as confirm_group:
                            confirm_md = gr.Markdown()
                            with gr.Row():
                                confirm_btn = gr.Button("Yes, remove it", variant="stop")
                                cancel_btn = gr.Button("Cancel")
                    with gr.Column(scale=3):
                        page_pick = gr.Dropdown([], label="Page", interactive=True, allow_custom_value=True)
                        lib_image = gr.Image(type="pil", format="jpeg", interactive=False, label="Original page")
                        lib_image_note = gr.Markdown()

                def lib_table(filter_value):
                    return rt.library.table(_subject_filter(filter_value))

                def pick_row(evt: gr.SelectData):
                    row = evt.row_value or []
                    try:
                        return int(float(row[0]))
                    except (IndexError, TypeError, ValueError):
                        return gr.update()

                def show_page(doc_id, page):
                    if not doc_id or not page:
                        return None, ""
                    try:
                        return rt.library.page_image(int(doc_id), int(page)), ""
                    except StudyBuddyError as exc:
                        return None, f"**Page image not shown:** {exc}"

                doc_outputs = [detail_box, page_pick, lib_image, lib_image_note, new_subject, confirm_group]

                def show_document(doc_id):
                    hidden = gr.update(visible=False)
                    if not doc_id:
                        return "", gr.update(choices=[], value=None), None, "", gr.update(), hidden
                    detail = rt.library.detail(int(doc_id))
                    if detail is None:
                        return ("That document no longer exists. Press Refresh.", gr.update(choices=[], value=None), None, "",
                                gr.update(), hidden)
                    first_flagged = next((p.page_number for p in detail.pages if p.flagged), None)
                    page = first_flagged or (detail.pages[0].page_number if detail.pages else None)
                    image, note = show_page(doc_id, page) if page else (None, "")
                    return (detail.describe(), gr.update(choices=[p.page_number for p in detail.pages], value=page), image,
                            note, gr.update(choices=subject_names(), value=detail.row.subject), hidden)

                def change_subject(doc_id, subject, filter_value):
                    if not doc_id:
                        return "Pick a document first.", gr.update(), gr.update()
                    outcome = rt.library.change_subject(int(doc_id), subject or "")
                    detail = rt.library.detail(int(doc_id))
                    return (("" if outcome.ok else "**Not changed:** ") + outcome.message, lib_table(filter_value),
                            detail.describe() if detail else gr.update())

                def delete_start(doc_id):
                    detail = rt.library.detail(int(doc_id)) if doc_id else None
                    if detail is None:
                        return gr.update(visible=False), "", "Pick a document first."
                    return (gr.update(visible=True),
                            f"Remove **{detail.row.title or 'this document'}** from the index? It will no longer be found in "
                            f"chat. The original file stays in your library.", "")

                def delete_confirm(doc_id, filter_value):
                    if not doc_id:
                        return "Pick a document first.", gr.update(), *[gr.update()] * 6
                    outcome = rt.library.delete(int(doc_id), confirmed=True)
                    message = ("" if outcome.ok else "**Not removed:** ") + outcome.message
                    if not outcome.ok:
                        return message, lib_table(filter_value), *[gr.update()] * 6
                    return (message, lib_table(filter_value), "", gr.update(choices=[], value=None), None, "", gr.update(),
                            gr.update(visible=False))

                lib_filter.focus(refresh_filter, [lib_filter], [lib_filter], api_name=False)
                lib_refresh_btn.click(lib_table, [lib_filter], [table], api_name="library_list")
                lib_filter.change(lib_table, [lib_filter], [table], api_name=False)
                library_tab.select(lambda f: (refresh_filter(f), lib_table(f)), [lib_filter], [lib_filter, table],
                                   api_name=False)
                table.select(pick_row, None, [selected_id], api_name=False)
                selected_id.change(show_document, [selected_id], doc_outputs, api_name="library_detail")
                page_pick.input(show_page, [selected_id, page_pick], [lib_image, lib_image_note], api_name="library_page")
                change_btn.click(change_subject, [selected_id, new_subject, lib_filter], [lib_message, table, detail_box],
                                 api_name="library_change_subject")
                delete_btn.click(delete_start, [selected_id], [confirm_group, confirm_md, lib_message],
                                 api_name="library_delete_start")
                confirm_btn.click(delete_confirm, [selected_id, lib_filter], [lib_message, table, *doc_outputs],
                                  api_name="library_delete_confirm")
                cancel_btn.click(lambda: gr.update(visible=False), None, [confirm_group], api_name=False)

            # ---- upload ----------------------------------------------------------------------
            with gr.Tab("Upload") as upload_tab:
                gr.Markdown("Drop photos of your papers or a PDF. They are read in the background; the status panel shows progress.")
                files = gr.File(label="Photos or PDF", file_count="multiple", file_types=UPLOAD_TYPES, type="filepath")
                with gr.Row():
                    up_subject = gr.Dropdown([AUTO_DETECT, *start_names], value=AUTO_DETECT, label="Subject (type a new one if needed)",
                                             allow_custom_value=True, scale=3)
                    album = gr.Checkbox(label="These photos are pages of one document (album)", value=False, scale=2)
                upload_btn = gr.Button("Add to my library", variant="primary")
                upload_msg = gr.Markdown()

                def upload(dropped, subject, as_album):
                    outcome = submit_uploads(rt.service, dropped, subject, bool(as_album))
                    return ("" if outcome.ok else "**Not queued:** ") + outcome.message, (gr.update(value=None) if outcome.ok else gr.update())

                upload_btn.click(upload, [files, up_subject, album], [upload_msg, files], api_name="upload")
                up_subject.focus(refresh_upload_subject, [up_subject], [up_subject], api_name=False)
                upload_tab.select(refresh_upload_subject, [up_subject], [up_subject], api_name=False)

        demo.load(lambda: (*status(), refresh_filter(None), refresh_filter(None), refresh_upload_subject(None),
                           rt.library.table(None)),
                  outputs=[*status_outputs, chat_subject, lib_filter, up_subject, table], api_name=False)

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
                       inbrowser=False, show_error=False, **kwargs)
