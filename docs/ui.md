# Web UI

Code: `src/studybuddy/ui/` (logic, tested offline) and `ui/app.py` (Gradio layout only). Start: `python -m studybuddy [--no-ocr]`, then open
`http://127.0.0.1:7860`. One process runs the web app, the ingest worker and the inbox watcher; Ctrl+C lets the document being read finish.

## Network exposure (read this)

`SB_HOST` (default `127.0.0.1`) and `SB_PORT` (default 7860). There is **no login**. Any host other than loopback prints a warning bar in the terminal and
shows a red banner at the top of every page: anyone who can connect can read, change and delete your library. Gradio runs with no sharing link, no
analytics and no `allowed_paths`.

## Tabs

- **Chat**: streamed answers, subject filter (refreshed from the live subject list on page load, on opening the tab and when the list is opened), **New chat**,
  and a list of the pages used. Click one to see the page image, `File:` and `Page:`, and the text read from it. A refusal says **Not in your papers** and lists the closest pages
  as *probably not relevant*; hard-to-read pages and program notes show as `Check this:` callouts. While nothing has arrived it shows "Searching your papers..." and, after 8 s,
  "Waking the model, this can take a minute...".
- **Library**: documents with subject, title, date, pages, summary, flagged-page count and path. Click a row for the detail, per-page quality in plain words and the original page image.
  **Change subject** moves the file(s) to the new subject's folder and updates the index (see below); **Delete from index** asks for confirmation. If the worker is reading a document the answer is "busy, try again shortly" at once:
  the action is refused, not queued, so it cannot run later by surprise.
- **Upload**: drop photos or PDFs, pick a subject or Auto-detect (a new name may be typed), tick *album* to make several photos one document (natural file-name order, photos only).
  Uploads are copied into `SB_DATA_DIR/uploads/` and queued with `IngestService.submit(origin="ui")`, the same call the Telegram bot will use. Gradio refuses file types the tab does not list.
- **Status** (above the tabs, refreshed every 3 s): reading / queued / waiting-to-retry / failed counts, the lists with reasons, the latest results, and **Retry failed**.

## Decisions to know

- **Delete never removes the original.** The file stays in the library; only the index entries go. Uploading the same file again works.
- **Change subject moves the file, all or nothing** (`ingest/relocate.py`, on the single writer). Each file (an album: its folder, pages and PDF) is copied into
  `SB_LIBRARY_DIR/<New subject>/` with the new subject in its name and checked by SHA-256 against the original before it is renamed into place; then the subject and
  every stored path (document, pages, the page records, chunks) are rewritten in one transaction and the vector store follows; only then are the old copies re-checked and deleted
  and the old folder removed if it is empty. If anything fails (a copy that does not match, a database or vector error) exactly what was created is deleted and the rows are put back,
  so nothing changes. It is refused, changing nothing, when a stored file is missing or lies outside the library. A journal in `SB_DATA_DIR/relocations/` lets the next start undo or finish a move a crash interrupted.
  Moving to the subject a document already has does nothing. A file the student renamed by hand keeps its name and only changes folder.
- **Only library files can be shown, and never by URL.** `ui/files.py` resolves symlinks, `..`, junctions, drive and stream forms, requires the real path inside `SB_LIBRARY_DIR`, checks the
  file type from its bytes and returns a decoded, size-limited image. A path in the index that points elsewhere is refused. The smoke test confirms `/gradio_api/file=<library file>` and `../` variants return 403/404.
- Chat questions run one at a time (a single GPU); other tabs keep working. Page text is shown in plain text boxes, and the chat sanitises HTML.

## Verified

Offline: 967 tests including a smoke test that launches the app and calls every endpoint through `gradio_client` (streamed answer with an openable cited page, upload to indexed, subject change,
confirmed delete, a busy worker, retry-failed, refused URL fetches, the exposure banner).

Live, 2026-10-18, temporary folders, real models (`qwen3-vl-8b`, nomic), the three sample photos: upload of 3 photos filed and indexed in 82 s (one with a typed subject, two auto-detected),
library rows and page image (960x1280) served, an answerable question answered with citations `[1][2][5]`, an unanswerable one refused with the closest pages marked not relevant, change subject and
delete left the original file in place. Cold start (models unloaded): waking state shown, first answer after 18 s; warm question about 5 s to the first word. The live run found one real bug, fixed:
after the model was unloaded LM Studio reports no context size, so chat failed; it now sends a one-token request to load the model first.

## Not verified

Looks, layout and phone width (nobody has seen the screen); real HEIC rendering; several browser tabs at once (a tab keeps its own conversation, but this was not exercised); the
click-selection of a table row (tested through its API equivalents, not with a mouse); the Library tab's behaviour with hundreds of documents. Gradio emits deprecation warnings for
`api_name=False` that will matter when moving to Gradio 6.
