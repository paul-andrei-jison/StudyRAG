# Ingest and filing

Code: `src/studybuddy/ingest/`. Commands: `python scripts/ingest.py run | once | status | retry-failed`.

## What happens to a document

`IngestService.submit(paths, subject=None, source_kind=None, received_at=None, origin=..., is_album=False)` (the bot calls this;
the inbox watcher calls it too) writes a job file to `SB_DATA_DIR/queue/` and returns a `Ticket`. One worker thread then does,
for each job, **all or nothing**:

1. Hash the files; if the content hash is already indexed it is a **duplicate** (reported, the new copy moved to `SB_DATA_DIR/duplicates/`). No model is called.
2. Check types from the file's bytes (not its name), decode every image, render scanned PDF pages, use PDF text layers.
3. Read every page (`read_pages`). If any page fails, the whole document fails (no half documents).
4. One CHAT call: title, 2-3 sentence summary and, when the student did not tag a subject, one entry from the subject list (or `Unsorted`).
5. Copy the original into `SB_LIBRARY_DIR/<Subject>/` under a temporary name, flush, verify by SHA-256, rename into place (exclusive creation, `_2`, `_3` on collision).
6. Index the document (chunks, vectors, pages) with each page pointing at the **filed** original.
7. Only now remove the original from the inbox, after re-hashing the filed copy.

If anything fails before step 7, exactly what the job created is deleted again and the inbox files are untouched.

## Naming

`2026-10-17_1425_Science_photosynthesis-worksheet.jpg`: received time, subject, topic slug from the title. Windows-safe (no `<>:"/\|?*`,
no trailing dot or space, reserved device names like CON/NUL escaped, paths kept under 240 characters). Apostrophes are dropped
(`earths-systems`). An **album** gets its own folder with `page-01.jpg ...` (the original bytes) and a combined PDF named after the
folder (`album.pdf` if a very long library path leaves no room). The PDF downscales photos to 2,500 px; the page photos keep full resolution.

## Inbox (`SB_DATA_DIR/inbox/`)

```
inbox/photo.jpg                  subject auto-detected
inbox/Science/photo.jpg          subject Science (a new folder name becomes a subject once a document is filed under it)
inbox/Science/trip/1.jpg 2.jpg   ONE album (2+ photos in a sub-folder), natural name order (img2 before img10)
```

A file is queued once its size and modification time stop changing for 3 s and it can be opened; an album folder once no more photos
arrive. Temporary and system files (`.tmp`, `.crdownload`, `~$`, dotfiles, `Thumbs.db`, `desktop.ini`) are ignored. The student's
subject folders stay; emptied album folders are removed. `watchdog` only wakes the scanner early, a 5 s rescan catches anything missed.

## Subjects

`subjects.json` in the repo is a default template. If `SB_SUBJECTS_FILE` is inside the repo, the live list is `SB_DATA_DIR/subjects.json`
(seeded from the template); if it points outside the repo, that file is used directly. Auto-detect must answer with a name from the list;
anything else becomes `Unsorted` (with a warning), so folders are never invented. A typed subject is validated when submitted and added to
the list when a document is filed under it. Reserved or empty names, names over 40 characters and Windows device names are rejected.
`_Unsorted` is reserved for the Telegram bot's arrival spool.

## Failures

| | examples | what happens |
|---|---|---|
| **transient** | LM Studio off, busy or with no model loaded; timeouts; HTTP 5xx/429; disk trouble; embedding model changed (run `index_admin.py reindex`) | job stays queued with backoff 30 s, 1, 2, 5, then 15 min, forever; reported once as `deferred` |
| **permanent** | corrupt or empty file, GIF/DOCX/unknown type, HEIC without `pillow-heif`, encrypted or corrupt PDF, a page the reader cannot read, a bad subject name, an unexpected bug | files move to `SB_DATA_DIR/failed/<job>/` with `reason.json` (error, attempts, traceback for bugs) |

`python scripts/ingest.py retry-failed` requeues everything in `failed/`. `status` lists the queue, failures, duplicates and subjects.
Exit code: 0 all fine, 1 something failed, 3 something is waiting for a retry.

## Result for the bot (`IngestResult`)

`status` (`indexed` | `duplicate` | `failed` | `deferred`), `subject` and `subject_source` (`user` | `auto` | `fallback`), `title`, `summary`,
`page_count`, `flagged_pages` (page number, quality, flags, plain-language reasons, whether to ask for a resend), `final_path` (file or album
folder), `pdf_path`, `doc_id`, `source_kinds`, `existing` (for duplicates), `error`, `transient`, `retry_in_seconds`, `failed_dir`,
`duplicate_dir`, `warnings`, `attempts`, `duration_seconds`. The ticket has `reported` (first outcome, which may be `deferred`) and `final`
(never `deferred`). Reminder from `reader-bakeoff.md`: the bot must tell the student that handwritten names, dates, scores and answer
letters are not reliable.

## Single writer

One worker thread is the only code that reads documents into the index or writes to the library. The UI uses `run_exclusive(fn)`
to delete a document or change a subject between jobs. `Index` also takes a write lock around its own writes. Tests run 20
simultaneous submissions and interleaved UI actions and prove processing and index writes never overlap.

## Verified live (temporary library and data folders, real models, 2026-10-17)

Three real photos (one in `inbox/Science/`, two at the root) were filed in 77 s with identical bytes, auto-detected subjects from the
list and no leftovers in the inbox; an album of three photos became a folder with the three original photos and a 3-page PDF; a
duplicate, a GIF and a truncated JPEG were reported as such (duplicate kept in `duplicates/`, the other two in `failed/` with reasons);
with the LM Studio URL pointed at a dead port the document was reported `deferred` once, stayed in the inbox, and was indexed after
the backoff when the real server was reachable. Searching the filed documents with 13 questions gave vector hit@1 0.92 and hybrid 0.77.

## Known limits

- Photos are read one at a time (one local GPU), about 15-25 s per page plus the title call.
- Handwriting, tables, formulas and multi-column pages are untested on real photos beyond the bake-off pages.
- The Telegram bot (chunk 7) must hold the original bytes in `_Unsorted` itself until it calls `submit()`.

## Several files in one submit (checked 2026-10-18; `submit()` now refuses the unsafe case)

One `submit()` call is ONE job, and a job is ONE document. `is_album` is the only thing that says the files are pages of that document.
**Since Chunk 7 `submit()` raises for several paths without `is_album=True`** ("Several files are one document only as an album ..."): nothing is queued or moved. `ingest.py once a b c` therefore submits each file on its own unless `--album` is given; the Upload tab and the inbox already did this. The paragraphs below describe what happened before that guard.

- **Several files, not an album:** the job's hash is that of the FIRST file only. If the first file is already in the library the whole job is reported `duplicate` and ALL files
  (including new ones) are moved to `data/duplicates/`: the new pages are kept but never read or indexed (seen: an indexed photo plus a new one gave `DUPLICATE`, both ended up in `duplicates/`).
  If the first file is new, `load_document` refuses with "Several files were submitted as one document but not as an album", the job goes to `data/failed/` with all files, nothing is
  read or indexed, and `retry-failed` fails the same way.
- **Several files as an album:** always ONE document: pages 1..n in the order given, one subject, one title and one summary (written from an excerpt of all pages, in practice
  dominated by the first), one hash of the page hashes. Three unrelated worksheets sent this way became "Earth's Life-Supporting Features, 3 pages" with a summary of the first sheet.
  Citations still name the right page, but deleting, changing the subject or an overview question ("summarize this worksheet") acts on all three together, and sending one of them
  again later is not recognised as a duplicate (it is a different hash).
- **For Chunk 7 (Telegram):** submit every received photo as its own job by default. Telegram marks photos sent together with a `media_group_id`, but three different worksheets sent in
  one go look exactly like three pages of one, so the bot must not guess: make an album only when the student asks (a caption such as `album`, or a reply to a "one document or separate?" prompt).
  Also worth a small fix there or earlier: make `submit()` raise for several paths without `is_album=True`, so the "first file decides" case cannot send new pages to `duplicates/`.
