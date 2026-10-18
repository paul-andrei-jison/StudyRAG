# Web UI

Code: `src/studybuddy/ui/` (logic and wording, tested offline) and `ui/app.py` (layout and wiring only). Look: `ui/theme.css` (one file: tokens, type, every screen), `ui/theme.py` (the Gradio theme and the small page script), `ui/markup.py` (the few pieces of HTML and the visible wording), `ui/offline.py` (no outside requests).
Start: `python -m studybuddy`, then open `http://127.0.0.1:7860`. One process runs the web app, the ingest worker, the inbox watcher and the bot; Ctrl+C lets the paper being read finish.

## The screens

Three areas and a corner indicator; nothing else is visible by default.

- **Ask** (home). Empty: a greeting, one question box, a small subject choice ("All subjects" or one of yours) and three example questions built from your own papers (the newest paper, your busiest subject, a key-terms question). With no papers it says how to add one.
  After a question it is a chat with streamed answers. Under the latest answer are small numbered **source chips**; choosing one lays the real page next to the answer (below it on a phone) on a "paper slip": file name and page, a plain warning when the page was hard to read, and the text read from it behind one disclosure.
  A refusal reads "I couldn't find that in your papers." with an optional **Show the closest pages**. An overview question that fits several papers shows their titles to tap ("Which paper do you mean?"). **New chat** is one small button.
  Chips belong to the **latest** answer only; earlier answers keep their `[1]` marks in the text but their chips are replaced when you ask again.
- **Library**. A search box, a subject choice and one button per paper: title, subject, date, pages, and a small mark ("1 page hard to read"). Choosing a paper opens it: summary, a plain explanation for every hard page, page chips and the page image. **More actions** holds **Change subject** and **Remove from Study Buddy**; removing asks once ("Remove from Study Buddy? Your original file stays in your folder."). On a phone the paper replaces the list, with **Back to papers**. An empty library says how to add the first paper, including by sending a photo to the Telegram bot.
- **Add**. One large drop area for photos or PDFs, a subject choice ("Let Study Buddy decide" by default, or type a new subject), the switch **These pages belong together** (off), and **Add papers**. The confirmation ("Added 1 paper. It is being read now...") stays until the next add.
- **Activity** (corner). A small pill, shown only when there is something to say: "2 being read", "1 needs a look", or "Waiting for LM Studio" (most urgent first). It opens a plain panel: being read now, next in line, what needs a look (with **Retry**), and what to do if LM Studio is off. Queue lines, results and file locations are under **Technical details**.

The header has a **theme toggle** (dark or light; it also follows your system setting and remembers your choice).

## Look

Tokens (cream `#FAF9F5`, ink `#141413`, terracotta `#D97757` for large marks and `#B4492A` for small text and buttons, plus a dark set) are CSS custom properties at the top of `theme.css`. `tests/test_ui_theme.py` checks the real values: every text pair at least 4.5:1, shapes and focus at least 3:1, in both palettes. Headings use a system serif stack (Iowan Old Style, Palatino Linotype, Georgia), everything else the system sans-serif; **no web fonts and no outside requests**: a small middleware removes the one script Gradio's page template loads from a public CDN, and a test shows neither the page nor the server looks anything up outside this machine.
Corner radii are a scale by hierarchy (small marks 6 px, inputs and buttons 10, cards 14, drop area and Activity panel 20, pills and chips fully round). Motion is limited to streaming text, the slip sliding in (160 ms) and the Activity panel fading in (120 ms), and is switched off under "reduce motion". Targets are at least 44 px; keyboard focus is a 2 px terracotta ring on every control; the layout has no sideways scrolling at 390 px.
Gradio scopes custom CSS rules inside `@media` blocks, which breaks `:root` there, so phone rules set variables on `#sb-root`.

## Network exposure (read this)

`SB_HOST` (default `127.0.0.1`) and `SB_PORT` (default 7860). There is **no login**. Any host other than loopback shows a warning at the top of every page ("This app can be reached from the network ... has no login") and in the terminal: anyone who can connect can read, change and delete your library. Gradio runs with no sharing link, no analytics and no `allowed_paths`.

## Decisions to know

- **Remove never deletes the original.** The file stays in the library; only the index entries go. Adding the same file again works.
- **Change subject moves the file, all or nothing** (`ingest/relocate.py`, on the single writer): each file is copied into `SB_LIBRARY_DIR/<New subject>/` and verified by SHA-256, then every stored path is rewritten in one transaction and the vector store follows; only then are the old copies removed. If anything fails the rows and files are put back. A journal in `SB_DATA_DIR/relocations/` lets the next start undo or finish a crash-interrupted move. It is refused, changing nothing, when a stored file is missing or outside the library. If the worker is reading a paper the answer is "Study Buddy is busy reading a paper. Try again in a moment." at once; the action is not queued.
- **Only library files can be shown, and never by URL.** `ui/files.py` resolves symlinks, `..`, junctions and drive forms, requires the real path inside `SB_LIBRARY_DIR`, checks the type from the file's bytes and returns a decoded, size-limited image.
- Chat questions run one at a time (a single GPU). Page text is shown in a plain text box and the chat sanitises HTML; every piece of HTML the screens build is escaped in `markup.py`.
- The list of papers is real buttons in an HTML list; Gradio only hears about a choice through a hidden number box (set by a few lines in the page script). That keeps rows keyboard-reachable.
- The `gradio_client` endpoint names are stable: `/ask /new_chat /show_source /library_list /library_detail /library_page /library_change_subject /library_delete_start /library_delete_confirm /upload /status /retry_failed /subjects`.

## Verified

Offline: the full suite (see the README for the current count), including a smoke test that launches the app and calls every endpoint through `gradio_client`, tests of every piece of wording (`test_ui_copy.py` fails on internal terms, ALL-CAPS words, dots in names and the old action names), the contrast test and the no-outside-requests test.

In a real browser (Microsoft Edge driven by the development-only `playwright`, on **synthetic** demo papers and a fake model; `scripts/ui_demo.py`, `scripts/ui_screenshots.py`, run the checks with `SB_BROWSER_TESTS=1 pytest tests/test_ui_browser.py`): every screen at 390 and 1280 px in light and dark has no sideways scroll, targets of at least 44 px, no console errors and no request to any host but this machine; the theme toggle, the remembered choice, the system dark setting, a focus ring on every keyboard stop, choosing a chip with the keyboard and reduced motion all pass.

Earlier, with the real models (2026-10-18): upload to indexed, a cited answer, a refusal, change subject and remove.

## Not verified

Nobody has used the redesigned screens with real photos on a phone; touch behaviour (as opposed to a narrow window), Safari and Firefox, a screen reader, hundreds of papers, and several browser tabs at once. Gradio emits deprecation warnings for `api_name=False` that will matter when moving to Gradio 6.
