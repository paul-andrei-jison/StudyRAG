# Telegram bot

Code: `src/studybuddy/bot/` (`flow.py` the conversation, `telegram_app.py` python-telegram-bot, `spool.py`, `state.py`, `whoami.py`).
It runs **inside the same process** as the web app and the ingest service: `python -m studybuddy`. Turn it off with `--no-bot` or `SB_TELEGRAM_ENABLED=0`.
It only **receives files and reports results**. Questions about your papers are asked in the web app (answering inside Telegram is out of scope).

## Set up (about five minutes)

1. In Telegram open **@BotFather**, send `/newbot`, choose a display name and a username that ends in `bot`. Copy the token it prints. Optional: `/setjoingroups` -> Disable.
2. Put the token in `.env` as `TELEGRAM_BOT_TOKEN=...`. `.env` is git-ignored; never paste the token anywhere else.
3. Open your new bot in Telegram and send it any message ("hi"). Stop the Study Buddy app if it is running, then run `python scripts/telegram_whoami.py`. It prints your numeric user id.
4. Put it in `.env` as `SB_TELEGRAM_USER_ID=<id>` (the old name `TELEGRAM_USER_ID` still works). **A token without a user id is refused at start-up**, because the bot would otherwise answer anybody.
5. `python -m studybuddy` prints `telegram: on (long polling; answers only user id ...)`.

## What you do, what it does

| You send | It does |
|---|---|
| a photo | saves it at once, asks which subject (buttons), files and reads it, tells you when it is ready |
| a photo as a **file** (full resolution) | same; recommended for pages that were hard to read |
| a PDF | same (scanned pages are read, text layers used) |
| a caption naming a known subject (`Science` or `subject: Science`) | no question; filed under it |
| a caption that is a valid but unknown name (`Astronomy`) | the question has an extra button "Add 'Astronomy' as a new subject" |
| 2+ photos at once | **one** question ("3 photos: same subject for all?"), then each photo is its own document |
| ... and you tap **Make these one document** | one album document (pages in the order sent); **Keep separate** undoes it |
| anything else (Word, video, voice, sticker...) | a plain message that only photos, image files and PDFs are filed; nothing is saved |
| a file over 20 MB | a plain message ("too big for a bot to download"); nothing is saved |

Subject buttons come from the live subject list, then **Other...** (your next text message is the new subject; bad names are explained and asked again; **Back** returns to the buttons) and **Auto-detect**.
No answer within **5 minutes**: auto-detect, and the message says so.

When a job ends you get one reply per document (per album):
**indexed**: the filed name, subject (marked auto-detected, or Unsorted with a pointer to the Library screen), title, page count, every flagged page with its reason in plain words (a "retake it or send it as a file" hint when a resend would help), any ingest warnings, and **always** "Handwritten names, dates, scores and answer letters may be read wrongly: check them against the photo."
**duplicate**: "I already have this one: '<title>' (<subject>, N pages)... the new copy is kept in <duplicates folder>".
**failed**: the reason, and where the file is kept (use Retry in the corner indicator of the web page). **deferred** (LM Studio off): the file is saved and it retries by itself.

## Decisions to know

- **Nothing is lost on arrival.** Every file is written to `SB_LIBRARY_DIR/_Unsorted/` (temporary name, flush, rename) before any question is asked. Ingest copies it into its subject folder, verifies the copy by hash and only then removes the spool file. A file the bot cannot download yet (network) is retried every minute for 5 tries, then you are asked to resend.
- **Pending state survives a restart.** `SB_DATA_DIR/bot/items|questions/*.json` (atomic writes) hold every file, question, deadline and job id. After a restart: unanswered questions are still answerable by their old buttons; ones past their deadline go to auto-detect at once; jobs that finished while the app was off get one "processed while I was switched off, see the library" message.
- **Each file is its own job.** `IngestService.submit()` now refuses several paths without `is_album=True`, so the bot cannot produce the "first file decides" duplicate trap described in `docs/ingest.md`.
- **Only your user id is served, only in a private chat.** Everyone else gets no reply; the log records `ignored a message from an unauthorised sender (id N)` (update type and numeric id only, never text, captions or file names). Button taps are checked the same way.
- **Long polling, no webhook, no open port.** The bot runs in its own thread and event loop (`Application.run_polling` is main-thread only). Messages sent while the computer slept are fetched when it wakes (Telegram keeps them about 24 h). With no connection at start-up it keeps trying every 60 s and the web app is not delayed. A rejected token stops only the bot, with a log line.
- **The token is a secret.** It is read from `.env`, left out of `Settings` repr, and `logsafe.py` removes anything shaped like a bot token from every log message and traceback before any handler sees it; the `httpx`/`telegram` loggers are set to WARNING because httpx logs request URLs (which contain the token). Tests prove it for log lines, tracebacks, `repr(settings)` and error messages.
- The READER prompt is unchanged.

## Verified and not verified

Offline (1,299 tests in the full suite; bot: `test_bot_flow.py` 64, `test_bot_telegram.py` 37, `test_bot_state.py` 13, `test_bot_runtime.py` 5, `test_bot_whoami.py` 8, `test_logsafe.py`): a fake Bot API (`tests/telegram_helpers.py`) is plugged into python-telegram-bot's HTTP layer, so real update parsing, handlers, polling thread, start/stop and file download run with no network. Covered: strangers and group chats, all file kinds, size limits, retries, groups and albums, captions, Other..., the 5-minute fallback, restarts, results of every kind, and one pass from a photo through the real ingest service into the library and back to a "Filed:" reply.

**Not verified: anything against real Telegram.** No bot token was available. Behaviour taken from Telegram's documentation and python-telegram-bot 22.8 but not seen live: photos sent together arrive as separate messages sharing a `media_group_id` within about a second (the 2 s settle time is a guess), the 20 MB `getFile` limit and the wording of its error ("file is too big"), 64-byte callback data, 24 h update retention, and recovery of the polling loop after the computer sleeps. See "Live test" below.

## Live test

With the app running (`telegram: on`), send from your phone, one at a time, and look at the reply and at `SB_LIBRARY_DIR`:

1. A photo with caption `Science`: no buttons, filed under Science, a "Filed:" reply ending with the handwriting warning.
2. A photo with no caption: buttons; tap one; the question message changes to "Queued ...".
3. Three photos together: one question ("3 photos: same subject for all?"); tap Auto-detect -> three replies. Again with three photos: tap **Make these one document**, pick a subject -> one reply, one folder with `page-01.jpg ...` and a PDF.
4. A photo sent **as a file**; a PDF; the same photo twice (second reply is the duplicate message).
5. A file over 20 MB and a Word document: a plain refusal each, nothing in `_Unsorted`.
6. Leave a question unanswered for 5 minutes: the auto-detect notice appears and the file is filed.
7. Send a photo, close the app before answering, start it again: the buttons still work.
8. Put the computer to sleep for a few minutes, send a photo from the phone, wake it: it is picked up.
9. Have another Telegram account message the bot: no reply, and the app log shows `ignored a message from an unauthorised sender (id ...)`.
10. Library screen: change the subject of a filed document; the file moves to the new folder and a page still opens.

## Known limits

- One chat only (yours). No commands besides `/start` and `/help` (both just print what to send).
- A photo group that arrives more than a few seconds apart is asked about as two groups (a late photo of a group that is still being asked joins its question).
- Telegram compresses ordinary photos; for small print send them "as a file".
- If the connection is down when a document finishes, the "Filed:" message is replaced by the generic "processed while I was switched off" note at the next start.
