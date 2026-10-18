# Study Buddy

A private study helper that runs on your own computer.

1. You take a photo of a handout, worksheet or notes (or have a PDF) and send it to your own Telegram bot.
2. Study Buddy files it in a folder by subject, reads the writing on the page, and tells you when it is ready.
3. Later you open a web page on your computer and ask questions ("What is the habitable zone?", "What is this worksheet about?"). It answers **only from your own pages** and shows which page it used, so you can check.

If the answer is not in your papers, it says **Not in your papers** instead of guessing.

Everything is read and answered by programs on your computer. Nothing is sent to an online AI service. See [Privacy](#privacy).

---

## What you need

| You need | Why | Notes |
|---|---|---|
| A Windows computer that can run an 8-billion-parameter model (16 GB of memory recommended) and a few GB of free disk space | the AI models run on it | a graphics card makes it much faster |
| **LM Studio** (free, https://lmstudio.ai) | runs the AI models | |
| **Python 3.12** (https://www.python.org/downloads/; tick "Add python.exe to PATH") | runs Study Buddy | 3.11 or newer works |
| The **Telegram** app on your phone | to send photos | |
| This project folder | | keep it somewhere that is not inside the library folder |

### The models (in LM Studio)

Download these in LM Studio's search tab, then load them in the *Developer* tab (or *Local Server*) and **start the server** (default address `http://localhost:1234`):

| Job | Model name | Setting in `.env` |
|---|---|---|
| reads the photo (vision) and answers questions | `qwen/qwen3-vl-8b` | `SB_READER_MODEL`, `SB_CHAT_MODEL` |
| turns text into search data | `text-embedding-nomic-embed-text-v1.5` | `SB_EMBED_MODEL` |

In LM Studio set the loaded model's **context length** as high as your memory allows (the project was tested at about 40,000). A small context makes long answers and "what is this document about" questions worse. If you switch to a different model later, change only the names in `.env`. If you change the embedding model, rebuild the search data: `python scripts\index_admin.py reindex`.

---

## Install (once)

Open **PowerShell** in the project folder (in File Explorer: click the address bar, type `powershell`, Enter) and run:

```
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m pip install -e . --no-deps
copy .env.example .env
```

Now open the new `.env` file in Notepad and fill it in:

```
SB_READER_MODEL=qwen/qwen3-vl-8b
SB_CHAT_MODEL=qwen/qwen3-vl-8b
SB_EMBED_MODEL=text-embedding-nomic-embed-text-v1.5
EMBED_DOC_PREFIX="search_document: "
EMBED_QUERY_PREFIX="search_query: "
SB_LIBRARY_DIR=C:\StudyBuddy          the folder you browse: your photos, sorted by subject
SB_DATA_DIR=C:\StudyBuddyData         the program's working area (index, queue, logs)
```

The two folders must be **outside** the project folder and not inside each other. They are created for you. Never share `.env`: it will hold your bot's secret token.

Check that it all works:

```
.\.venv\Scripts\python.exe -m studybuddy --check
```

Every line should say `[PASS]`. A `[FAIL]` line tells you what to fix.

## Connect Telegram (once)

1. In Telegram search for **@BotFather** (the official one, with a blue tick) and send `/newbot`. Choose a name, then a username that ends in `bot`. BotFather replies with a **token** (long, like `123456789:AA...`). Treat it like a password.
2. In `.env` add: `TELEGRAM_BOT_TOKEN=` followed by the token.
3. Open your new bot in Telegram and send it any message, for example `hi`. (It will not answer yet.)
4. In PowerShell run `.\.venv\Scripts\python.exe scripts\telegram_whoami.py`. It prints your numeric **user id**. (Study Buddy must not be running.)
5. In `.env` add: `SB_TELEGRAM_USER_ID=` followed by that number.

From now on the bot answers **only your account**. If the token is set but the user id is missing, Study Buddy refuses to start rather than answer strangers. More detail, and a list of things to try: [docs/bot.md](docs/bot.md).

## Subjects

The list of subjects is in `subjects.json` (Math, Science, English, History, Filipino to begin with). When you send a photo the bot offers these as buttons, plus **Other...** (type a new subject; it is added to the list) and **Auto-detect** (the AI picks). Write the subject as the photo's caption (for example `Science`) to skip the question. In the web page, the **Library** tab can change a document's subject later and moves the file to the right folder.

---

## Daily use

1. Double-click **`start.bat`**. It checks LM Studio and the models and starts Study Buddy. Keep the window open.
2. Open **http://127.0.0.1:7860** in your browser. There are three areas: **Ask**, **Library** and **Add**.
3. **Send pages**: in Telegram send photos (or PDFs) to your bot, or use **Add** in the browser.
   - In Telegram tap the subject button. If you do not answer within 5 minutes it lets Study Buddy decide and tells you.
   - Several photos sent together get **one** question and are filed as separate papers. If they are pages of one handout, tap **Make these one document**. In the browser, turn on **These pages belong together**.
   - For small print or messy handwriting, send the photo **as a file** (the paperclip, then File) so Telegram does not shrink it.
   - When a page is ready you get a message with the file name, subject, number of pages and any page that was hard to read. Handwritten names, dates, scores and answer letters may be read wrongly: always check those against your photo.
4. **Ask** a question. Pick a subject first if you like. The answer shows small numbered chips; choose one to see the real page next to the answer (below it on a phone). You can also ask about a whole paper ("What is this worksheet about?"). If it says "I couldn't find that in your papers", **Show the closest pages** lists what it looked at. **New chat** starts over.
5. **Library**: search your papers, open one to read its summary and see its pages. Under **More actions** you can change its subject or remove it from Study Buddy (your original file always stays in your folder).
6. **The corner indicator** shows what Study Buddy is doing:
   - **"2 being read"**: papers are being read now or waiting their turn. Nothing to do.
   - **"1 needs a look"**: a paper could not be read. Open the indicator and press **Retry**, or add a clearer photo.
   - **"Waiting for LM Studio"**: LM Studio is off, or no model is loaded. Open LM Studio, load the models, start its server. Waiting papers carry on by themselves; you do not need to send them again.
   - Nothing shown means everything is done. The little half-moon button switches dark and light mode.
7. When you finish, close the window (or press Ctrl+C). It finishes the page it is reading first.

To start with Windows: [docs/start.md](docs/start.md).

## Backups

Run now and then (use a USB stick or another drive):

```
.\.venv\Scripts\python.exe scripts\backup.py D:\Backups
```

The photos are the part that cannot be recreated. Restore steps: [docs/backup.md](docs/backup.md).

---

## When something goes wrong

Start with `.\.venv\Scripts\python.exe -m studybuddy --check`. It says what is wrong in plain words.

| Problem | What to do |
|---|---|
| `start.bat` says LM Studio is not reachable | Open LM Studio, load the models, open the *Developer*/*Local Server* tab and start the server. Run `start.bat` again. |
| A model is "not loaded" | Load it in LM Studio. The name must match `.env` exactly (copy it from LM Studio's list). |
| The bot does not answer | Is Study Buddy running (the `start.bat` window)? Does the window say `telegram: on`? Is the token right and `SB_TELEGRAM_USER_ID` your own id? Only one copy may run at a time. The log (below) says more. |
| I sent photos while the computer was asleep or off | Telegram keeps them for about a day. When Study Buddy runs again they are picked up. Photos saved on arrival are never lost, even if the computer restarts. |
| The computer went to sleep during a long read | It continues after waking. If a paper shows as "needs a look" in the corner, open it and press **Retry**. |
| "I already have this one" | You sent a page that is already filed (identical file). Nothing was added; the message names the existing copy. A different photo of the same page is a new document. |
| The page was read badly | Retake it in good light, flat and straight, or send it **as a file**. Pages the program doubts are listed in the reply and in the Library tab. |
| An answer has no source or seems wrong | Check the page it shows. Answers without a source are marked. Trust the page, not the answer. |
| "Port 7860 is in use" | Study Buddy is probably already running; look for another window. Or set `SB_PORT=7861` in `.env`. |
| It is slow | The first question after LM Studio loads a model takes longer. Reading a page takes from about half a minute to a few minutes, depending on the computer. Close other heavy programs. |
| `--check` warns about disk space | Free some space; photos and the index keep growing. |

**Log file:** `SB_DATA_DIR\logs\studybuddy.log` (the path is printed when it starts). It never contains your bot token or the content of your photos. If you ask someone for help, you can send it; it does contain file names and error messages.

## Privacy

- The AI models run **on your computer** in LM Studio. Your pages and your questions are not sent to any online AI service.
- Your photos, their text and the search index are stored only in the two folders you chose.
- Telegram is the one outside service: a photo you send to the bot travels through Telegram's servers, like any message. Study Buddy uses only the bot you made, talks to Telegram by "long polling" (it asks Telegram for new messages; nothing on your computer is opened to the internet), and answers only your own account.
- The web page at 127.0.0.1 can only be opened from this computer. It has no password, so keep `SB_HOST=127.0.0.1` (the default).
- The bot token is kept in `.env` only. It is hidden from the log file and from the `--check` report, and `.env` is never committed to git.
- Don't put the project's `.env`, the library folder or the data folder in a shared or public place (GitHub, a shared drive). Backups contain all your photos: keep them private too.

---

## For developers

- Standing rules for working on the code: [CLAUDE.md](CLAUDE.md). Docs per part: `docs/` (`index.md`, `ingest.md`, `chat.md`, `ui.md`, `bot.md`, `start.md`, `backup.md`, reader notes).
- Tests (offline, no models, no network): `.\.venv\Scripts\python.exe -m pytest -q`. One test is skipped on a normal Windows account: it needs permission to create symbolic links (Developer Mode or an administrator shell); a similar test with a junction covers the same case.
- Real-model check on your own sample pages, in temporary folders: `python scripts\e2e_live.py` (needs `SB_SAMPLES_DIR`).
- Git hooks are not part of a clone. In a new clone run once: `python scripts\install_hooks.py` (it protects the reader prompt). Merge instructions: [docs/merge.md](docs/merge.md).
- Settings are listed with comments in `.env.example`.
