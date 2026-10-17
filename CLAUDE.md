# Study Buddy — standing rules

Private study assistant for one user (a senior high school student). Photos/PDFs arrive via a
Telegram bot, are filed into a human-facing library folder, transcribed, indexed, and queried
in a Gradio chat with file + page citations.

## Environment
- Models run in LM Studio (OpenAI-compatible, default http://localhost:1234/v1). No Ollama.
- Three roles, each its own env var: READER (vision), CHAT (text), EMBED. Never hardcode model
  names outside `config.py`. Upgrading a model is a config change only. Warn when the embedding
  model changes (vectors incompatible).
- `SB_LIBRARY_DIR` (browsable, OUTSIDE the repo) and `SB_DATA_DIR` (queue, index, failed) are configurable.
- Stack: Python 3.11+ (venv uses 3.12), pypdfium2 + Pillow, SQLite FTS5, LanceDB, watchdog,
  python-telegram-bot (long polling), Gradio 5, pytest. Pin versions.
- Do not assume library or LM Studio behavior from memory: check installed-version docs and verify
  with live calls (e.g. GET /v1/models).
- Cross-platform: pathlib only; sanitize filenames for Windows.

## Extraction quality (critical)
- Never silently skip or misread a page: every page yields a record with a quality score.
  Pages in must equal pages out, or fail loudly.
- READER: temperature 0, transcribe exactly, never correct/summarize/reorder, `[illegible]` for
  unreadable words, LaTeX for formulas, `[figure: ...]` for diagrams, strip reasoning text.
- Originals are never modified or deleted; process copies. Save originals to `_Unsorted` first.
- Store the original image path on every chunk.

## Git rules
- Feature branch, never main. Do not push or merge without asking.
- One logical change per commit, code with its tests. Conventional Commits, subject < 72 chars,
  imperative, body explains why.
- Run tests before every commit. Never commit secrets, .env, data/, index files or photos.

## Process
- Work in 8 chunks (foundation, reader, index+search, ingest+filing, chat, Gradio UI, Telegram bot,
  hardening). Per chunk: plan, wait for approval, implement, test, show results, commit, show
  `git log --oneline`, then STOP for review.
- Ask only when a decision is expensive to reverse; otherwise pick the simplest option and note it.
- Tests run offline with mocked LM Studio and Telegram. Separately verify against real LM Studio
  and report plainly what could not be tested.
- Out of scope (do not build, do not block): flashcards, quizzes, answering inside Telegram,
  multi-user, web UI auth.
