# Study Buddy

Private study assistant: photograph handouts, send them to a Telegram bot, and ask questions
later with answers cited to your own pages. Full setup instructions arrive in the final chunk.

## Set up

In the project folder, once per virtual environment: `.\.venv\Scripts\python.exe -m pip install -r requirements.txt` and then
`.\.venv\Scripts\python.exe -m pip install -e . --no-deps` (an editable install, so `python -m studybuddy` works from any fresh shell without `PYTHONPATH`).

## Run it

`python -m studybuddy`, then open http://127.0.0.1:7860 (see `docs/ui.md`; `SB_HOST` / `SB_PORT` in `.env`; there is no login, so keep the host on 127.0.0.1).

## Telegram bot

Send photos, photos-as-files and PDFs to your own private bot; it asks which subject (buttons), files and reads them, and tells you when each is ready.
It runs inside `python -m studybuddy` (long polling, no open port) and answers only your Telegram user id. Setup with @BotFather, finding your id,
what it does and how to test it live: `docs/bot.md`. Turn it off with `--no-bot`.
