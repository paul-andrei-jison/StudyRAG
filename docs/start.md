# Starting Study Buddy

## Every day

Double-click **`start.bat`** in the project folder. It:

1. looks for the project's Python (`.venv`); if it is missing it tells you to follow "Install" in the README;
2. runs `python -m studybuddy --check`, which reports LM Studio, the three models, folders, free disk space, Telegram and the index;
3. if every line is fine it starts the app (open http://127.0.0.1:7860);
4. if a line says `[FAIL]` it explains what to do (open LM Studio, load the models, turn on its local server) and asks "Start Study Buddy anyway?". Answer `N`, fix it, and run `start.bat` again.

The window must stay open while you use the app. Close it (or press Ctrl+C) to stop; the document being read is finished first.

`python -m studybuddy --check` can be run by itself at any time (`--deep` also asks each model for a tiny answer). The Telegram token is never printed.

## Start with Windows

1. Press `Win + R`, type `shell:startup`, press Enter. A folder opens.
2. In the project folder, right-click `start.bat` -> *Show more options* -> *Create shortcut*, and move the shortcut into the startup folder.
3. Right-click the shortcut -> *Properties*. In **Target** add a space and the word `wait` after the path (`...\start.bat" wait`). In **Run** choose *Minimized*.

With `wait`, the script does not ask questions: if LM Studio is not ready it checks again every 15 seconds for up to 10 minutes (time for you to open LM Studio and load the models), then gives up and leaves the window open with the reason.

Notes:
- LM Studio itself does not start with Windows unless you turn that on in its settings. Study Buddy needs its local server running.
- If the computer sleeps, the app pauses. Photos you send to the bot meanwhile are kept by Telegram for about a day and are picked up when the computer wakes.
- Logs are in `SB_DATA_DIR\logs\studybuddy.log` (the path is printed at start-up).
