# Backup and restore

## Make a backup

```
.\.venv\Scripts\python.exe scripts\backup.py D:\Backups
```

Use a folder on **another drive or a USB stick** (a backup on the same disk does not help if the disk dies). The script refuses a folder inside the library, the data folder or the project. It is safe to run while the app is open: the database is copied with SQLite's own backup function. Every run makes a new folder `studybuddy-backup-YYYYMMDD-HHMMSS`; nothing is overwritten. Each file is read back and compared with a SHA-256 checksum before the folder gets its final name; if anything differs the half-made folder is deleted and you are told.

What is in it:

| Included | Not included |
|---|---|
| the whole library (`SB_LIBRARY_DIR`: photos, PDFs, subject folders, `_Unsorted`) | `.env` (it holds the bot token: keep your own private copy) |
| the index: `index/studybuddy.db` (page text, titles, summaries) and `index/vectors` (search data) | logs, `samples`, `eval`, older backups |
| `embed_meta.json` (which embedding model the vectors belong to) | files still being written (`*.partial`) |
| the Telegram bot's pending state, the job queue, uploads, inbox, duplicates | |

Check an old backup any time: `python scripts\backup.py --verify D:\Backups\studybuddy-backup-...` (it reports changed, missing and unexpected files).

How often: after a study session that added new papers, or at least weekly. The library is the part that cannot be recreated (the photos). The index can always be rebuilt from the text in the database (`index_admin.py reindex`), and the database itself can be rebuilt by sending the photos again.

## Restore

Restoring replaces what you have now, so do it deliberately. The same steps are in `RESTORE.txt` inside each backup.

1. Close Study Buddy (check that no window is still running it).
2. Verify the backup: `python scripts\backup.py --verify <backup folder>`. Do not continue if it reports problems; try an older backup.
3. Keep what you have: rename the current library folder and data folder (add `-old` to the names). Delete them only when the restored app works.
4. Copy the backup's `library` folder to the place `SB_LIBRARY_DIR` points to in `.env`, and its `data` folder to `SB_DATA_DIR`. (Different drive or username? Change those two lines in `.env` to the new places; the stored paths inside the index are rewritten only by the app's own subject-move, so restore to the **same** paths whenever you can. If the paths differ, search and answers still work, but photos in citations will not open because they are stored by full path; this case has no tool yet.)
5. Run `python -m studybuddy --check`. The `index` and `embedding model` lines must be PASS.
6. Start the app. If searches look wrong, run `python scripts\index_admin.py reindex`; it rebuilds the search data from the stored text (needs LM Studio and the embedding model loaded).
7. Put your private copy of `.env` back if the project folder is new (token, user id, model names).

Tested: a backup of the real library and index was restored into temporary folders and `--check` passed on it (see the Chunk 8 report). Not tested: restoring onto a different drive letter.
