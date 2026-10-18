# Putting the work on `main`

Nothing here has been run. These are the steps for when you say go.

## What the branches look like

`feat/foundation`, `feat/reader` and `feat/index` are **one straight line**: each branch contains all the commits of the one before it, and `feat/index` is the newest. You can confirm that on any day:

```
git merge-base --is-ancestor feat/foundation feat/reader && echo "foundation is inside reader"
git merge-base --is-ancestor feat/reader feat/index && echo "reader is inside index"
git log --oneline --graph feat/index | head -20      # one line, no forks
```

There is **no `main` yet**, neither on your computer nor on GitHub (`git ls-remote --heads origin` lists only the three `feat/` branches). So "combining" means making `main` point at the end of the line. No merge commit and no conflicts are possible, and every commit keeps its message and its place.

## Steps

1. Finish and commit everything on `feat/index`; `git status` must say "nothing to commit".
2. Last checks on the final commit:
   ```
   .\.venv\Scripts\python.exe -m pytest -q
   .\.venv\Scripts\python.exe scripts\leak_check.py --names-file .leakcheck
   ```
   (`.leakcheck` is your own list of words to look for: your full name, school, section, user id. It is ignored by git. Without it the check only looks for tokens, e-mail addresses, photos, databases and `.env`.)
3. Create `main` at the end of the line:
   ```
   git switch -c main feat/index
   git log --oneline | measure-object -line      # same number of commits as feat/index
   ```
4. Only when you are happy, publish it (this is the step that goes outside your computer):
   ```
   git push -u origin main
   ```
   On GitHub: *Settings -> Branches -> Default branch* -> `main`. If you want the repository to be private (recommended: it is a personal study tool), check *Settings -> General -> Danger zone -> Change visibility* **before** pushing anything else.
5. The `feat/` branches can stay (they are the same commits) or be deleted later: `git push origin --delete feat/foundation feat/reader feat/index` and `git branch -d ...`. Nothing is lost, because `main` contains all of their commits.

### If a `main` already exists on GitHub by the time you do this

(For example GitHub created one with a README.) Do not force anything. Fetch it and look first:

```
git fetch origin
git log --oneline origin/main
```

If it only holds a README or licence commit, bring it in with `git switch -c main feat/index` then `git merge origin/main --allow-unrelated-histories` (fix the README conflict by keeping ours), run the tests, then push. Ask before doing this: it is the only case where a merge commit is created.

## A fresh clone (for you on another computer, or anyone you share it with)

Git does **not** copy `.env`, your library, your index or git hooks. After cloning:

```
git clone <the repository address> StudyRag
cd StudyRag
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m pip install -e . --no-deps
copy .env.example .env                 (then fill it in: README.md, "Install")
.\.venv\Scripts\python.exe scripts\install_hooks.py      the hook install step
.\.venv\Scripts\python.exe -m studybuddy --check
```

`install_hooks.py` sets `core.hooksPath` to `scripts/git-hooks` in **that clone only**. The `commit-msg` hook refuses a commit that changes the reader prompt's baseline without a `Bake-off:` line (see `src/studybuddy/reader/prompt_baseline.py`). Without it, everything still runs; you just lose that guard.
