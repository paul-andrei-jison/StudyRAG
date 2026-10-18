"""Scan all branches and the working tree for tokens, names, e-mail addresses, photos, databases and .env. See studybuddy/leakcheck.py.

  python scripts/leak_check.py --names-file .leakcheck        (.leakcheck: your own words, one per line; git ignores it)
  python scripts/leak_check.py --name "Your Name" --name "Your School"
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from studybuddy.leakcheck import (  # noqa: E402
    _run, check_history, check_working_tree, compile_names, format_report, read_names_file, real_findings)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--path", type=Path, default=Path(__file__).resolve().parents[1], help="the repository (default: this one)")
    parser.add_argument("--names-file", type=Path, default=None, help="file with your own words to look for (default: .leakcheck if present)")
    parser.add_argument("--name", action="append", default=[], help="a word to look for (full name, school, section, user id); repeatable")
    args = parser.parse_args(argv)
    repo = args.path.resolve()
    words = list(args.name) + read_names_file(args.names_file or repo / ".leakcheck")
    names = compile_names(words)
    if not names:
        print("Note: no personal words given (--name or .leakcheck), so only tokens, e-mails, files and secrets are checked.")
    branches = [b.strip().lstrip("* ") for b in _run(["branch", "--all", "--format=%(refname:short)"], repo).splitlines() if b.strip()]
    findings, stats = check_history(repo, names)
    findings += check_working_tree(repo, names)
    print(format_report(findings, stats, branches))
    return 1 if real_findings(findings) else 0


if __name__ == "__main__":
    sys.exit(main())
