"""Back up the library and the index to a folder you choose, or verify an earlier backup.

  python scripts/backup.py D:/Backups
  python scripts/backup.py --verify D:/Backups/studybuddy-backup-20261018-190000

See docs/backup.md (including how to restore).
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from studybuddy.backup import BackupError, make_backup, verify_backup  # noqa: E402
from studybuddy.config import load_settings  # noqa: E402
from studybuddy.errors import StudyBuddyError  # noqa: E402


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Back up the Study Buddy library and index.")
    parser.add_argument("destination", type=Path, help="folder to put the backup in (a new dated sub-folder is made)")
    parser.add_argument("--verify", action="store_true", help="treat the argument as an existing backup folder and check it")
    args = parser.parse_args(argv)
    if args.verify:
        problems = verify_backup(args.destination)
        for line in problems:
            print(f"[FAIL] {line}")
        print("The backup is intact: every file matches its recorded checksum." if not problems else f"{len(problems)} problem(s) found.")
        return 1 if problems else 0
    try:
        result = make_backup(load_settings(), args.destination)
    except (BackupError, StudyBuddyError) as exc:
        print(f"[FAIL] {exc}")
        return 1
    print(f"Backup written to {result.folder}\n  {result.files} file(s), {result.bytes / 1024 ** 2:.1f} MB, every file read back and checked.")
    for line in result.skipped:
        print(f"  [WARN] not copied: {line}")
    print("Restore steps are in RESTORE.txt next to the backup and in docs/backup.md.")
    return 1 if result.skipped else 0


if __name__ == "__main__":
    sys.exit(main())
