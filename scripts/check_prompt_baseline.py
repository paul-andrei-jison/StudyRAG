"""Exit 1 if the READER prompt no longer matches the recorded bake-off baseline."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from studybuddy.reader.prompt_baseline import check_baseline  # noqa: E402


def main() -> int:
    problems = check_baseline()
    for problem in problems:
        print(f"[prompt baseline] {problem}", file=sys.stderr)
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
