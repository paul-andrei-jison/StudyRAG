"""Search the index from the command line.

  python scripts/search.py "what is the habitable zone" --subject Science --k 5 --mode hybrid
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from studybuddy.index.search_cli import search_main  # noqa: E402

if __name__ == "__main__":
    sys.exit(search_main(sys.argv[1:]))
