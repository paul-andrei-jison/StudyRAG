"""Ingest documents: watch the inbox, ingest files, show status, retry failures.

  python scripts/ingest.py run                       # watch SB_DATA_DIR/inbox and process documents
  python scripts/ingest.py run --until-idle          # process what is there and exit
  python scripts/ingest.py once photo.jpg --subject Science
  python scripts/ingest.py once 1.jpg 2.jpg 3.jpg --album --subject Science
  python scripts/ingest.py status
  python scripts/ingest.py retry-failed
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from studybuddy.ingest.cli import ingest_main  # noqa: E402

if __name__ == "__main__":
    sys.exit(ingest_main(sys.argv[1:]))
