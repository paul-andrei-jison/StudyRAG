"""Inspect and maintain the search index: status, check, repair, reindex, reset, add-text.

  python scripts/index_admin.py status
  python scripts/index_admin.py reindex          # after changing SB_EMBED_MODEL or EMBED_DOC_PREFIX
  python scripts/index_admin.py reset --yes      # delete the index (never the library or your photos)
  python scripts/index_admin.py add-text notes.txt --subject Science
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from studybuddy.index.cli import admin_main  # noqa: E402

if __name__ == "__main__":
    sys.exit(admin_main(sys.argv[1:]))
