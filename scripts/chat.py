"""Ask questions about your own papers: answers use only your indexed pages and cite file and page.

  python scripts/chat.py
  python scripts/chat.py --subject Science
  python scripts/chat.py --ask "What percentage of Earth's surface is covered by water?"

In the chat: /subject [name|none]   /sources   /new   /help   /quit
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from studybuddy.chat.cli import chat_main  # noqa: E402

if __name__ == "__main__":
    sys.exit(chat_main(sys.argv[1:]))
