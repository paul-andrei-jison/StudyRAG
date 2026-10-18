"""Development tool: serve the real web app on SYNTHETIC demo papers and a scripted fake model (no LM Studio, no real data).

  python scripts/ui_demo.py            serve at http://127.0.0.1:7861 until Ctrl+C

Used by scripts/ui_screenshots.py. The pages are drawn with Pillow, the answers are canned, and everything lives in a temporary
folder. Needs the dev-only packages used by the tests (pytest); it never touches your library, index or Telegram bot.
"""
from __future__ import annotations

import dataclasses
import sys
import tempfile
import threading
import time
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tests"))

from PIL import Image, ImageDraw, ImageFont  # noqa: E402

from chat_helpers import LAX_GATE, ScriptedClient  # noqa: E402
from conftest import FakeEmbedder  # noqa: E402
from studybuddy.config import Settings  # noqa: E402
from studybuddy.index.index import Index  # noqa: E402
from studybuddy.index.store import DocumentMeta  # noqa: E402
from studybuddy.ingest.service import IngestService  # noqa: E402
from studybuddy.ingest.subjects import SubjectList  # noqa: E402
from studybuddy.llm import StreamEnd  # noqa: E402
from studybuddy.reader.quality import PageRecord  # noqa: E402
from studybuddy.ui.runtime import Runtime  # noqa: E402

PAPERS = [
    ("Science", "Water cycle worksheet", "water-cycle", 0.95, [], [
        "The water cycle describes how water moves around the Earth.",
        "Evaporation: the sun heats water in oceans and lakes and it rises as vapour.",
        "Condensation: vapour cools and forms clouds.",
        "Precipitation: water falls as rain, hail or snow.",
        "Collection: water gathers in rivers, lakes and the ocean again.",
    ], "How water evaporates, condenses, falls as rain and is collected again."),
    ("Science", "Layers of the Earth", "earth-layers", 0.93, [], [
        "The Earth has four layers: crust, mantle, outer core and inner core.",
        "The outer core is liquid. Moving molten iron there creates Earth's magnetic field.",
        "The magnetic field protects the atmosphere from the solar wind.",
    ], "The four layers of the Earth and how the outer core makes the magnetic field."),
    ("Math", "Fractions practice", "fractions", 0.96, [], [
        "Fractions practice: add and subtract fractions with unlike denominators.",
        "1/2 + 1/3 = 5/6    3/4 - 1/8 = 5/8",
    ], "Adding and subtracting fractions with different denominators."),
    ("English", "Story notes: theme and setting", "story-notes", 0.41, ["many_illegible"], [
        "Theme: courage is choosing to act even when you are afraid.",
        "Setting: a small fishing village during a storm.",
    ], "Notes on the theme and setting of a short story."),
]

FAILED_NOTE = "reader exploded"


class DemoClient(ScriptedClient):
    """A pretend language model: picks a canned, cited answer from the question's words, and streams it slowly."""

    def __init__(self, delay: float = 0.04):
        super().__init__()
        self.delay = delay

    def chat_stream(self, messages, temperature=0.0, max_tokens=None):
        question = messages[-1]["content"].lower()
        if "water" in question or "evaporat" in question or "cycle" in question:
            pieces = ["Water evaporates from oceans and lakes, ", "condenses into clouds, ", "and falls as rain or snow ", "[1]", ". It is then collected again ", "[1]", "."]
        elif "magnet" in question or "core" in question or "layer" in question:
            pieces = ["Moving molten iron in the outer core ", "creates Earth's magnetic field ", "[1]", ". It shields the atmosphere ", "from the solar wind ", "[1]", "."]
        elif "fraction" in question:
            pieces = ["To add fractions with different denominators, ", "first find a common denominator ", "[1]", "."]
        else:
            pieces = ["Water evaporates from oceans and lakes, ", "condenses into clouds, ", "and falls as rain or snow ", "[1]", ". It is then collected again ", "[1]", "."]
        for piece in pieces:
            time.sleep(self.delay)
            yield piece
        yield StreamEnd(None, "stop")


def draw_page(path: Path, title: str, lines: list[str], faint: bool = False) -> Path:
    """A plausible-looking worksheet photo, 900x1200, drawn from scratch (no real material is ever used)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    image = Image.new("RGB", (900, 1200), (246, 243, 234))
    draw = ImageDraw.Draw(image)
    ink = (170, 170, 165) if faint else (30, 30, 40)
    head = ImageFont.load_default(size=40)
    body = ImageFont.load_default(size=26)
    draw.text((60, 60), title, fill=ink, font=head)
    draw.line((60, 125, 840, 125), fill=ink, width=2)
    y = 170
    for line in lines:
        words, current = line.split(), ""
        for word in words:
            trial = (current + " " + word).strip()
            if draw.textlength(trial, font=body) > 780:
                draw.text((60, y), current, fill=ink, font=body)
                y, current = y + 42, word
            else:
                current = trial
        draw.text((60, y), current, fill=ink, font=body)
        y += 70
    for k in range(9):                                   # faint ruled lines at the bottom, like a notebook
        draw.line((60, 900 + k * 34, 840, 900 + k * 34), fill=(205, 208, 215), width=1)
    image.save(path, format="JPEG", quality=88)
    return path


@dataclass
class Demo:
    root: Path
    rt: Runtime
    demo: object
    url: str
    release: threading.Event | None = None

    def close(self) -> None:
        if self.release:
            self.release.set()
        self.demo.close()
        self.rt.close()


class SlowReader:
    """Holds every document until `release` is set (to show 'being read' in the activity panel)."""

    def __init__(self, release: threading.Event):
        self.release = release

    def transcribe(self, image):
        self.release.wait(120)
        raise RuntimeError("demo reader stopped")


def demo_settings(root: Path) -> Settings:
    return Settings(base_url="http://lm.test/v1", api_key="k", reader_model="reader-m", chat_model="chat-m", embed_model="embed-m",
                    library_dir=root / "library", data_dir=root / "data", subjects_file=root / "data" / "subjects.json",
                    tag_timeout_seconds=300, telegram_token=None, telegram_user_id=None, chat_expand_queries=False)


def build_demo(root: Path, *, papers: bool = True, host: str = "127.0.0.1", port: int | None = None, busy: bool = False) -> Demo:
    from studybuddy.ui.app import build_app, launch

    settings = demo_settings(root)
    settings.library_dir.mkdir(parents=True, exist_ok=True)
    settings.data_dir.mkdir(parents=True, exist_ok=True)
    settings = dataclasses.replace(settings, host=host)
    index = Index.open(settings, FakeEmbedder())
    subjects = SubjectList(settings.data_dir / "subjects.json", ["Math", "Science", "English", "History", "Filipino"])
    if papers:
        for subject, title, key, quality, flags, lines, summary in PAPERS:
            pages = [lines] if key != "fractions" else [lines[:1], lines[1:]]
            path = None
            records = []
            for number, page_lines in enumerate(pages, 1):
                image = draw_page(settings.library_dir / subject / f"{key}-{number}.jpg", title, page_lines, faint=bool(flags))
                path = path or str(image)
                records.append(PageRecord(number, str(image), "image", text=" ".join(page_lines), quality=quality, flags=list(flags)))
            index.add_document(DocumentMeta(key, path, "photo", subject, title, summary), records)
    release = threading.Event() if busy else None
    service = IngestService(settings, index, SlowReader(release) if busy else _NoReader(), None, DemoClient(), subjects, poll_interval=0.05)
    rt = Runtime(settings, DemoClient(), index, subjects, service, gate=LAX_GATE)
    rt.start()
    demo = build_app(rt)
    _, url, _ = launch(demo, host, port, prevent_thread_lock=True)
    return Demo(root, rt, demo, url, release)


class _NoReader:
    def transcribe(self, image):
        raise RuntimeError(FAILED_NOTE)


def main() -> int:
    with tempfile.TemporaryDirectory(prefix="sb-demo-", ignore_cleanup_errors=True) as tmp:
        demo = build_demo(Path(tmp), port=7861)
        print(f"Demo (synthetic papers, fake model) at {demo.url}   Ctrl+C to stop", flush=True)
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            pass
        finally:
            demo.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
