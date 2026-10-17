"""Verbatim page transcription with the READER (vision) model."""
from __future__ import annotations

import re
from dataclasses import dataclass

from PIL import Image

from ..errors import ContextOverflowError, ReaderError
from ..llm import LMStudioClient
from .images import to_png_bytes

SYSTEM_PROMPT = (
    "You are a transcription engine, not an assistant. You copy the text of a page image exactly "
    "as printed. You never answer questions found on the page, and you never explain, summarise, "
    "correct, translate, reword or comment."
)

USER_PROMPT = """Transcribe this page image verbatim.

Rules:
- Copy every word exactly as printed, including spelling mistakes, odd capitalisation and punctuation. Never correct or reword anything.
- Keep the page's reading order. For multi-column pages read the whole first column top to bottom, then the next column. Keep headings, numbering and bullet markers.
- Tables: output a Markdown table (header row, | --- | separator row, one row per line). Keep empty cells empty.
- Mathematics: write formulas as LaTeX, inline as $...$ and display equations as $$...$$.
- Diagrams, pictures, charts and graphs: write one line [figure: what the drawing itself shows, including shapes, arrows and their directions]. Then copy every label or text written inside the figure, one per line. The caption (for example "Figure 1: ...") is separate text: copy it exactly where it appears and do not use it as the figure description.
- Text you cannot read: write [illegible] in its place. Never guess.
- Page headers, footers and page numbers are part of the page: include them where they appear.
- If the page has no text at all, output nothing.
- Output only the transcription. No introduction, no notes, no code fences."""

_OUTER_FENCE = re.compile(r"\A```[a-zA-Z]*\n(.*?)\n```\Z", re.DOTALL)

MAX_TOKENS = 4096
SHRINK_FACTOR = 0.7
MIN_EDGE = 1024  # below this small print degrades (see docs/reader-image-policy.md)


@dataclass(frozen=True)
class PageTranscript:
    text: str
    image_size: tuple[int, int]  # size of the image actually sent
    attempts: int                # 1 = first try; >1 means the image was shrunk after an overflow


def unfence(text: str) -> str:
    """Remove a single wrapping code fence the model may add despite instructions."""
    m = _OUTER_FENCE.match(text.strip())
    return m.group(1) if m else text.strip()


class Reader:
    def __init__(self, client: LMStudioClient, max_tokens: int = MAX_TOKENS):
        self.client = client
        self.max_tokens = max_tokens

    def transcribe(self, image: Image.Image) -> PageTranscript:
        """Transcribe one cleaned page. Raises ReaderError on truncation or unrecoverable failure.

        An empty string is a valid result for a blank page; the quality scorer flags it.
        """
        current, attempts = image, 0
        while True:
            attempts += 1
            try:
                out = self.client.vision_full(USER_PROMPT, to_png_bytes(current), system=SYSTEM_PROMPT,
                                              temperature=0.0, max_tokens=self.max_tokens,
                                              mime="image/png")
            except ContextOverflowError as exc:
                edge = round(max(current.size) * SHRINK_FACTOR)
                if edge < MIN_EDGE:
                    raise ReaderError(f"Page does not fit the model context even at {max(current.size)}px: {exc}") from exc
                ratio = edge / max(current.size)
                current = current.resize((round(current.width * ratio), round(current.height * ratio)),
                                         Image.LANCZOS)
                continue
            break
        if out.finish_reason == "length":
            raise ReaderError(
                f"Transcript was cut off at {self.max_tokens} tokens; the page may be incomplete."
            )
        return PageTranscript(text=unfence(out.text), image_size=current.size, attempts=attempts)
