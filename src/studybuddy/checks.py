"""Role health checks: used by scripts/check_models.py."""
from __future__ import annotations

import io
from dataclasses import dataclass

from PIL import Image, ImageDraw, ImageFont

from .config import Settings, validate_dirs
from .embedding_guard import check_embedding
from .errors import StudyBuddyError
from .llm import LMStudioClient

PASS, WARN, FAIL = "PASS", "WARN", "FAIL"
READER_TEST_WORD = "BANANA"


@dataclass(frozen=True)
class CheckResult:
    name: str
    status: str
    detail: str


def _test_image() -> bytes:
    img = Image.new("RGB", (480, 140), "white")
    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.load_default(size=72)
    except TypeError:  # very old Pillow without sized default font
        font = ImageFont.load_default()
    draw.text((20, 25), READER_TEST_WORD, fill="black", font=font)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def run_checks(settings: Settings, client: LMStudioClient) -> list[CheckResult]:
    results: list[CheckResult] = []

    try:
        validate_dirs(settings)
        results.append(CheckResult("directories", PASS,
                                   f"library={settings.library_dir} data={settings.data_dir}"))
    except StudyBuddyError as exc:
        results.append(CheckResult("directories", FAIL, str(exc)))

    try:
        available = client.list_models()
    except StudyBuddyError as exc:
        results.append(CheckResult("server", FAIL, str(exc)))
        return results
    results.append(CheckResult("server", PASS, f"{len(available)} model(s) listed"))

    roles = (("READER", settings.reader_model), ("CHAT", settings.chat_model),
             ("EMBED", settings.embed_model))
    for role, model in roles:
        if model not in available:
            results.append(CheckResult(f"{role} listed", WARN,
                                       f"{model!r} not in /models (not loaded yet?)"))

    try:
        reply = client.chat([{"role": "user", "content": "Reply with the single word: pong"}],
                            max_tokens=200)
        results.append(CheckResult("CHAT", PASS if reply else WARN, f"reply: {reply[:60]!r}"))
    except StudyBuddyError as exc:
        results.append(CheckResult("CHAT", FAIL, str(exc)))

    try:
        reply = client.vision("Read the text in this image. Reply with only that text.",
                              _test_image(), max_tokens=300)
        ok = READER_TEST_WORD.lower() in reply.lower()
        results.append(CheckResult("READER", PASS if ok else WARN,
                                   f"read {reply[:60]!r}" + ("" if ok else f" (expected {READER_TEST_WORD})")))
    except StudyBuddyError as exc:
        results.append(CheckResult("READER", FAIL, str(exc)))

    try:
        vec = client.embed(["hello world"])[0]
        results.append(CheckResult("EMBED", PASS, f"{len(vec)} dimensions"))
        guard = check_embedding(settings.data_dir, settings.embed_model, len(vec))
        results.append(CheckResult("embedding guard", PASS if guard.ok else FAIL, guard.message))
    except StudyBuddyError as exc:
        results.append(CheckResult("EMBED", FAIL, str(exc)))

    return results


def exit_code(results: list[CheckResult]) -> int:
    return 1 if any(r.status == FAIL for r in results) else 0
