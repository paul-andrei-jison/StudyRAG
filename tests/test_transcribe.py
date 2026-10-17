import base64
import io
import json

import httpx
import pytest
from PIL import Image

from studybuddy.errors import ReaderError
from studybuddy.llm import LMStudioClient
from studybuddy.reader.transcribe import USER_PROMPT, Reader, unfence

OVERFLOW = "n_keep: 40009>= n_ctx: 13824). Try to load the model with a larger context length"


def reader_with(settings, handler):
    return Reader(LMStudioClient(settings, transport=httpx.MockTransport(handler)))


def ok(content, finish="stop"):
    return httpx.Response(200, json={"choices": [{"message": {"content": content}, "finish_reason": finish}]})


def page(w=1500, h=2000):
    return Image.new("L", (w, h), 255)


def test_request_shape_and_verbatim_rules(settings):
    seen = []

    def handler(request):
        seen.append(json.loads(request.content))
        return ok("Hello page")

    out = reader_with(settings, handler).transcribe(page())
    body = seen[0]
    assert out.text == "Hello page" and out.attempts == 1
    assert body["model"] == "reader-m" and body["temperature"] == 0.0 and body["max_tokens"] > 1000
    assert "transcription engine" in body["messages"][0]["content"]
    text, image = body["messages"][1]["content"]
    assert text["text"] == USER_PROMPT
    for needle in ("verbatim", "Markdown table", "LaTeX", "[figure:", "[illegible]", "reading order"):
        assert needle in USER_PROMPT
    png = base64.b64decode(image["image_url"]["url"].split(",", 1)[1])
    assert Image.open(io.BytesIO(png)).size == (1500, 2000)


def test_reasoning_text_stripped(settings):
    out = reader_with(settings, lambda r: ok("<think>let me see</think>Real text")).transcribe(page())
    assert out.text == "Real text"


def test_wrapping_code_fence_removed(settings):
    out = reader_with(settings, lambda r: ok("```markdown\n| a | b |\n| --- | --- |\n| 1 | 2 |\n```")).transcribe(page())
    assert out.text == "| a | b |\n| --- | --- |\n| 1 | 2 |"


def test_inner_fences_kept():
    text = "intro\n```\ncode\n```\nend"
    assert unfence(text) == text


def test_truncated_output_fails_loudly(settings):
    with pytest.raises(ReaderError, match="cut off"):
        reader_with(settings, lambda r: ok("half a pa", finish="length")).transcribe(page())


def test_blank_page_returns_empty_text(settings):
    assert reader_with(settings, lambda r: ok("")).transcribe(page()).text == ""


def test_context_overflow_retries_with_smaller_image(settings):
    sizes = []

    def handler(request):
        body = json.loads(request.content)
        url = body["messages"][1]["content"][1]["image_url"]["url"]
        sizes.append(Image.open(io.BytesIO(base64.b64decode(url.split(",", 1)[1]))).size)
        if len(sizes) == 1:
            return httpx.Response(400, json={"error": OVERFLOW})
        return ok("fits now")

    out = reader_with(settings, handler).transcribe(page(1500, 2000))
    assert out.text == "fits now" and out.attempts == 2
    assert sizes[1] == (1050, 1400) and out.image_size == (1050, 1400)


def test_persistent_overflow_gives_up_loudly(settings):
    calls = []

    def handler(request):
        calls.append(1)
        return httpx.Response(400, json={"error": OVERFLOW})

    with pytest.raises(ReaderError, match="does not fit"):
        reader_with(settings, handler).transcribe(page(1500, 2000))
    assert 1 < len(calls) < 5


def test_other_llm_errors_propagate(settings):
    from studybuddy.errors import LLMError
    with pytest.raises(LLMError):
        reader_with(settings, lambda r: httpx.Response(500, text="boom")).transcribe(page())
