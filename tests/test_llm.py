import base64
import json

import httpx
import pytest

from conftest import FakeLM
from studybuddy.errors import LLMError
from studybuddy.llm import EMBED_BATCH_SIZE, LMStudioClient, strip_reasoning


def client(settings, fake):
    return LMStudioClient(settings, transport=fake.transport())


def test_list_models(settings):
    assert client(settings, FakeLM()).list_models() == ["reader-m", "chat-m", "embed-m"]


def test_chat_uses_chat_model_and_strips_reasoning(settings):
    fake = FakeLM(chat_reply="<think>hmm</think>answer")
    assert client(settings, fake).chat([{"role": "user", "content": "hi"}]) == "answer"
    body = json.loads(fake.requests[0].content)
    assert body["model"] == "chat-m" and body["temperature"] == 0.0
    assert fake.requests[0].headers["authorization"] == "Bearer k"
    assert str(fake.requests[0].url) == "http://lm.test/v1/chat/completions"


def test_vision_payload(settings):
    fake = FakeLM()
    client(settings, fake).vision("read", b"\x89PNGdata", system="sys")
    body = json.loads(fake.requests[0].content)
    assert body["model"] == "reader-m"
    assert body["messages"][0] == {"role": "system", "content": "sys"}
    parts = body["messages"][1]["content"]
    assert parts[0] == {"type": "text", "text": "read"}
    expected = "data:image/png;base64," + base64.b64encode(b"\x89PNGdata").decode()
    assert parts[1]["image_url"]["url"] == expected


def test_vision_from_path_guesses_mime(settings, tmp_path):
    p = tmp_path / "a.jpg"
    p.write_bytes(b"jpg")
    fake = FakeLM()
    client(settings, fake).vision("read", p)
    url = json.loads(fake.requests[0].content)["messages"][0]["content"][1]["image_url"]["url"]
    assert url.startswith("data:image/jpeg;base64,")


def test_embed_batches_and_orders(settings):
    fake = FakeLM()
    n = EMBED_BATCH_SIZE + 5
    vectors = client(settings, fake).embed([f"t{i}" for i in range(n)])
    assert len(vectors) == n and len(fake.requests) == 2
    assert vectors[1] == [1.0, 1.0, 1.0]  # reversed server rows restored by index
    assert json.loads(fake.requests[0].content)["model"] == "embed-m"


def test_connection_error_wrapped(settings):
    def boom(request):
        raise httpx.ConnectError("refused")
    c = LMStudioClient(settings, transport=httpx.MockTransport(boom))
    with pytest.raises(LLMError, match="Is the server running"):
        c.list_models()


def test_http_error_wrapped(settings):
    c = LMStudioClient(settings, transport=httpx.MockTransport(lambda r: httpx.Response(500, text="bad")))
    with pytest.raises(LLMError, match="500"):
        c.chat([])


def test_bad_shape(settings):
    c = LMStudioClient(settings, transport=httpx.MockTransport(lambda r: httpx.Response(200, json={})))
    with pytest.raises(LLMError, match="Unexpected"):
        c.chat([])


@pytest.mark.parametrize("raw,expected", [
    ("<think>a\nb</think>\nText", "Text"),
    ("reasoning without open tag</think>Text", "Text"),
    ("plain", "plain"),
    ("<THINK>x</THINK>A<think>y</think>B", "AB"),
])
def test_strip_reasoning(raw, expected):
    assert strip_reasoning(raw) == expected
