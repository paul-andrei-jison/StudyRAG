import json

import httpx
import pytest

from studybuddy.errors import ContextOverflowError, LLMError, TransientLLMError
from studybuddy.llm import LMStudioClient, ReasoningStripper, StreamEnd


def sse(deltas, usage=None, finish="stop", done=True, extra_lines=()):
    lines = [f"data: {json.dumps({'choices': [{'delta': {'content': d}, 'finish_reason': None}]})}\n\n" for d in deltas]
    lines.append(f"data: {json.dumps({'choices': [{'delta': {}, 'finish_reason': finish}]})}\n\n")
    if usage:
        lines.append(f"data: {json.dumps({'choices': [], 'usage': usage})}\n\n")
    lines.extend(extra_lines)
    if done:
        lines.append("data: [DONE]\n\n")
    return "".join(lines).encode()


def client(settings, handler):
    return LMStudioClient(settings, transport=httpx.MockTransport(handler))


def run(c, messages=None):
    items = list(c.chat_stream(messages or [{"role": "user", "content": "hi"}]))
    text = "".join(i for i in items if isinstance(i, str))
    return text, items[-1], items


def test_stream_yields_text_pieces_then_the_end_with_usage(settings):
    usage = {"prompt_tokens": 24, "completion_tokens": 3}
    c = client(settings, lambda r: httpx.Response(200, content=sse(["Photo", "synthesis", " works."], usage=usage)))
    text, end, items = run(c)
    assert text == "Photosynthesis works." and items[:3] == ["Photo", "synthesis", " works."]
    assert isinstance(end, StreamEnd) and end.usage == usage and end.finish_reason == "stop"


def test_request_asks_for_a_stream_with_usage_and_uses_the_chat_model(settings):
    seen = {}

    def handler(request):
        seen.update(json.loads(request.content))
        seen["auth"] = request.headers["authorization"]
        return httpx.Response(200, content=sse(["ok"]))

    c = client(settings, handler)
    list(c.chat_stream([{"role": "user", "content": "q"}], temperature=0.0, max_tokens=700))
    assert seen["stream"] is True and seen["stream_options"] == {"include_usage": True}
    assert seen["model"] == "chat-m" and seen["max_tokens"] == 700 and seen["temperature"] == 0.0
    assert seen["auth"] == "Bearer k"


def test_empty_and_missing_content_deltas_are_ignored(settings):
    body = (b'data: {"choices": [{"delta": {"role": "assistant"}}]}\n\n'
            b'data: {"choices": [{"delta": {"content": null}}]}\n\n'
            b'data: {"choices": [{"delta": {"reasoning_content": "thinking..."}}]}\n\n'
            b'data: {"choices": [{"delta": {"content": "Answer"}, "finish_reason": "stop"}]}\n\n'
            b"data: [DONE]\n\n")
    text, end, _ = run(client(settings, lambda r: httpx.Response(200, content=body)))
    assert text == "Answer" and end.finish_reason == "stop"


def test_reasoning_blocks_are_removed_from_the_stream(settings):
    deltas = ["<thi", "nk>I should look at source 1</th", "ink>", "The answer", " is 71%."]
    text, _, _ = run(client(settings, lambda r: httpx.Response(200, content=sse(deltas))))
    assert text == "The answer is 71%."


def test_a_truncated_answer_is_reported_through_finish_reason(settings):
    text, end, _ = run(client(settings, lambda r: httpx.Response(200, content=sse(["half an ans"], finish="length"))))
    assert text == "half an ans" and end.finish_reason == "length"


def test_garbage_and_comment_lines_in_the_stream_are_skipped(settings):
    body = sse(["fine"], extra_lines=(": keep-alive\n\n", "data: {not json}\n\n", "event: ping\n\n"))
    assert run(client(settings, lambda r: httpx.Response(200, content=body)))[0] == "fine"


class DroppedStream(httpx.SyncByteStream):
    def __init__(self, chunks):
        self.chunks = chunks

    def __iter__(self):
        yield from self.chunks
        raise httpx.ReadError("connection reset by peer")


def test_a_dropped_connection_keeps_the_text_so_far_then_raises(settings):
    first = ("data: " + json.dumps({"choices": [{"delta": {"content": "Earth has a magn"}}]}) + "\n\n").encode()
    c = client(settings, lambda r: httpx.Response(200, stream=DroppedStream([first])))
    got = []
    with pytest.raises(TransientLLMError, match="dropped during the answer"):
        for item in c.chat_stream([{"role": "user", "content": "q"}]):
            got.append(item)
    assert got == ["Earth has a magn"]  # the caller has the partial text and knows it is incomplete


def test_a_stream_that_stops_without_finishing_is_an_error(settings):
    body = ("data: " + json.dumps({"choices": [{"delta": {"content": "partial"}}]}) + "\n\n").encode()
    c = client(settings, lambda r: httpx.Response(200, content=body))
    got = []
    with pytest.raises(TransientLLMError, match="ended the answer early"):
        for item in c.chat_stream([{"role": "user", "content": "q"}]):
            got.append(item)
    assert got == ["partial"]


def test_connection_refused_and_timeout_are_transient(settings):
    def refuse(request):
        raise httpx.ConnectError("refused")

    def slow(request):
        raise httpx.ReadTimeout("too slow")

    with pytest.raises(TransientLLMError, match="dropped|reach"):
        list(client(settings, refuse).chat_stream([]))
    with pytest.raises(TransientLLMError, match="timed out"):
        list(client(settings, slow).chat_stream([]))


@pytest.mark.parametrize("status,error", [(503, TransientLLMError), (429, TransientLLMError), (404, LLMError)])
def test_error_statuses_before_the_stream_starts(settings, status, error):
    with pytest.raises(error):
        list(client(settings, lambda r: httpx.Response(status, text="nope")).chat_stream([]))


def test_a_context_overflow_before_streaming_has_its_own_error(settings):
    msg = "n_keep: 40009>= n_ctx: 13824). Try to load the model with a larger context length"
    with pytest.raises(ContextOverflowError):
        list(client(settings, lambda r: httpx.Response(400, json={"error": msg})).chat_stream([]))


# ---- the reasoning stripper on its own -----------------------------------------------------

def strip_stream(pieces):
    s = ReasoningStripper()
    return "".join(s.feed(p) for p in pieces) + s.flush()


def test_stripper_is_correct_for_every_way_of_splitting_the_text():
    text = "Before <think>secret reasoning</think>After and <think>more</think>done."
    for i in range(len(text) + 1):
        for j in range(i, len(text) + 1):
            assert strip_stream([text[:i], text[i:j], text[j:]]) == "Before After and done."


def test_stripper_leaves_ordinary_angle_brackets_alone():
    assert strip_stream(["if a <b and c> d, then ", "x < 3 and <", "b>"]) == "if a <b and c> d, then x < 3 and <b>"
    assert strip_stream(["plain text"]) == "plain text" and strip_stream([]) == ""


def test_stripper_drops_an_unterminated_block_but_keeps_held_back_text_otherwise():
    assert strip_stream(["answer<think>never closed"]) == "answer"
    assert strip_stream(["ends with <thi"]) == "ends with <thi"


# ---- context length lookup -----------------------------------------------------------------

def models_handler(payload, status=200):
    def handler(request):
        assert request.url.path == "/api/v0/models"
        return httpx.Response(status, json=payload)
    return handler


def test_context_length_from_lm_studio(settings):
    data = {"data": [{"id": "embed-m", "state": "loaded", "loaded_context_length": 2048, "max_context_length": 2048},
                     {"id": "chat-m", "state": "loaded", "loaded_context_length": 13721, "max_context_length": 262144}]}
    c = client(settings, models_handler(data))
    assert c.context_length() == 13721 and c.context_length("embed-m") == 2048


def test_context_length_unknown_cases_are_none(settings):
    assert client(settings, models_handler({"data": [{"id": "chat-m", "state": "not-loaded"}]})).context_length() is None
    assert client(settings, models_handler({"data": []})).context_length() is None
    assert client(settings, models_handler({}, status=404)).context_length() is None
    assert client(settings, lambda r: httpx.Response(200, text="not json")).context_length() is None

    def refuse(request):
        raise httpx.ConnectError("refused")

    assert client(settings, refuse).context_length() is None


def test_context_length_falls_back_to_the_max_when_only_loaded_state_is_given(settings):
    data = {"data": [{"id": "chat-m", "state": "loaded", "max_context_length": 8192}]}
    assert client(settings, models_handler(data)).context_length() == 8192
