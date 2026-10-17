import json

import httpx
import pytest

from studybuddy.config import Settings


@pytest.fixture
def settings(tmp_path):
    return Settings(
        base_url="http://lm.test/v1", api_key="k", reader_model="reader-m", chat_model="chat-m",
        embed_model="embed-m", library_dir=tmp_path / "lib", data_dir=tmp_path / "data",
        subjects_file=tmp_path / "subjects.json", tag_timeout_seconds=300,
        telegram_token=None, telegram_user_id=None,
    )


class FakeLM:
    """Mock LM Studio: records requests, answers /models, /chat/completions, /embeddings."""

    def __init__(self, chat_reply="hello", dim=3, models=("reader-m", "chat-m", "embed-m")):
        self.requests: list[httpx.Request] = []
        self.chat_reply, self.dim, self.models = chat_reply, dim, list(models)

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        path = request.url.path
        if path.endswith("/models"):
            return httpx.Response(200, json={"data": [{"id": m} for m in self.models]})
        body = json.loads(request.content)
        if path.endswith("/chat/completions"):
            return httpx.Response(200, json={"choices": [{"message": {"content": self.chat_reply}}]})
        if path.endswith("/embeddings"):
            data = [{"index": i, "embedding": [float(i)] * self.dim} for i in range(len(body["input"]))]
            return httpx.Response(200, json={"data": data[::-1]})
        return httpx.Response(404, text="nope")

    def transport(self):
        return httpx.MockTransport(self)
