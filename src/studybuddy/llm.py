"""LM Studio client (OpenAI-compatible HTTP API) for chat, vision and embeddings."""
from __future__ import annotations

import base64
import mimetypes
import re
from pathlib import Path
from typing import Any

import httpx

from .config import Settings
from .errors import LLMError

EMBED_BATCH_SIZE = 32

_THINK_BLOCK = re.compile(r"<(think|thinking|reasoning)>.*?</\1>", re.DOTALL | re.IGNORECASE)
_THINK_UNOPENED = re.compile(r"^.*?</(think|thinking|reasoning)>", re.DOTALL | re.IGNORECASE)


def strip_reasoning(text: str) -> str:
    """Remove model reasoning blocks (<think>...</think>), including a missing opening tag."""
    text = _THINK_BLOCK.sub("", text)
    text = _THINK_UNOPENED.sub("", text)
    return text.strip()


class LMStudioClient:
    def __init__(self, settings: Settings, timeout: float = 300.0,
                 transport: httpx.BaseTransport | None = None):
        self.settings = settings
        self._http = httpx.Client(
            base_url=settings.base_url,
            headers={"Authorization": f"Bearer {settings.api_key}"},
            timeout=timeout,
            transport=transport,
        )

    def close(self) -> None:
        self._http.close()

    def _request(self, method: str, path: str, **kwargs: Any) -> dict:
        try:
            resp = self._http.request(method, path.lstrip("/"), **kwargs)
        except httpx.TimeoutException as exc:
            raise LLMError(f"LM Studio timed out on {path}") from exc
        except httpx.HTTPError as exc:
            raise LLMError(
                f"Cannot reach LM Studio at {self.settings.base_url}. Is the server running?"
            ) from exc
        if resp.status_code >= 400:
            raise LLMError(f"LM Studio {path} returned {resp.status_code}: {resp.text[:300]}")
        try:
            return resp.json()
        except ValueError as exc:
            raise LLMError(f"LM Studio {path} returned non-JSON: {resp.text[:200]}") from exc

    def list_models(self) -> list[str]:
        data = self._request("GET", "/models")
        return [m["id"] for m in data.get("data", [])]

    def _complete(self, model: str, messages: list[dict], temperature: float,
                  max_tokens: int | None) -> str:
        body: dict[str, Any] = {"model": model, "messages": messages, "temperature": temperature}
        if max_tokens:
            body["max_tokens"] = max_tokens
        data = self._request("POST", "/chat/completions", json=body)
        try:
            content = data["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as exc:
            raise LLMError(f"Unexpected chat response shape: {str(data)[:200]}") from exc
        return strip_reasoning(content or "")

    def chat(self, messages: list[dict], temperature: float = 0.0,
             max_tokens: int | None = None) -> str:
        return self._complete(self.settings.chat_model, messages, temperature, max_tokens)

    def vision(self, prompt: str, image: bytes | Path | str, system: str | None = None,
               temperature: float = 0.0, max_tokens: int | None = None,
               mime: str | None = None) -> str:
        """Send one image + prompt to the READER model."""
        if isinstance(image, (str, Path)):
            path = Path(image)
            mime = mime or mimetypes.guess_type(path.name)[0]
            image = path.read_bytes()
        mime = mime or "image/png"
        url = f"data:{mime};base64,{base64.b64encode(image).decode('ascii')}"
        messages: list[dict] = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": [
            {"type": "text", "text": prompt},
            {"type": "image_url", "image_url": {"url": url}},
        ]})
        return self._complete(self.settings.reader_model, messages, temperature, max_tokens)

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Embed chunks for storage, adding EMBED_DOC_PREFIX."""
        prefix = self.settings.embed_doc_prefix
        return self.embed([prefix + t for t in texts])

    def embed_queries(self, texts: list[str]) -> list[list[float]]:
        """Embed questions for search, adding EMBED_QUERY_PREFIX."""
        prefix = self.settings.embed_query_prefix
        return self.embed([prefix + t for t in texts])

    def embed(self, texts: list[str]) -> list[list[float]]:
        """Raw embedding with no prefix; prefer embed_documents / embed_queries."""
        out: list[list[float]] = []
        for i in range(0, len(texts), EMBED_BATCH_SIZE):
            batch = texts[i:i + EMBED_BATCH_SIZE]
            data = self._request("POST", "/embeddings",
                                 json={"model": self.settings.embed_model, "input": batch})
            try:
                rows = sorted(data["data"], key=lambda r: r["index"])
                vectors = [r["embedding"] for r in rows]
            except (KeyError, TypeError) as exc:
                raise LLMError(f"Unexpected embeddings response shape: {str(data)[:200]}") from exc
            if len(vectors) != len(batch):
                raise LLMError(f"Asked for {len(batch)} embeddings, got {len(vectors)}")
            out.extend(vectors)
        return out
