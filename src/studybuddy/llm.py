"""LM Studio client (OpenAI-compatible HTTP API) for chat, vision and embeddings."""
from __future__ import annotations

import base64
import json
import mimetypes
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator

import httpx

from .config import Settings
from .errors import ContextOverflowError, LLMError, TransientLLMError

EMBED_BATCH_SIZE = 32
TRANSIENT_STATUS = frozenset({408, 425, 429, 500, 502, 503, 504})

_THINK_BLOCK = re.compile(r"<(think|thinking|reasoning)>.*?</\1>", re.DOTALL | re.IGNORECASE)
_THINK_UNOPENED = re.compile(r"^.*?</(think|thinking|reasoning)>", re.DOTALL | re.IGNORECASE)


@dataclass(frozen=True)
class Completion:
    text: str
    finish_reason: str | None  # "stop" normally; "length" means the output was cut off


@dataclass(frozen=True)
class StreamEnd:
    """Last item of a chat stream."""
    usage: dict | None          # prompt_tokens / completion_tokens as reported by the server
    finish_reason: str | None   # "stop" normally; "length" means the answer was cut off


class ReasoningStripper:
    """Removes <think>...</think> blocks from a stream of text pieces, even when a tag is split across pieces."""

    OPEN, CLOSE = "<think>", "</think>"

    def __init__(self) -> None:
        self._buffer = ""
        self._inside = False

    def feed(self, piece: str) -> str:
        self._buffer += piece
        out: list[str] = []
        while True:
            if self._inside:
                end = self._buffer.find(self.CLOSE)
                if end < 0:
                    self._buffer = self._buffer[-(len(self.CLOSE) - 1):]  # keep a possible partial closing tag
                    break
                self._buffer = self._buffer[end + len(self.CLOSE):]
                self._inside = False
                continue
            start = self._buffer.find(self.OPEN)
            if start >= 0:
                out.append(self._buffer[:start])
                self._buffer = self._buffer[start + len(self.OPEN):]
                self._inside = True
                continue
            hold = self._partial_open_length()
            out.append(self._buffer[: len(self._buffer) - hold])
            self._buffer = self._buffer[len(self._buffer) - hold:]
            break
        return "".join(out)

    def _partial_open_length(self) -> int:
        for size in range(min(len(self.OPEN) - 1, len(self._buffer)), 0, -1):
            if self.OPEN.startswith(self._buffer[-size:]):
                return size
        return 0

    def flush(self) -> str:
        """Whatever is left at the end of the stream (text held back as a possible tag, unless inside a block)."""
        rest, self._buffer = ("" if self._inside else self._buffer), ""
        return rest


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
            raise TransientLLMError(f"LM Studio timed out on {path}") from exc
        except httpx.HTTPError as exc:
            raise TransientLLMError(
                f"Cannot reach LM Studio at {self.settings.base_url}. Is the server running?"
            ) from exc
        self._raise_for_status(path, resp.status_code, resp.text)
        try:
            return resp.json()
        except ValueError as exc:
            raise LLMError(f"LM Studio {path} returned non-JSON: {resp.text[:200]}") from exc

    @staticmethod
    def _raise_for_status(path: str, status: int, text: str) -> None:
        if status == 400 and "context length" in text.lower():
            raise ContextOverflowError(f"Input does not fit the model context: {text[:300]}")
        if status >= 400:
            message = f"LM Studio {path} returned {status}: {text[:300]}"
            if status in TRANSIENT_STATUS or (status == 400 and "no models loaded" in text.lower()):
                raise TransientLLMError(message)  # busy, restarting, or waiting for the student to load a model
            raise LLMError(message)

    def context_length(self, model: str | None = None) -> int | None:
        """The loaded context size of `model` (default: the CHAT model) from LM Studio's REST API, or None if unknown."""
        root = self.settings.base_url[:-3] if self.settings.base_url.endswith("/v1") else self.settings.base_url
        try:
            resp = self._http.get(f"{root}/api/v0/models")
            data = resp.json().get("data", []) if resp.status_code == 200 else []
        except (httpx.HTTPError, ValueError, AttributeError):
            return None
        wanted = model or self.settings.chat_model
        for entry in data:
            if isinstance(entry, dict) and entry.get("id") == wanted:
                size = entry.get("loaded_context_length") or (entry.get("max_context_length") if entry.get("state") == "loaded" else None)
                return int(size) if size else None
        return None

    def list_models(self) -> list[str]:
        data = self._request("GET", "/models")
        return [m["id"] for m in data.get("data", [])]

    def _complete(self, model: str, messages: list[dict], temperature: float,
                  max_tokens: int | None) -> Completion:
        body: dict[str, Any] = {"model": model, "messages": messages, "temperature": temperature}
        if max_tokens:
            body["max_tokens"] = max_tokens
        data = self._request("POST", "/chat/completions", json=body)
        try:
            choice = data["choices"][0]
            content = choice["message"]["content"]
        except (KeyError, IndexError, TypeError) as exc:
            raise LLMError(f"Unexpected chat response shape: {str(data)[:200]}") from exc
        return Completion(strip_reasoning(content or ""), choice.get("finish_reason"))

    def chat(self, messages: list[dict], temperature: float = 0.0,
             max_tokens: int | None = None) -> str:
        return self._complete(self.settings.chat_model, messages, temperature, max_tokens).text

    def chat_stream(self, messages: list[dict], temperature: float = 0.0,
                    max_tokens: int | None = None) -> Iterator[str | StreamEnd]:
        """Stream a CHAT answer: text pieces, then one StreamEnd (usage and finish reason).

        Reasoning blocks are removed. If the connection drops part way, the pieces already yielded stand and a
        TransientLLMError is raised after them, so a half answer is never mistaken for a complete one.
        """
        body: dict[str, Any] = {"model": self.settings.chat_model, "messages": messages, "temperature": temperature,
                                "stream": True, "stream_options": {"include_usage": True}}
        if max_tokens:
            body["max_tokens"] = max_tokens
        stripper, usage, finish, done = ReasoningStripper(), None, None, False
        try:
            with self._http.stream("POST", "chat/completions", json=body) as resp:
                if resp.status_code >= 400:
                    resp.read()
                    self._raise_for_status("/chat/completions", resp.status_code, resp.text)
                for line in resp.iter_lines():
                    if not line.startswith("data:"):
                        continue
                    payload = line[5:].strip()
                    if payload == "[DONE]":
                        done = True
                        break
                    try:
                        event = json.loads(payload)
                    except ValueError:
                        continue
                    if event.get("usage"):
                        usage = event["usage"]
                    for choice in event.get("choices") or []:
                        piece = stripper.feed((choice.get("delta") or {}).get("content") or "")
                        if piece:
                            yield piece
                        finish = choice.get("finish_reason") or finish
        except httpx.TimeoutException as exc:
            raise TransientLLMError("LM Studio timed out while streaming the answer") from exc
        except httpx.HTTPError as exc:
            raise TransientLLMError(f"The connection to LM Studio dropped during the answer ({type(exc).__name__})") from exc
        tail = stripper.flush()
        if tail:
            yield tail
        if not done and finish is None:
            raise TransientLLMError("LM Studio ended the answer early (the stream stopped without finishing)")
        yield StreamEnd(usage, finish)

    def vision(self, prompt: str, image: bytes | Path | str, system: str | None = None,
               temperature: float = 0.0, max_tokens: int | None = None,
               mime: str | None = None) -> str:
        """Send one image + prompt to the READER model; returns the text only."""
        return self.vision_full(prompt, image, system, temperature, max_tokens, mime).text

    def vision_full(self, prompt: str, image: bytes | Path | str, system: str | None = None,
                    temperature: float = 0.0, max_tokens: int | None = None,
                    mime: str | None = None) -> Completion:
        """Like vision(), but also reports finish_reason so callers can detect truncation."""
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
