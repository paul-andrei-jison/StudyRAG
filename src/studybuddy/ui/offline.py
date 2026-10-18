"""Keep the web page from asking the internet for anything.

Gradio's page template loads one script from a public CDN (`iframeResizer`, only useful when the app is embedded in another site).
This app is private and must work offline, so this small ASGI middleware removes every `<script>` or `<link>` that points at an
outside address from the HTML Gradio serves. Everything else passes through untouched.
"""
from __future__ import annotations

import re

_EXTERNAL = re.compile(rb"<(?:script|link)\b[^>]*\b(?:src|href)\s*=\s*[\"']?(?:https?:)?//(?!127\.0\.0\.1|localhost)[^>]*>(?:\s*</script>)?", re.I)


def strip_external(html: bytes) -> bytes:
    return _EXTERNAL.sub(b"", html)


class StripExternalScripts:
    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        start: dict | None = None
        chunks: list[bytes] = []

        async def capture(message):
            nonlocal start
            if message["type"] == "http.response.start":
                headers = dict(message.get("headers") or [])
                if headers.get(b"content-type", b"").startswith(b"text/html"):
                    start = message                    # hold it back until the body is known
                    return
                await send(message)
            elif message["type"] == "http.response.body" and start is not None:
                chunks.append(message.get("body", b""))
                if message.get("more_body"):
                    return
                body = strip_external(b"".join(chunks))
                headers = [(k, v) for k, v in start["headers"] if k.lower() != b"content-length"]
                headers.append((b"content-length", str(len(body)).encode()))
                await send({**start, "headers": headers})
                await send({"type": "http.response.body", "body": body})
            else:
                await send(message)

        await self.app(scope, receive, capture)
