"""The look: colour contrast of the real token values, no outside requests, and a few rules the CSS must keep."""
import asyncio
import re

import httpx
import pytest

from studybuddy.ui.offline import StripExternalScripts, strip_external
from studybuddy.ui.theme import CSS_PATH, build_theme, css

CSS_TEXT = css()


def tokens(block_start: str) -> dict:
    """The custom properties of the first CSS block that starts with `block_start`."""
    start = CSS_TEXT.index(block_start)
    body = CSS_TEXT[CSS_TEXT.index("{", start) + 1:CSS_TEXT.index("}", start)]
    return dict(re.findall(r"--([a-z-]+):\s*(#[0-9A-Fa-f]{6})", body))


LIGHT = tokens(":root {")
DARK = tokens(':root[data-theme="dark"]')


def luminance(hex_colour: str) -> float:
    r, g, b = (int(hex_colour.lstrip("#")[i:i + 2], 16) / 255 for i in (0, 2, 4))
    lin = lambda c: c / 12.92 if c <= 0.03928 else ((c + 0.055) / 1.055) ** 2.4
    return 0.2126 * lin(r) + 0.7152 * lin(g) + 0.0722 * lin(b)


def contrast(a: str, b: str) -> float:
    hi, lo = sorted((luminance(a), luminance(b)), reverse=True)
    return (hi + 0.05) / (lo + 0.05)


# (foreground token or hex, background token, minimum ratio): text needs 4.5, shapes and focus 3
TEXT_PAIRS = [("ink", "bg"), ("ink", "surface"), ("ink", "sunk"), ("muted", "bg"), ("muted", "surface"), ("muted", "sunk"),
              ("accent-text", "bg"), ("accent-text", "surface"), ("accent-text", "sunk"), ("#FFFFFF", "accent-fill"),
              ("#FFFFFF", "accent-fill-hover"), ("warn", "warn-bg"), ("danger", "bg"), ("danger", "surface")]
SHAPE_PAIRS = [("line-strong", "bg"), ("line-strong", "surface"), ("accent-text", "bg")]


def pick(palette, name):
    return name if name.startswith("#") else palette[name]


@pytest.mark.parametrize("palette_name,palette", [("light", LIGHT), ("dark", DARK)])
def test_text_pairs_meet_aa(palette_name, palette):
    for fg, bg in TEXT_PAIRS:
        ratio = contrast(pick(palette, fg), pick(palette, bg))
        assert ratio >= 4.5, f"{palette_name}: {fg} on {bg} is {ratio:.2f}:1"


@pytest.mark.parametrize("palette_name,palette", [("light", LIGHT), ("dark", DARK)])
def test_shapes_and_focus_meet_three_to_one(palette_name, palette):
    for fg, bg in SHAPE_PAIRS:
        ratio = contrast(pick(palette, fg), pick(palette, bg))
        assert ratio >= 3.0, f"{palette_name}: {fg} on {bg} is {ratio:.2f}:1"


def test_the_raw_terracotta_is_never_used_for_small_text_or_button_labels():
    assert contrast(LIGHT["accent"], LIGHT["bg"]) < 4.5      # which is why --accent-text exists
    for line in CSS_TEXT.splitlines():
        if re.match(r"\s*color:\s*var\(--accent\)", line):
            pytest.fail(f"text colour uses the raw accent: {line.strip()}")


def test_the_dark_palette_has_every_token_the_light_one_has():
    assert set(LIGHT) == set(DARK)


def test_fonts_are_system_stacks_with_no_web_font_or_import():
    assert "@font-face" not in CSS_TEXT and "@import" not in CSS_TEXT and "http" not in CSS_TEXT
    assert "Palatino Linotype" in CSS_TEXT and "system-ui" in CSS_TEXT
    theme_css = build_theme()._get_theme_css()
    assert "googleapis" not in theme_css and "gstatic" not in theme_css and "@font-face" not in theme_css


def test_the_css_asks_for_reduced_motion_and_keeps_a_visible_focus_ring():
    assert "prefers-reduced-motion: reduce" in CSS_TEXT and ":focus-visible" in CSS_TEXT and "outline: 2px solid" in CSS_TEXT
    assert CSS_PATH.is_file() and "outline: none" not in CSS_TEXT and "outline: 0" not in CSS_TEXT


def test_corner_radii_are_a_scale_not_one_value():
    radii = set(re.findall(r"--r-(?:sm|md|lg|xl|pill):\s*(\d+)px", CSS_TEXT))
    assert len(radii) >= 4


# ---- no request to the outside world ------------------------------------------------------------------------------

def test_outside_scripts_and_links_are_removed_from_html():
    page = (b'<head><script src="https://cdnjs.cloudflare.com/ajax/libs/x.js"></script>'
            b'<link rel="stylesheet" href="//fonts.example.org/a.css"><script src="/static/app.js"></script>'
            b'<link rel="icon" href="http://127.0.0.1:7860/favicon.ico"></head>')
    cleaned = strip_external(page)
    assert b"cloudflare" not in cleaned and b"fonts.example" not in cleaned
    assert b'src="/static/app.js"' in cleaned and b"127.0.0.1" in cleaned


def test_the_middleware_fixes_the_length_and_leaves_other_responses_alone():
    html = b'<html><script src="https://cdn.example.com/a.js"></script><body>hi</body></html>'
    sent = []

    async def app(scope, receive, send):
        kind = b"text/html; charset=utf-8" if scope["path"] == "/" else b"application/json"
        body = html if scope["path"] == "/" else b'{"url": "https://cdn.example.com/a.js"}'
        await send({"type": "http.response.start", "status": 200, "headers": [(b"content-type", kind), (b"content-length", str(len(body)).encode())]})
        await send({"type": "http.response.body", "body": body})

    async def get(path):
        sent.clear()
        await StripExternalScripts(app)({"type": "http", "path": path}, None, lambda m: _collect(m))
        return list(sent)

    async def _collect(message):
        sent.append(message)

    page = asyncio.run(get("/"))
    body = page[-1]["body"]
    assert b"cdn.example.com" not in body and body.endswith(b"<body>hi</body></html>")
    assert dict(page[0]["headers"])[b"content-length"] == str(len(body)).encode()
    other = asyncio.run(get("/config"))
    assert b"cdn.example.com" in other[-1]["body"]


def test_the_served_page_and_its_first_requests_never_leave_the_machine(tmp_path, settings, monkeypatch):
    import socket
    import traceback

    from test_ui_app import Served

    looked_up = []
    real = socket.getaddrinfo

    def spy(host, *args, **kwargs):
        if "gradio_client" not in "".join(traceback.format_stack()):     # the test's own client may phone home; the app must not
            looked_up.append(host)
        return real(host, *args, **kwargs)

    monkeypatch.setenv("GRADIO_ANALYTICS_ENABLED", "False")      # what python -m studybuddy sets: Gradio's own usage ping and version check
    monkeypatch.setattr(socket, "getaddrinfo", spy)
    app = Served(tmp_path, settings)
    try:
        page = httpx.get(app.url).text
        assert not re.findall(r"""(?:src|href)=["'](?:https?:)?//(?!127\.0\.0\.1|localhost)""", page)
        config = httpx.get(app.url + "config").text
        assert "googleapis" not in config and "cdnjs" not in config
    finally:
        app.close()
    outside = sorted({h for h in looked_up if h and h not in ("127.0.0.1", "localhost", "::1", "0.0.0.0")})
    assert outside == [], outside
