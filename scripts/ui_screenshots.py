"""Development tool: screenshot every screen at phone and desktop width, light and dark, and run layout checks.

  python scripts/ui_screenshots.py [OUT_DIR] [--only NAME ...]

Needs `pip install playwright` (development only, NOT in requirements.txt) and Microsoft Edge (it drives the installed Edge, so no
browser download). Serves scripts/ui_demo.py: SYNTHETIC papers and a fake model, in a temporary folder. Never your real data.
Checks for every scene: no sideways scrolling, touch targets of at least 44 px, no request to any host except 127.0.0.1, no
console errors. Prints a table; exit code 1 if a check failed. Images go to OUT_DIR (default: a temporary folder; never the repo).
"""
from __future__ import annotations

import argparse
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from ui_demo import build_demo  # noqa: E402

SIZES = {"phone": (390, 844), "desktop": (1280, 800)}
MODES = ("light", "dark")
MIN_TARGET = 44

JS_OVERFLOW = "() => document.documentElement.scrollWidth - document.documentElement.clientWidth"
JS_SMALL_TARGETS = """
(min) => [...document.querySelectorAll('#sb-root button, #sb-root [role=tab], #sb-root a[href], #sb-root input:not([type=hidden]):not([type=file]), #sb-root textarea')]
  .filter(e => { const t = (e.matches('input') && e.closest('label')) || (e.matches('input') && e.closest('.wrap')) || e; const r = t.getBoundingClientRect(); const s = getComputedStyle(e);
                 return !e.closest('.visually-hidden') && r.width > 0 && r.height > 0 && s.visibility !== 'hidden' && s.display !== 'none' && (r.height < min - 0.5 || r.width < min - 0.5); })
  .map(e => (e.id || e.getAttribute('aria-label') || e.textContent || e.tagName).trim().slice(0, 40) + ' ' + Math.round(e.getBoundingClientRect().width) + 'x' + Math.round(e.getBoundingClientRect().height))
"""


def wait_app(page) -> None:
    page.wait_for_selector("#sb-root", timeout=30000)
    page.wait_for_timeout(600)


def tab(page, name: str) -> None:
    page.get_by_role("tab", name=name, exact=True).click()
    page.wait_for_timeout(500)


# ---- scenes: each takes (page, demo) and leaves the screen ready to photograph -------------------------------------------

def scene_ask_empty(page, demo):
    tab(page, "Ask")


def scene_library(page, demo):
    tab(page, "Library")


def scene_add(page, demo):
    tab(page, "Add")


def ask(page, text: str):
    page.locator("#sb-question textarea").fill(text)
    page.locator("#sb-send").click()


def scene_ask_answer(page, demo):
    tab(page, "Ask")
    ask(page, "How does water evaporate in the water cycle?")
    page.wait_for_selector("#sb-chips label", state="visible", timeout=30000)
    page.locator("#sb-chips label").first.click()
    page.wait_for_selector("#sb-slip-image img", state="visible", timeout=15000)
    page.wait_for_timeout(600)


def scene_ask_refusal(page, demo):
    tab(page, "Ask")
    ask(page, "What is the capital of France?")
    page.wait_for_selector("#sb-closest", state="visible", timeout=30000)
    page.locator("#sb-closest").click()
    page.wait_for_timeout(500)


def scene_ask_which(page, demo):
    tab(page, "Ask")
    ask(page, "give me a summary of the entire paper")
    page.wait_for_selector("#sb-which label", state="visible", timeout=30000)


def scene_library_paper(page, demo):
    tab(page, "Library")
    page.locator(".sb-row").nth(1).click()
    page.wait_for_selector("#sb-paper-head h2", state="visible", timeout=15000)
    page.wait_for_timeout(600)


def scene_library_remove(page, demo):
    scene_library_paper(page, demo)
    page.get_by_text("More actions").click()
    page.get_by_role("button", name="Remove from Study Buddy").click()
    page.wait_for_selector("#sb-confirm", state="visible", timeout=10000)


def scene_library_search(page, demo):
    tab(page, "Library")
    page.locator("#sb-search textarea, #sb-search input").first.fill("fraction")
    page.wait_for_timeout(800)


def scene_add_done(page, demo):
    from ui_demo import draw_page
    tab(page, "Add")
    photo = draw_page(demo.root / "incoming" / "my photo.jpg", "New worksheet", ["A freshly photographed page."])
    page.locator("#sb-drop input[type=file]").set_input_files(str(photo))
    page.wait_for_timeout(800)
    page.locator("#sb-add-btn").click()
    page.wait_for_selector("#sb-added .sb-outcome", timeout=15000)
    page.wait_for_timeout(500)


def submit_demo_files(demo, count: int = 1):
    from ui_demo import draw_page
    for number in range(count):
        photo = draw_page(demo.root / "incoming" / f"new-{number}.jpg", f"New page {number}", ["A page that is being read."])
        demo.rt.service.submit([photo], subject="Math", source_kind="photo")


def open_activity(page):
    page.wait_for_selector("#sb-pill", state="visible", timeout=20000)
    page.locator("#sb-pill").click()
    page.wait_for_selector("#sb-activity", state="visible", timeout=10000)
    page.wait_for_timeout(500)


def scene_activity_reading(page, demo):
    submit_demo_files(demo, 2)
    open_activity(page)


def scene_activity_failed(page, demo):
    submit_demo_files(demo, 1)
    demo.rt.service.wait_idle(30)
    open_activity(page)
    page.get_by_text("Technical details").click()
    page.wait_for_timeout(400)


def scene_activity_waiting(page, demo):
    submit_demo_files(demo, 1)
    open_activity(page)


def scene_banner(page, demo):
    tab(page, "Ask")
    page.wait_for_selector(".sb-warning", state="visible", timeout=10000)


SCENES = {
    "banner": scene_banner,
    "activity-reading": scene_activity_reading,
    "activity-failed": scene_activity_failed,
    "activity-waiting": scene_activity_waiting,
    "add-done": scene_add_done,
    "library-paper": scene_library_paper,
    "library-remove": scene_library_remove,
    "library-search": scene_library_search,
    "ask-empty": scene_ask_empty,
    "ask-answer": scene_ask_answer,
    "ask-refusal": scene_ask_refusal,
    "ask-which": scene_ask_which,
    "library": scene_library,
    "library-empty": scene_library,
    "add": scene_add,
}


def safe(text: str) -> str:
    return text.encode("ascii", "replace").decode()


# Gradio keeps upload bookkeeping in module globals bound to the first event loop, so a scene that uploads runs each variant in its own process.
ISOLATE = {"add-done"}


def run(out: Path, only: list[str] | None, variant: str | None = None) -> int:
    from playwright.sync_api import sync_playwright

    failures: list[str] = []
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    with tempfile.TemporaryDirectory(prefix="sb-shots-", ignore_cleanup_errors=True) as tmp, sync_playwright() as pw:
        browser = pw.chromium.launch(channel="msedge", headless=True)
        for name, scene in SCENES.items():
            if only and name not in only:
                continue
            for size_name, (width, height) in SIZES.items():
                for mode in MODES:
                    if variant and variant != f"{size_name}-{mode}":
                        continue
                    if name in ISOLATE and not variant:
                        import subprocess
                        done = subprocess.run([sys.executable, __file__, str(out), "--only", name, "--variant", f"{size_name}-{mode}"])
                        failures += [f"{name} {size_name} {mode}"] if done.returncode else []
                        continue
                    root = Path(tmp) / f"{name}-{size_name}-{mode}"
                    demo = build_demo(root, papers=name != "library-empty", busy=name == "activity-reading",
                                      reader="down" if name == "activity-waiting" else "fails",
                                      host="0.0.0.0" if name == "banner" else "127.0.0.1")
                    context = browser.new_context(viewport={"width": width, "height": height}, color_scheme=mode, device_scale_factor=1)
                    page = context.new_page()
                    hosts, errors = set(), []
                    page.on("request", lambda r: hosts.add(r.url.split("/")[2] if "://" in r.url else r.url[:20]))
                    page.on("console", lambda m: errors.append(m.text) if m.type == "error" else None)
                    page.on("pageerror", lambda e: errors.append(str(e)))
                    try:
                        page.goto(demo.url)
                        wait_app(page)
                        scene(page, demo)
                        page.wait_for_timeout(300)
                        overflow = page.evaluate(JS_OVERFLOW)
                        small = page.evaluate(JS_SMALL_TARGETS, MIN_TARGET)
                        outside = sorted(h for h in hosts if not h.startswith(("127.0.0.1", "localhost", "data:", "blob:")))
                        problems = []
                        if overflow > 0:
                            problems.append(f"sideways scroll {overflow}px")
                        if small:
                            problems.append("small targets: " + "; ".join(small[:6]))
                        if outside:
                            problems.append("outside requests: " + ", ".join(outside))
                        real_errors = [e for e in errors if "favicon" not in e and "manifest" not in e]
                        if real_errors:
                            problems.append("console: " + real_errors[0][:120])
                        page.screenshot(path=str(out / f"{name}-{size_name}-{mode}.png"), full_page=True)
                        rows.append((name, size_name, mode, "ok" if not problems else "; ".join(problems)))
                        failures += [f"{name} {size_name} {mode}: {p}" for p in problems]
                    finally:
                        context.close()
                        demo.close()
        browser.close()
    for row in rows:
        print(safe(f"{row[0]:22} {row[1]:8} {row[2]:6} {row[3]}"))
    print(f"\nImages in {out}")
    return 1 if failures else 0


# ---- behaviour checks: theme toggle, system theme, keyboard focus, reduced motion ---------------------------------------------

JS_FOCUS = r"""
() => { const e = document.activeElement; if (!e || e === document.body) return null;
  const own = getComputedStyle(e); const label = e.closest('label'); const lab = label ? getComputedStyle(label) : null;
  const ring = (s) => s && s.outlineStyle !== 'none' && parseFloat(s.outlineWidth) >= 2;
  return { name: (e.id || e.getAttribute('aria-label') || e.textContent || e.tagName).trim().slice(0, 40), ok: ring(own) || ring(lab) }; }
"""


def run_checks() -> int:
    from playwright.sync_api import sync_playwright

    results: list[tuple[str, bool, str]] = []

    def record(name: str, ok: bool, detail: str = "") -> None:
        results.append((name, bool(ok), detail))

    with tempfile.TemporaryDirectory(prefix="sb-checks-", ignore_cleanup_errors=True) as tmp, sync_playwright() as pw:
        demo = build_demo(Path(tmp))
        browser = pw.chromium.launch(channel="msedge", headless=True)
        try:
            # 1. the system's dark setting is followed without a click
            ctx = browser.new_context(color_scheme="dark", viewport={"width": 1280, "height": 800})
            page = ctx.new_page()
            page.goto(demo.url)
            wait_app(page)
            record("follows a dark system setting", page.evaluate("document.documentElement.dataset.theme") == "dark")
            # 2. the toggle switches, and the choice is remembered after a reload
            page.locator("#sb-theme").click()
            page.wait_for_timeout(200)
            record("toggle switches to light", page.evaluate("document.documentElement.dataset.theme") == "light")
            page.reload()
            wait_app(page)
            record("the choice is remembered", page.evaluate("document.documentElement.dataset.theme") == "light")
            bg = page.evaluate("getComputedStyle(document.body).backgroundColor")
            record("light background is cream", bg == "rgb(250, 249, 245)", bg)
            page.locator("#sb-theme").click()
            page.wait_for_timeout(200)
            bg = page.evaluate("getComputedStyle(document.body).backgroundColor")
            record("dark background is warm dark", bg == "rgb(31, 30, 27)", bg)
            ctx.close()

            # 3. keyboard: every stop has a visible focus ring, and a chip can be chosen without a mouse
            ctx = browser.new_context(viewport={"width": 390, "height": 844})
            page = ctx.new_page()
            page.goto(demo.url)
            wait_app(page)
            page.locator("#sb-question textarea").fill("How does water evaporate in the water cycle?")
            page.locator("#sb-send").click()
            page.wait_for_selector("#sb-chips label", state="visible", timeout=30000)
            page.locator("body").click(position={"x": 5, "y": 5})
            bad, stops = [], 0
            for _ in range(24):
                page.keyboard.press("Tab")
                info = page.evaluate(JS_FOCUS)
                if info:
                    stops += 1
                    if not info["ok"]:
                        bad.append(info["name"])
            record("every keyboard stop shows a focus ring", stops > 8 and not bad, f"{stops} stops; without a ring: {sorted(set(bad))[:5]}")
            page.locator("#sb-chips label input").first.focus()
            page.keyboard.press("Space")
            try:
                page.wait_for_selector("#sb-slip-image img", state="visible", timeout=10000)
                record("a source chip opens with the keyboard", True)
            except Exception:
                record("a source chip opens with the keyboard", False)
            ctx.close()

            # 4. reduced motion: nothing animates
            ctx = browser.new_context(reduced_motion="reduce", viewport={"width": 1280, "height": 800})
            page = ctx.new_page()
            page.goto(demo.url)
            wait_app(page)
            page.locator("#sb-question textarea").fill("How does water evaporate in the water cycle?")
            page.locator("#sb-send").click()
            page.wait_for_selector("#sb-chips label", state="visible", timeout=30000)
            page.locator("#sb-chips label").first.click()
            page.wait_for_selector("#sb-slip-image img", state="visible", timeout=10000)
            animation = page.evaluate("getComputedStyle(document.getElementById('sb-slip')).animationName")
            record("reduced motion: the slip does not animate", animation == "none", animation)
            ctx.close()
        finally:
            browser.close()
            demo.close()
    for name, ok, detail in results:
        print(safe(f"{'ok ' if ok else 'FAIL'} {name}{'  ' + detail if detail and not ok else ''}"))
    return 0 if all(ok for _, ok, _ in results) else 1


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("out", nargs="?", default=None)
    parser.add_argument("--only", nargs="*")
    parser.add_argument("--checks", action="store_true", help="run the behaviour checks (theme toggle, focus ring, reduced motion) instead of screenshots")
    parser.add_argument("--variant", default=None, help="size-mode, e.g. phone-dark (used internally for isolated scenes)")
    args = parser.parse_args()
    if args.checks:
        return run_checks()
    out = Path(args.out) if args.out else Path(tempfile.gettempdir()) / "sb-screenshots"
    return run(out, args.only, args.variant)


if __name__ == "__main__":
    sys.exit(main())
