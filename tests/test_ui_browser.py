"""Real-browser checks (theme toggle, focus ring, reduced motion, phone layout). Off by default: they need Microsoft Edge and the
development-only `playwright` package (see scripts/ui_screenshots.py). Run them with:  SB_BROWSER_TESTS=1 pytest tests/test_ui_browser.py"""
import importlib.util
import os
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
pytestmark = pytest.mark.skipif(not os.environ.get("SB_BROWSER_TESTS") or importlib.util.find_spec("playwright") is None,
                                reason="set SB_BROWSER_TESTS=1 and install playwright (development only) to run the browser checks")


def run(*args):
    return subprocess.run([sys.executable, str(ROOT / "scripts" / "ui_screenshots.py"), *args], capture_output=True, text=True, timeout=900)


def test_behaviour_checks_pass():
    done = run("--checks")
    assert done.returncode == 0, done.stdout[-1500:]


def test_every_screen_fits_a_phone_and_has_44px_targets_and_makes_no_outside_request(tmp_path):
    done = run(str(tmp_path))
    assert done.returncode == 0, "\n".join(line for line in done.stdout.splitlines() if not line.endswith(" ok"))[-2000:]
