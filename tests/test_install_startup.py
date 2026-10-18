import shutil
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "install_startup.ps1"
pytestmark = pytest.mark.skipif(sys.platform != "win32" or shutil.which("powershell") is None, reason="Windows PowerShell only")


def run(action, folder):
    done = subprocess.run(["powershell", "-NoProfile", "-ExecutionPolicy", "Bypass", "-File", str(SCRIPT), "-Action", action, "-Folder", str(folder)],
                          capture_output=True, text=True, timeout=60)
    return done.returncode, done.stdout + done.stderr


def test_add_status_and_remove_use_only_the_folder_they_are_given(tmp_path):
    startup = tmp_path / "Startup"
    code, text = run("status", startup)
    assert code == 0 and "Not installed" in text and not startup.exists()          # looking creates nothing
    code, text = run("add", startup)
    link = startup / "Study Buddy.lnk"
    assert code == 0 and link.is_file() and "shell:startup" in text
    code, text = run("status", startup)
    assert "Installed" in text and "start.bat" in text and "wait" in text and str(ROOT) in text
    (startup / "other.lnk").write_text("keep me")
    code, text = run("remove", startup)
    assert code == 0 and not link.exists() and (startup / "other.lnk").read_text() == "keep me"   # only its own shortcut goes
    assert "Nothing to remove" in run("remove", startup)[1]


def test_the_script_never_touches_the_real_startup_folder_without_being_run():
    text = SCRIPT.read_text(encoding="utf-8")
    assert "Mandatory = $true" in text and "'add', 'remove', 'status'" in text
