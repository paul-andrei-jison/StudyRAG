"""Point this clone's git at scripts/git-hooks (local config only, nothing is pushed)."""
import subprocess
import sys
from pathlib import Path

root = Path(__file__).resolve().parents[1]
hook = root / "scripts" / "git-hooks" / "commit-msg"
hook.chmod(hook.stat().st_mode | 0o111)
subprocess.run(["git", "config", "core.hooksPath", "scripts/git-hooks"], cwd=root, check=True)
print("Installed: git now runs scripts/git-hooks/commit-msg for this clone.")
sys.exit(0)
