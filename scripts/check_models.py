"""Check directories and the READER / CHAT / EMBED roles against the running LM Studio."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from studybuddy.checks import exit_code, run_checks  # noqa: E402
from studybuddy.config import load_settings  # noqa: E402
from studybuddy.errors import ConfigError  # noqa: E402
from studybuddy.llm import LMStudioClient  # noqa: E402


def main() -> int:
    try:
        settings = load_settings()
    except ConfigError as exc:
        print(f"[FAIL] config: {exc}")
        return 1
    print(f"LM Studio: {settings.base_url}")
    client = LMStudioClient(settings)
    try:
        results = run_checks(settings, client)
    finally:
        client.close()
    for r in results:
        print(f"[{r.status}] {r.name}: {r.detail}")
    return exit_code(results)


if __name__ == "__main__":
    sys.exit(main())
