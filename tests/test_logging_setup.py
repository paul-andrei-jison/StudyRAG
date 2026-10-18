import logging

import pytest

from studybuddy.config import load_settings
from studybuddy.errors import ConfigError
from studybuddy.logging_setup import LOG_FILENAME, setup_logging
from studybuddy.logsafe import install_log_safety

TOKEN = "123456789:AAHfake_token-value0123456789abcdefg"


def make_settings(tmp_path, **extra):
    env = {"SB_READER_MODEL": "r", "SB_CHAT_MODEL": "c", "SB_EMBED_MODEL": "e", "SB_LIBRARY_DIR": str(tmp_path / "lib"),
           "SB_DATA_DIR": str(tmp_path / "data"), "TELEGRAM_BOT_TOKEN": TOKEN, "SB_TELEGRAM_USER_ID": "5", **extra}
    return load_settings(env, repo_root=tmp_path / "repo")


@pytest.fixture
def logs(tmp_path):
    install_log_safety([TOKEN])
    root = logging.getLogger()
    before, level = list(root.handlers), root.level
    yield tmp_path
    for handler in [h for h in root.handlers if h not in before]:
        root.removeHandler(handler)
        handler.close()
    root.setLevel(level)


def flush():
    for h in logging.getLogger().handlers:
        h.flush()


def test_lines_go_to_the_log_file_under_the_data_dir(logs):
    path = setup_logging(make_settings(logs))
    assert path == logs / "data" / "logs" / LOG_FILENAME
    logging.getLogger("studybuddy.test").info("job %s finished", 7)
    flush()
    assert "INFO    studybuddy.test: job 7 finished" in path.read_text(encoding="utf-8")


def test_the_token_never_reaches_the_file_not_even_in_a_traceback(logs):
    path = setup_logging(make_settings(logs))
    log = logging.getLogger("studybuddy.test")
    log.warning("GET https://api.telegram.org/bot%s/getUpdates", TOKEN)
    try:
        raise RuntimeError(f"failed for bot{TOKEN}")
    except RuntimeError:
        log.exception("boom")
    flush()
    assert TOKEN not in path.read_text(encoding="utf-8") and "<telegram-token>" in path.read_text(encoding="utf-8")


def test_the_file_rotates_and_keeps_only_the_last_few(logs):
    path = setup_logging(make_settings(logs), max_bytes=2000, backups=2)
    for i in range(200):
        logging.getLogger("studybuddy.test").info("line %03d %s", i, "x" * 60)
    flush()
    names = sorted(p.name for p in path.parent.iterdir())
    assert names == [LOG_FILENAME, LOG_FILENAME + ".1", LOG_FILENAME + ".2"]


def test_the_level_comes_from_settings_and_is_checked(logs):
    path = setup_logging(make_settings(logs, SB_LOG_LEVEL="warning"))
    log = logging.getLogger("studybuddy.test")
    log.info("quiet")
    log.warning("loud")
    flush()
    text = path.read_text(encoding="utf-8")
    assert "loud" in text and "quiet" not in text
    with pytest.raises(ConfigError, match="SB_LOG_LEVEL"):
        make_settings(logs, SB_LOG_LEVEL="chatty")


def test_calling_it_twice_does_not_duplicate_lines(logs):
    settings = make_settings(logs)
    setup_logging(settings)
    path = setup_logging(settings)
    logging.getLogger("studybuddy.test").info("once")
    flush()
    assert path.read_text(encoding="utf-8").count("once") == 1
