import logging

import httpx
import pytest

from studybuddy import logsafe
from studybuddy.config import bot_plan, load_settings
from studybuddy.errors import ConfigError
from studybuddy.logsafe import REDACTED, install_log_safety, redact

TOKEN = "123456789:AAHfake_token-value0123456789abcdefg"


@pytest.fixture(autouse=True)
def installed():
    install_log_safety([TOKEN])
    yield
    logsafe._secrets.discard(TOKEN)


def test_anything_shaped_like_a_bot_token_is_hidden():
    assert redact(f"POST https://api.telegram.org/bot{TOKEN}/getUpdates") == f"POST https://api.telegram.org/bot{REDACTED}/getUpdates"
    assert redact("version 3.12:2026 time 10:30") == "version 3.12:2026 time 10:30"       # ordinary colons are left alone


def test_a_known_secret_is_hidden_even_if_it_does_not_look_like_a_token():
    logsafe._secrets.add("weird-secret-value")
    try:
        assert redact("the key is weird-secret-value ok") == f"the key is {REDACTED} ok"
    finally:
        logsafe._secrets.discard("weird-secret-value")


def test_log_lines_and_tracebacks_never_contain_the_token(caplog):
    log = logging.getLogger("studybuddy.test")
    with caplog.at_level(logging.INFO):
        log.info("calling https://api.telegram.org/bot%s/getMe", TOKEN)
        try:
            raise RuntimeError(f"failed: https://api.telegram.org/bot{TOKEN}/getMe")
        except RuntimeError:
            log.exception("it broke")
    text = caplog.text + "".join(r.getMessage() + (r.exc_text or "") for r in caplog.records)
    assert TOKEN not in text and REDACTED in text


def test_httpx_request_lines_are_not_logged_at_info(caplog):
    transport = httpx.MockTransport(lambda request: httpx.Response(200, json={"ok": True}))
    with caplog.at_level(logging.INFO), httpx.Client(transport=transport) as client:
        client.post(f"https://api.telegram.org/bot{TOKEN}/getMe")
    assert TOKEN not in caplog.text
    assert logging.getLogger("httpx").level == logging.WARNING


def settings_for(tmp_path, **over):
    env = {"SB_READER_MODEL": "r", "SB_CHAT_MODEL": "c", "SB_EMBED_MODEL": "e", "SB_LIBRARY_DIR": str(tmp_path / "lib"),
           "SB_DATA_DIR": str(tmp_path / "data"), **over}
    return load_settings(env, repo_root=tmp_path)


def test_the_token_is_not_in_the_settings_repr_or_str(tmp_path):
    s = settings_for(tmp_path, TELEGRAM_BOT_TOKEN=TOKEN, SB_TELEGRAM_USER_ID="42")
    assert s.telegram_token == TOKEN and TOKEN not in repr(s) and TOKEN not in str(s)


def test_the_bot_needs_a_token_and_the_students_user_id(tmp_path):
    assert not bot_plan(settings_for(tmp_path)).enabled and "TELEGRAM_BOT_TOKEN" in bot_plan(settings_for(tmp_path)).reason
    on = bot_plan(settings_for(tmp_path, TELEGRAM_BOT_TOKEN=TOKEN, SB_TELEGRAM_USER_ID="42"))
    assert on.enabled and on.reason == "on"
    with pytest.raises(ConfigError, match="would answer anybody") as caught:
        bot_plan(settings_for(tmp_path, TELEGRAM_BOT_TOKEN=TOKEN))
    assert TOKEN not in str(caught.value)


def test_the_bot_can_be_switched_off_by_flag_or_setting(tmp_path):
    full = dict(TELEGRAM_BOT_TOKEN=TOKEN, SB_TELEGRAM_USER_ID="42")
    assert "--no-bot" in bot_plan(settings_for(tmp_path, **full), no_bot=True).reason
    assert not bot_plan(settings_for(tmp_path, SB_TELEGRAM_ENABLED="0", **full)).enabled
    assert not bot_plan(settings_for(tmp_path, TELEGRAM_BOT_TOKEN=TOKEN), no_bot=True).enabled      # off: no user id needed
    with pytest.raises(ConfigError, match="SB_TELEGRAM_ENABLED"):
        settings_for(tmp_path, SB_TELEGRAM_ENABLED="maybe")


def test_the_user_id_has_a_new_name_and_the_old_one_still_works(tmp_path):
    assert settings_for(tmp_path, SB_TELEGRAM_USER_ID="7").telegram_user_id == 7
    assert settings_for(tmp_path, TELEGRAM_USER_ID="8").telegram_user_id == 8
    assert settings_for(tmp_path, SB_TELEGRAM_USER_ID="7", TELEGRAM_USER_ID="8").telegram_user_id == 7
    with pytest.raises(ConfigError, match="SB_TELEGRAM_USER_ID must be a number"):
        settings_for(tmp_path, SB_TELEGRAM_USER_ID="abc")
