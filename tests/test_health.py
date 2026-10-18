import dataclasses
from types import SimpleNamespace

import httpx

from conftest import FakeLM
from studybuddy.__main__ import main
from studybuddy.bot.state import DOWNLOADING, Item, StateStore
from studybuddy.checks import FAIL, PASS, WARN, exit_code
from studybuddy.health import format_results, run_health
from studybuddy.llm import LMStudioClient

TOKEN = "123456789:AAHfake_token-value0123456789abcdefg"


def plenty(path):
    return SimpleNamespace(free=500 * 1024 ** 3)


def free_port(host, port):
    return False


def health(settings, fake=None, deep=False, usage=plenty, port_busy=free_port):
    client = LMStudioClient(settings, transport=(fake or FakeLM(chat_reply="BANANA pong")).transport())
    return {r.name: r for r in run_health(settings, client, deep=deep, usage=usage, port_busy=port_busy)}


def test_a_healthy_setup_passes_everything(settings):
    results = health(settings)
    assert all(r.status == PASS for r in results.values()), [r for r in results.values() if r.status != PASS]
    assert "3 model(s) loaded" in results["LM Studio"].detail and results["index"].detail.startswith("empty")


def test_lm_studio_off_is_a_failure_with_the_fix(settings):
    def refuse(request):
        raise httpx.ConnectError("refused")
    client = LMStudioClient(settings, transport=httpx.MockTransport(refuse))
    results = run_health(settings, client, usage=plenty, port_busy=free_port)
    lm = next(r for r in results if r.name == "LM Studio")
    assert lm.status == FAIL and "start its local server" in lm.detail and exit_code(results) == 1


def test_a_model_that_is_not_loaded_is_named(settings):
    results = health(settings, FakeLM(models=("chat-m", "embed-m")))
    assert results["model READER"].status == FAIL and "reader-m" in results["model READER"].detail
    assert results["model CHAT"].status == PASS


def test_low_disk_space_warns_then_fails(settings):
    assert health(settings, usage=lambda p: SimpleNamespace(free=3 * 1024 ** 3))["disk space (library)"].status == WARN
    assert health(settings, usage=lambda p: SimpleNamespace(free=100 * 1024 ** 2))["disk space (data)"].status == FAIL


def test_the_token_is_never_printed(settings):
    configured = dataclasses.replace(settings, telegram_token=TOKEN, telegram_user_id=42)
    text = format_results(list(health(configured).values()))
    assert TOKEN not in text and "token set (not shown)" in text and "user id 42" in text


def test_a_token_without_a_user_id_is_reported(settings):
    result = health(dataclasses.replace(settings, telegram_token=TOKEN, telegram_user_id=None))["telegram"]
    assert result.status == FAIL and TOKEN not in result.detail


def test_a_busy_port_is_a_warning(settings):
    assert health(settings, port_busy=lambda h, p: True)["web port"].status == WARN


def test_deep_mode_asks_the_models(settings):
    assert "CHAT" not in health(settings)
    results = health(settings, deep=True)
    assert results["CHAT"].status == PASS and results["READER"].status == PASS and results["EMBED"].status == PASS


def test_files_waiting_in_the_bot_queue_are_counted(settings):
    StateStore(settings.data_dir / "bot").save_item(Item(id="a1", chat_id=1, message_id=2, kind="photo", file_id="f", stage=DOWNLOADING))
    assert health(settings)["bot queue"].status == WARN and "1 file" in health(settings)["bot queue"].detail


def test_the_command_prints_and_exits_with_the_result(monkeypatch, capsys, settings):
    monkeypatch.setattr("studybuddy.config.load_settings", lambda *a, **k: settings)
    monkeypatch.setattr("studybuddy.llm.LMStudioClient.list_models", lambda self: ["reader-m", "chat-m", "embed-m"])
    monkeypatch.setattr("studybuddy.health.port_in_use", lambda h, p: False)
    assert main(["--check"]) == 0
    assert "[PASS] LM Studio" in capsys.readouterr().out
