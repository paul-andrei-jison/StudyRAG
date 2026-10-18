"""The bot inside the app: start-up order, shutdown order, the on/off switch and the console line."""
import dataclasses

from bot_helpers import StubService
from chat_helpers import ScriptedClient
from ingest_helpers import make_env
from studybuddy.__main__ import bot_line
from studybuddy.bot.telegram_app import TelegramBot
from studybuddy.config import BotPlan, bot_plan
from studybuddy.ingest.service import IngestService
from studybuddy.ui.runtime import Runtime, build_runtime
from telegram_helpers import FakeTelegram

TOKEN = "123456789:AAHfake_token-value0123456789abcdefg"


class Recorder:
    def __init__(self, events, name):
        self.events, self.name = events, name

    def start(self):
        self.events.append(f"{self.name}.start")

    def stop(self, *a, **k):
        self.events.append(f"{self.name}.stop")


def test_the_bot_starts_before_the_ingest_service_and_stops_before_it(tmp_path, settings):
    env = make_env(tmp_path, settings)
    service = IngestService(env.settings, env.index, env.reader, None, env.chat, env.subjects, poll_interval=0.02)
    events: list[str] = []
    real_start, real_stop = service.start, service.stop
    service.start = lambda: (events.append("service.start"), real_start())[1]
    service.stop = lambda *a, **k: (events.append("service.stop"), real_stop(*a, **k))[1]
    rt = Runtime(env.settings, ScriptedClient(), env.index, env.subjects, service, bot=Recorder(events, "bot"))
    rt.start()
    rt.close()
    assert events == ["bot.start", "service.start", "bot.stop", "service.stop"]


def test_without_a_bot_nothing_changes(tmp_path, settings):
    env = make_env(tmp_path, settings)
    service = IngestService(env.settings, env.index, env.reader, None, env.chat, env.subjects, poll_interval=0.02)
    rt = Runtime(env.settings, ScriptedClient(), env.index, env.subjects, service)
    assert rt.bot is None
    rt.start()
    rt.close()


def test_the_real_runtime_gets_a_bot_only_when_asked(tmp_path, settings):
    s = dataclasses.replace(settings, library_dir=tmp_path / "lib", data_dir=tmp_path / "data", subjects_file=tmp_path / "data" / "subjects.json",
                            telegram_token=TOKEN, telegram_user_id=42)
    (tmp_path / "lib").mkdir()
    (tmp_path / "data").mkdir()
    on = build_runtime(s, use_ocr=False, with_bot=True)
    try:
        assert isinstance(on.bot, TelegramBot) and on.bot.allowed_id == 42
    finally:
        on.close()
    off = build_runtime(s, use_ocr=False, with_bot=False)
    try:
        assert off.bot is None
    finally:
        off.close()


def test_the_console_line_says_whether_the_bot_runs_and_never_shows_the_token(tmp_path, settings):
    s = dataclasses.replace(settings, telegram_token=TOKEN, telegram_user_id=42)
    assert bot_line(bot_plan(s), s) == "  telegram: on (long polling; answers only user id 42)"
    assert bot_line(bot_plan(s, no_bot=True), s) == "  telegram: off (turned off with --no-bot)"
    assert bot_line(BotPlan(False, "no TELEGRAM_BOT_TOKEN in .env"), s) == "  telegram: off (no TELEGRAM_BOT_TOKEN in .env)"
    assert TOKEN not in bot_line(bot_plan(s), s)


def test_a_shutdown_while_the_bot_is_still_connecting_does_not_hang(tmp_path, settings):
    s = dataclasses.replace(settings, library_dir=tmp_path / "lib", data_dir=tmp_path / "data", telegram_token=TOKEN, telegram_user_id=42)
    (tmp_path / "lib").mkdir()
    (tmp_path / "data").mkdir()
    api = FakeTelegram()
    api.fail_all = True                                     # no connection at all
    bot = TelegramBot(s, StubService(), type("S", (), {"names": lambda self: []})(), request=api, connect_retry_seconds=30)
    bot.start()
    assert not bot.ready.wait(0.5) and "waiting for Telegram" in bot.status
    bot.stop(timeout=5)
    assert not bot._thread.is_alive()
