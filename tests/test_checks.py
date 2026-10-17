import dataclasses

import httpx

from conftest import FakeLM
from studybuddy.checks import FAIL, PASS, WARN, exit_code, run_checks
from studybuddy.llm import LMStudioClient


def run(settings, fake, tmp_path):
    # Directories must be outside the repo: tmp_path is.
    return run_checks(settings, LMStudioClient(settings, transport=fake.transport()))


def by_name(results):
    return {r.name: r for r in results}


def test_all_pass(settings, tmp_path):
    results = by_name(run(settings, FakeLM(chat_reply="BANANA pong"), tmp_path))
    for name in ("directories", "server", "CHAT", "READER", "EMBED", "embedding guard"):
        assert results[name].status == PASS, results[name]
    assert exit_code(list(results.values())) == 0


def test_reader_misread_warns(settings, tmp_path):
    results = by_name(run(settings, FakeLM(chat_reply="apple"), tmp_path))
    assert results["READER"].status == WARN


def test_unlisted_model_warns(settings, tmp_path):
    results = run(settings, FakeLM(chat_reply="BANANA", models=("chat-m",)), tmp_path)
    assert any(r.status == WARN and "READER listed" == r.name for r in results)


def test_server_down_fails(settings, tmp_path):
    def boom(request):
        raise httpx.ConnectError("refused")
    client = LMStudioClient(settings, transport=httpx.MockTransport(boom))
    results = run_checks(settings, client)
    assert by_name(results)["server"].status == FAIL
    assert exit_code(results) == 1


def test_embedding_change_fails(settings, tmp_path):
    run(settings, FakeLM(), tmp_path)
    changed = dataclasses.replace(settings, embed_model="other")
    results = run(changed, FakeLM(dim=5), tmp_path)
    assert by_name(results)["embedding guard"].status == FAIL
    assert exit_code(results) == 1
