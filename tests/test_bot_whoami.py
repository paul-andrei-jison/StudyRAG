import httpx
import pytest

from studybuddy.bot.whoami import WhoamiError, fetch_updates, senders

TOKEN = "123456789:AAHfake_token-value0123456789abcdefg"


def client(status=200, body=None, seen=None):
    def handler(request):
        if seen is not None:
            seen.append(request)
        return httpx.Response(status, json=body if body is not None else {"ok": True, "result": []})
    return httpx.Client(transport=httpx.MockTransport(handler))


def message(user_id, username="paul", chat="private"):
    return {"update_id": 1, "message": {"message_id": 1, "chat": {"id": user_id, "type": chat},
                                        "from": {"id": user_id, "is_bot": False, "first_name": "Paul", "username": username}, "text": "hi"}}


def test_senders_are_listed_with_counts_most_active_first():
    found = senders([message(5), message(7, "other"), message(7, "other")])
    assert [(s.user_id, s.count, s.username) for s in found] == [(7, 2, "other"), (5, 1, "paul")]
    assert senders([]) == [] and senders([{"update_id": 3}]) == []


def test_a_button_tap_counts_as_a_sender_too():
    tap = {"update_id": 2, "callback_query": {"id": "1", "from": {"id": 9, "first_name": "X"}, "message": {"chat": {"id": 9, "type": "private"}}}}
    assert senders([tap])[0].user_id == 9


def test_fetching_only_reads(tmp_path):
    seen = []
    result = fetch_updates(TOKEN, client(body={"ok": True, "result": [message(5)]}, seen=seen))
    assert result[0]["message"]["from"]["id"] == 5
    assert [r.url.path.rsplit("/", 1)[-1] for r in seen] == ["getUpdates"] and "offset" not in str(seen[0].url)   # nothing is confirmed


@pytest.mark.parametrize("status,words", [(401, "does not accept this bot token"), (404, "does not accept this bot token"),
                                          (409, "Another program is already reading"), (500, "HTTP 500")])
def test_failures_are_explained_without_the_token(status, words):
    with pytest.raises(WhoamiError, match=words) as caught:
        fetch_updates(TOKEN, client(status))
    assert TOKEN not in str(caught.value)


def test_no_connection_is_explained_without_the_token():
    def refuse(request):
        raise httpx.ConnectError(f"cannot connect to {request.url}")
    with pytest.raises(WhoamiError, match="Could not reach Telegram") as caught:
        fetch_updates(TOKEN, httpx.Client(transport=httpx.MockTransport(refuse)))
    assert TOKEN not in str(caught.value) and caught.value.__cause__ is None
