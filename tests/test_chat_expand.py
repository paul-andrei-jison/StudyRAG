import pytest

from chat_helpers import EARTH, LAX_GATE, WATER, ScriptedClient, add_doc, make_setup
from studybuddy.chat.expand import expand_query, fuse_searches, parse_phrasings
from studybuddy.chat.session import ChatConfig
from studybuddy.errors import LLMError, TransientLLMError

Q = "which layer allows plates to move"


@pytest.fixture
def setup(settings, tmp_path):
    s = make_setup(settings, tmp_path)
    yield s
    s.index.close()


class FakeChat:
    def __init__(self, reply):
        self.reply, self.calls = reply, []

    def chat(self, messages, temperature=0.0, max_tokens=None):
        self.calls.append(messages)
        if isinstance(self.reply, Exception):
            raise self.reply
        return self.reply


@pytest.mark.parametrize("raw,expected", [
    ('["what part of Earth lets tectonic plates drift", "which layer lets plates slide"]',
     ["what part of Earth lets tectonic plates drift", "which layer lets plates slide"]),
    ('Sure! ["a rephrasing", "another one"] hope that helps', ["a rephrasing", "another one"]),
    ('["only one"]', ["only one"]),
    ('["x", "y", "z"]', ["x", "y"]),                                   # at most two
    ('["WHICH LAYER ALLOWS PLATES TO MOVE", "a new one"]', ["a new one"]),  # a copy of the question adds nothing
    ('["same", "Same", " same "]', ["same"]),
    ('["", "  ", 5, null, "ok"]', ["ok"]),                              # only real strings count
    ("no list at all", []), ("[not json", []), ("{}", []), ("", []),
])
def test_parse_phrasings(raw, expected):
    assert parse_phrasings(raw, Q) == expected


def test_overlong_phrasings_are_dropped():
    assert parse_phrasings('["' + "x" * 400 + '", "fine"]', Q) == ["fine"]


def test_expand_query_asks_for_rephrasings_not_answers():
    chat = FakeChat('["a", "b"]')
    assert expand_query(chat, Q) == ["a", "b"]
    system = chat.calls[0][0]["content"]
    assert "Do not answer it" in system and "JSON list" in system and Q in chat.calls[0][1]["content"]


def test_expand_failures_give_no_expansion_but_transient_ones_propagate():
    assert expand_query(FakeChat(LLMError("returned 400")), Q) == []
    with pytest.raises(TransientLLMError):
        expand_query(FakeChat(TransientLLMError("LM Studio is off")), Q)


def test_fusion_finds_a_chunk_that_only_a_rephrasing_reaches(setup):
    add_doc(setup.index, "mantle", "Plates glide over the asthenosphere, the soft upper part of the mantle.", title="Layers")
    plain = [h.title for h in setup.index.search("which layer allows plates to move", k=20)]
    fused = fuse_searches(setup.index, ["something unrelated entirely", "asthenosphere soft upper mantle plates glide"], 20)
    assert "Layers" in [h.title for h in fused]
    assert isinstance(plain, list)


def test_fused_hits_from_a_rephrasing_carry_no_raw_scores_but_original_hits_keep_theirs(setup):
    add_doc(setup.index, "mantle", "Plates glide over the asthenosphere, the soft upper part of the mantle.", title="Layers")
    fused = fuse_searches(setup.index, ["magnetic field of Earth molten iron", "asthenosphere soft upper mantle plates glide"], 2)
    by_title = {h.title: h for h in fused}
    assert by_title["Earth notes"].vector_score is not None and by_title["Earth notes"].keyword_score is not None
    assert by_title["Layers"].vector_score is None and by_title["Layers"].keyword_score is None  # found only by the rephrasing


def test_fusion_ranks_hits_found_by_several_queries_higher(setup):
    fused = fuse_searches(setup.index, ["magnetic field of Earth", "Earth magnetic field iron core"], 20)
    assert fused[0].title == "Earth notes"
    scores = [h.score for h in fused]
    assert scores == sorted(scores, reverse=True)
    assert fused[0].score > 1 / (60 + 1)  # more than a single list could give


def test_fusion_respects_k_and_the_subject_filter(setup):
    assert len(fuse_searches(setup.index, ["Earth", "plates"], 2)) == 2
    only = fuse_searches(setup.index, ["Earth tectonic plates", "plates move"], 20, subject="Geography")
    assert only and {h.subject for h in only} == {"Geography"}


def test_a_single_query_fuses_to_its_own_ranking(setup):
    one = fuse_searches(setup.index, ["magnetic field of Earth"], 20)
    plain = setup.index.search("magnetic field of Earth", k=20)
    assert [h.chunk_id for h in one] == [h.chunk_id for h in plain]


def test_the_session_uses_expansion_only_when_switched_on(setup):
    reply = ["The plates move over the mantle [1]."]
    off = ScriptedClient([reply], rewrites=['["a", "b"]'])
    setup.session(off).ask("tectonic plates move over the mantle")
    assert off.chat_calls == []  # off by default: no extra model call
    on = ScriptedClient([reply], rewrites=['["plates drift over the mantle", "mantle plates slide"]'])
    session = setup.session(on, config=ChatConfig(expand_queries=True))
    final = session.ask("tectonic plates move over the mantle")
    assert len(on.chat_calls) == 1 and final.status == "answered"


def test_an_unusable_expansion_falls_back_to_the_plain_search(setup):
    client = ScriptedClient([["Plates move [1]."]], rewrites=["I can't do that."])
    final = setup.session(client, config=ChatConfig(expand_queries=True)).ask("tectonic plates move over the mantle")
    assert final.status == "answered" and final.sources


def test_config_from_settings_follows_the_setting_and_explicit_overrides_win(settings):
    import dataclasses
    on = dataclasses.replace(settings, chat_expand_queries=True)
    off = dataclasses.replace(settings, chat_expand_queries=False)
    assert ChatConfig.from_settings(on).expand_queries is True and ChatConfig.from_settings(off).expand_queries is False
    assert ChatConfig.from_settings(on, expand_queries=False).expand_queries is False
    assert ChatConfig().expand_queries is False  # a bare ChatConfig (as tests build it) never makes the extra call
