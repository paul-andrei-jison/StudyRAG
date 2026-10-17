import io
import subprocess
import sys
from pathlib import Path

import pytest

from chat_helpers import LAX_GATE, ScriptedClient, add_doc, make_setup
from studybuddy.chat.cli import chat_main, format_sources
from studybuddy.chat.gate import GateConfig
from studybuddy.chat.session import ChatSession
from studybuddy.errors import StudyBuddyError, TransientLLMError
from studybuddy.ingest.subjects import SubjectList

ROOT = Path(__file__).resolve().parents[1]
GOOD = ["The field comes from molten iron ", "[1]", "."]
Q = "What generates the magnetic field of Earth?"


@pytest.fixture
def setup(settings, tmp_path):
    s = make_setup(settings, tmp_path)
    yield s
    s.index.close()


def run(setup, argv, lines=(), client=None, gate=LAX_GATE, with_subjects=True, tmp_path=None):
    client = client or ScriptedClient([GOOD] * 6)
    subjects = SubjectList(setup.settings.data_dir / "subjects.json", ["Science", "Geography"]) if with_subjects else None
    feed = iter(lines)

    def factory(settings, args):
        session = ChatSession(setup.index, client, settings, subjects, gate=gate)
        return session, lambda: None

    def input_fn(prompt):
        try:
            return next(feed)
        except StopIteration:
            raise EOFError

    out = io.StringIO()
    code = chat_main(argv, setup.settings, factory, input_fn, out)
    return code, out.getvalue(), client


def test_one_shot_prints_the_streamed_answer_and_the_sources_with_file_paths(setup):
    code, text, _ = run(setup, ["--ask", Q])
    assert code == 0
    assert "The field comes from molten iron [1]." in text
    assert "Sources:" in text and "[1] *cited* Science | Earth notes | C:/lib/Science/earth.jpg, page 1" in text
    assert "image: C:/lib/Science/earth.jpg" in text


def test_no_stream_prints_the_complete_answer_once(setup):
    code, text, _ = run(setup, ["--ask", Q, "--no-stream"])
    assert code == 0 and text.count("The field comes from molten iron [1].") == 1


def test_interactive_loop_answers_until_quit(setup):
    code, text, client = run(setup, [], lines=[Q, "", "/quit", "never reached"])
    assert code == 0 and "Ask a question about your papers" in text and "Study Buddy> " in text
    assert len(client.stream_calls) == 1


def test_end_of_input_ends_the_chat_cleanly(setup):
    code, text, _ = run(setup, [], lines=[Q])
    assert code == 0 and "molten iron" in text


def test_a_refusal_shows_the_closest_pages_labelled_probably_not_relevant(setup):
    code, text, client = run(setup, ["--ask", "Who won the football world cup?"], gate=GateConfig(0.99, 0.99, 1.0))
    assert code == 0 and "don't seem to cover" in text and "Closest pages (probably not relevant):" in text
    assert "[why]" in text and client.stream_calls == []


def test_subject_command_lists_sets_and_clears_the_filter(setup):
    code, text, _ = run(setup, [], lines=["/subject", "/subject geography", "/subject", "/subject none", "/subject Cooking", "/quit"])
    assert "Subject filter: none (all subjects)" in text and "Known subjects: Science, Geography" in text
    assert "Only answering from Geography." in text and "Subject filter: Geography" in text
    assert "Answering from all subjects." in text and "[FAIL] Unknown subject 'Cooking'" in text


def test_subject_filter_from_the_command_line_applies(setup):
    code, text, _ = run(setup, ["--subject", "geography", "--ask", "Earth tectonic plates magnetic field"])
    assert code == 0 and "Science |" not in text and "Geography |" in text
    assert run(setup, ["--subject", "Cooking", "--ask", Q])[0] == 1


def test_sources_command_reprints_the_last_sources_and_works_before_any_answer(setup):
    _, text, _ = run(setup, [], lines=["/sources", Q, "/sources", "/quit"])
    assert "No answer yet." in text and text.count("Sources:") == 2  # once after the answer, once from /sources


def test_new_command_forgets_the_conversation(setup):
    client = ScriptedClient([GOOD, GOOD])
    _, text, client = run(setup, [], lines=[Q, "/new", Q, "/quit"], client=client)
    assert "Started a new conversation." in text
    assert client.chat_calls == []  # after /new the second question has no history, so no rewrite call


def test_follow_ups_are_rewritten_and_the_search_query_is_shown(setup):
    client = ScriptedClient([GOOD, ["Water covers 71% [1]."]], rewrites=["How much of the surface of Earth is covered by water?"])
    _, text, _ = run(setup, [], lines=[Q, "and how much is water?", "/quit"], client=client)
    assert "[searched for] How much of the surface of Earth is covered by water?" in text


def test_help_and_unknown_commands(setup):
    _, text, _ = run(setup, [], lines=["/help", "/nonsense", "/quit"])
    assert "/subject" in text and "/sources" in text and "/new" in text and "Unknown command '/nonsense'" in text


def test_show_scores_adds_the_raw_numbers(setup):
    _, text, _ = run(setup, ["--ask", Q, "--show-scores"])
    assert "cosine 0." in text and "bm25" in text
    assert "cosine" not in run(setup, ["--ask", Q])[1]


def test_hard_to_read_pages_are_marked_in_the_source_list(setup):
    add_doc(setup.index, "blurry", "Tectonic plates move over the mantle slowly.", title="Blurry", quality=0.2, flags=["empty"])
    _, text, _ = run(setup, ["--ask", "tectonic plates mantle"], client=ScriptedClient([["Plates move [1]."]]))
    assert "this page was hard to read: check the photo" in text


def test_a_failing_question_reports_and_the_chat_continues(setup):
    client = ScriptedClient([TransientLLMError("LM Studio is off"), GOOD])
    code, text, _ = run(setup, [], lines=[Q, Q, "/quit"], client=client)
    assert code == 0 and "[FAIL] LM Studio is not available right now" in text and "molten iron" in text


def test_a_failing_one_shot_question_exits_with_1(setup):
    code, text, _ = run(setup, ["--ask", Q], client=ScriptedClient([TransientLLMError("LM Studio is off")]))
    assert code == 1 and "[FAIL]" in text


def test_a_context_that_is_too_small_is_a_clear_failure(setup):
    code, text, _ = run(setup, ["--ask", Q], client=ScriptedClient(context=900))
    assert code == 1 and "[FAIL]" in text and "900 tokens" in text and "LM Studio" in text


def test_startup_failure_is_a_clean_message(setup):
    def broken(settings, args):
        raise StudyBuddyError("EMBEDDING MODEL CHANGED. Run: python scripts/index_admin.py reindex")

    out = io.StringIO()
    assert chat_main(["--ask", Q], setup.settings, broken, input, out) == 1 and "reindex" in out.getvalue()


def test_format_sources_without_sources():
    from studybuddy.chat.answers import ChatAnswer
    assert format_sources(ChatAnswer(text="x", status="answered")) == "Sources: none found."


def test_script_help_runs():
    out = subprocess.run([sys.executable, str(ROOT / "scripts" / "chat.py"), "--help"], capture_output=True, text=True, timeout=60)
    assert out.returncode == 0 and "--subject" in out.stdout and "--ask" in out.stdout
