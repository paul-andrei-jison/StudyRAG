import pytest

from studybuddy.chat.overview import detect

OVERVIEW = [
    # summaries
    "give me a summary of the entire paper",
    "Give me a summary of the whole handout",
    "summarize this handout",
    "Summarize the worksheet please",
    "can you summarise my notes",
    "I need an overview of the lesson",
    "overview of this document",
    "recap the whole chapter",
    "what are the main points of this article",
    "key takeaways from the reviewer",
    "summarize everything",
    "summarize it",
    "give me the gist of the activity sheet",
    # contents
    "what is this worksheet about",
    "What is this worksheet about?",
    "what's the handout about",
    "what is the Earth article about?",
    "what does the quiz cover",
    "What does this paper contain?",
    "what topics does the quiz cover",
    "what topics are in this worksheet?",
    "what is in this handout",
    "list everything about the Earth article",
    "tell me everything about this document",
    # Filipino / Taglish
    "buod ng papel na ito",
    "ano ang buod ng worksheet",
    "ibuod mo ito",
    "i-summarize mo ito",
    "paki-summarize ang handout",
    "ano ang laman nito",
    "ano ang laman ng papel na ito",
    "ano ang nasa worksheet",
    "ano ang mga paksa ng quiz",
    "tungkol saan ito",
    "tungkol saan ang worksheet",
    "ano ang mga paksa sa handout na ito",
]

NOT_OVERVIEW = [
    # near misses: document words, but a fact is asked for
    "what is the test for acids",
    "what do my notes say about gravity",
    "what page is the quiz on",
    "how many items are in the worksheet",
    "What is the lesson topic of the activity sheet?",
    "when is the quiz",
    "which chapter covers photosynthesis",
    "who wrote this article",
    "what is the date on the handout",
    # specific facts and topics
    "How much of Earth's surface is covered by water?",
    "What creates Earth's magnetic field?",
    "Which gases make up most of the atmosphere?",
    "Why is Earth considered an interconnected system?",
    "What does a volcanic eruption release into the air?",
    "Summarize photosynthesis",
    "explain how plates move",
    # clearly not in the papers
    "Who won the FIFA World Cup in 1998?",
    "How do I solve the quadratic equation x^2 - 5x + 6 = 0?",
    "What caused the French Revolution?",
    "What is a good recipe for chocolate chip cookies?",
    "How does mitosis differ from meiosis?",
    "What is the plot of Romeo and Juliet?",
    "What is the capital of the Philippines?",
    "How are earthquakes measured on the Richter scale?",
    "Which planet in our solar system has the most moons?",
    "Who discovered that the Earth has a magnetic field?",
    "ano ang kabisera ng Pilipinas",
    "sino ang nanalo sa eleksyon",
    "",
]


@pytest.mark.parametrize("question", OVERVIEW)
def test_overview_questions_are_recognised(question):
    found = detect(question)
    assert found.is_overview and found.pattern, question


@pytest.mark.parametrize("question", NOT_OVERVIEW)
def test_near_misses_and_ordinary_questions_stay_on_the_normal_path(question):
    assert not detect(question).is_overview, question


def test_a_named_topic_is_reported_so_that_it_can_be_checked_against_the_papers():
    found = detect("summarize the French Revolution paper")
    assert found.is_overview and found.topic_terms == ("french", "revolution")   # the session refuses it: no paper matches
    assert detect("give me a summary of the entire paper").topic_terms == ()
    assert detect("list everything about the Earth article").topic_terms == ("earth",)
    assert detect("buod ng papel na ito").topic_terms == ()


def test_the_recency_hint_is_reported():
    assert detect("summarize this handout").recent_hint
    assert detect("summarize the latest paper I uploaded").recent_hint
    assert detect("buod ng papel na ito").recent_hint
    assert not detect("summarize the Earth article").recent_hint


def test_curly_apostrophes_and_case_do_not_matter():
    assert detect("WHAT’S THE HANDOUT ABOUT").is_overview
