"""Overview questions: "summarize the entire paper", "what is this worksheet about", "buod ng papel na ito".

Such a question resembles no single page, so the answerability gate (which compares the question with chunks) would refuse it.
`detect` recognises them with plain rules (English, Filipino and Taglish); the session then answers from whole documents
instead. The rules are deliberately narrow: a document word alone ("what page is the quiz on") never makes an overview.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

from ..index.search import STOPWORDS

_SUMMARY = (r"summar\w*|summary|overview|recap|gist|outline|synops\w*|takeaways?|key (?:points?|ideas?|concepts?)|"
            r"main (?:points?|ideas?|topics?|concepts?)|buod|ibuod|i-buod|lagom|pangkalahatan")
_DOC = (r"papers?|handouts?|worksheets?|activity sheets?|sheets?|quiz(?:zes)?|tests?|exams?|reviewers?|lessons?|lectures?|notes?|"
        r"articles?|documents?|files?|pages?|pdfs?|chapters?|modules?|activit(?:y|ies)|readings?|assignments?|materials?|"
        r"photos?|pictures?|images?|papel|aralin|dokumento|pagsusulit|sulatin|babasahin")
_WHOLE = r"entire|whole|everything|complete|full|all of it|buong|buo|lahat|ito|nito|iyan|niyan|iyon|this|these|that|those"
_FILLER = (r"give|gimme|tell|show|write|make|list|explain|describe|down|please|pls|paki|po|can|could|you|would|want|need|me|my|mo|ko|akin|ang|ng|sa|na|ay|mga|"
           r"and|also|of|for|to|in|on|a|an|the|is|are|it|its|do|does|ano|ang|ba|yung|'yung|saan|sino|"
           r"summar\w*|ize|ise|i")

SUMMARY_WORD = re.compile(rf"\b(?:{_SUMMARY})\b", re.IGNORECASE)
DOC_WORD = re.compile(rf"\b(?:{_DOC})\b", re.IGNORECASE)
WHOLE_WORD = re.compile(rf"\b(?:{_WHOLE})\b", re.IGNORECASE)
RECENT_WORD = re.compile(r"\b(?:this|these|latest|newest|just|recent|recently|last|new|ito|nito|kaka\w+|pinakabago|bago)\b", re.IGNORECASE)

_ARTICLE = r"(?:(?:this|the|my|that|our|these|those|iyong|yung)\s+)?"
_CONTENTS = [
    # what is this worksheet about / what's the Earth article about
    re.compile(rf"\bwhat(?:'s| is| are)\s+{_ARTICLE}(?:[\w']+\s+){{0,3}}?(?:{_DOC})\s+(?:all\s+)?about\s*\??$", re.IGNORECASE),
    # what does the quiz cover / contain / include / discuss
    re.compile(rf"\bwhat\s+(?:does|do|will|did)\s+{_ARTICLE}(?:[\w']+\s+){{0,3}}?(?:{_DOC})\s+(?:cover|contain|include|discuss|deal with|talk about|say|have)\s*\??$",
               re.IGNORECASE),
    # what topics does the quiz cover / what topics are in this worksheet
    re.compile(rf"\bwhat\s+(?:topics?|subjects?|things|lessons?|concepts?|ideas?|chapters?|parts?)\s+(?:does|do|did|is|are|will)\b.*\b(?:{_DOC}|{_WHOLE})\b", re.IGNORECASE),
    # what's in this worksheet
    re.compile(rf"\bwhat(?:'s| is| are)\s+(?:in|inside|on)\s+{_ARTICLE}(?:[\w']+\s+){{0,3}}?(?:{_DOC})\s*\??$", re.IGNORECASE),
    # list / tell me everything about the Earth article
    re.compile(rf"\b(?:list|tell me|show me|give me|explain|write)\s+(?:down\s+)?(?:everything|all of it|all)\s+(?:about|in|from|on)\b.*\b(?:{_DOC})\b", re.IGNORECASE),
    # Filipino / Taglish
    re.compile(rf"\btungkol\s+(?:sa\s+)?(?:saan|ano)\b.*\b(?:{_DOC}|ito|nito|iyan|iyon)\b", re.IGNORECASE),
    re.compile(rf"\bano\s+(?:ang\s+|ng\s+)?(?:mga\s+)?(?:laman|nilalaman|nasa|nakasulat|nandito|nandiyan|paksa|topic|aralin)\b.*\b(?:{_DOC}|ito|nito|iyan|iyon)\b",
               re.IGNORECASE),
]
_WORDS = re.compile(r"[\w']+", re.UNICODE)
_FILLER_WORD = re.compile(rf"^(?:{_FILLER})$", re.IGNORECASE)


@dataclass(frozen=True)
class Detection:
    is_overview: bool
    topic_terms: tuple[str, ...]    # what is left of the question once overview, document and filler words are removed
    recent_hint: bool               # "this", "latest", "just uploaded", "ito"... : the newest document may be meant
    pattern: str = ""               # which rule matched (for the evaluation report)


def _normalise(question: str) -> str:
    text = question.replace("’", "'").replace("‘", "'").lower()
    return " ".join(text.replace("-", "-").split())


def topic_terms(text: str) -> tuple[str, ...]:
    """Words that name a topic: not a summary, document or whole-ness word, not a stopword or request filler."""
    out = []
    for word in _WORDS.findall(text):
        w = word.strip("'")
        if (not w or w in STOPWORDS or _FILLER_WORD.match(w) or SUMMARY_WORD.fullmatch(w) or DOC_WORD.fullmatch(w)
                or WHOLE_WORD.fullmatch(w) or RECENT_WORD.fullmatch(w)):
            continue
        out.append(w)
    return tuple(dict.fromkeys(out))


def detect(question: str) -> Detection:
    """Is this a request about whole documents rather than a fact in one place?"""
    text = _normalise(question)
    topic = topic_terms(text)
    recent = bool(RECENT_WORD.search(text))
    for number, pattern in enumerate(_CONTENTS, 1):
        if pattern.search(text):
            return Detection(True, topic, recent, f"contents-{number}")
    if SUMMARY_WORD.search(text) and (DOC_WORD.search(text) or WHOLE_WORD.search(text) or not topic):
        return Detection(True, topic, recent, "summary")
    return Detection(False, topic, recent)
