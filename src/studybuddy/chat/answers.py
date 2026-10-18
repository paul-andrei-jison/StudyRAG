"""Data returned by a chat turn: the answer text, the numbered sources and what was checked about them."""
from __future__ import annotations

from dataclasses import dataclass, field

from ..index.search import Hit
from ..reader.quality import ATTENTION_FLAGS, ATTENTION_QUALITY
from ..ingest.results import FLAG_REASONS

ANSWERED, NOT_IN_PAPERS_STATUS, ERROR = "answered", "not_in_papers", "error"
NEEDS_CHOICE = "needs_choice"      # an overview question that could mean several papers: the answer text asks which
SEARCH, OVERVIEW = "search", "overview"


@dataclass
class Source:
    number: int                    # the [n] the answer uses
    subject: str
    title: str
    file_path: str                 # the document in the library
    page_number: int
    image_path: str                # the original image of the cited page (a PDF path for PDF pages)
    text: str
    page_quality: float | None
    page_flags: list[str]
    vector_score: float | None = None    # raw cosine similarity
    keyword_score: float | None = None   # raw bm25
    cited: bool = False

    @property
    def hard_to_read(self) -> bool:
        return (self.page_quality is not None and self.page_quality < ATTENTION_QUALITY) or bool(ATTENTION_FLAGS & set(self.page_flags))

    @property
    def problems(self) -> list[str]:
        """Plain-language reasons this page may be misread."""
        reasons = [FLAG_REASONS.get(f, f) for f in self.page_flags if f in ATTENTION_FLAGS]
        if self.page_quality is not None and self.page_quality < ATTENTION_QUALITY and not reasons:
            reasons.append("the reading quality of this page was low")
        return reasons

    @classmethod
    def from_hit(cls, number: int, hit: Hit) -> "Source":
        return cls(number, hit.subject, hit.title, hit.file_path, hit.page_number, hit.image_path, hit.text,
                   hit.page_quality, list(hit.page_flags), hit.vector_score, hit.keyword_score)


@dataclass(frozen=True)
class Progress:
    """Yielded by `ask_stream` while slow preparation (condensing pages) is under way; it is not part of the answer."""
    text: str


@dataclass
class ChatAnswer:
    text: str                                  # the answer shown to the student, citations validated, notes appended
    status: str                                # answered | not_in_papers | error | needs_choice
    sources: list[Source] = field(default_factory=list)   # numbered; for a refusal: the closest pages (not cited)
    cited: list[int] = field(default_factory=list)
    invalid_citations: list[int] = field(default_factory=list)   # numbers that named no source (stripped)
    uncited: bool = False                      # an answer that cites nothing: treat with care
    likely_source: int | None = None           # for an uncited answer: the source whose wording matches (a hint, not a citation)
    notes: list[str] = field(default_factory=list)       # deterministic notes added after the model's text
    warnings: list[str] = field(default_factory=list)    # for the program or the CLI (not part of the answer text)
    query: str = ""                            # what was searched for (the rewritten question)
    rewritten: bool = False
    refusal_reason: str | None = None
    subject_filter: str | None = None
    truncated: bool = False                    # the model hit its length limit, or the connection dropped
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    context_tokens: int | None = None
    first_token_seconds: float | None = None
    total_seconds: float = 0.0
    mode: str = SEARCH                         # search | overview (answered from whole documents, no page-similarity gate)
    overview_notes: list[str] = field(default_factory=list)   # which papers, and how long ones were reduced
    model_calls_condensing: int = 0

    @property
    def cited_sources(self) -> list[Source]:
        return [s for s in self.sources if s.number in self.cited]
