"""What the ingest pipeline reports back for each document (the bot turns this into its message)."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

from ..reader.quality import PageRecord

INDEXED, DUPLICATE, FAILED, DEFERRED = "indexed", "duplicate", "failed", "deferred"

FLAG_REASONS = {
    "empty": "no text could be read from this page",
    "error": "this page could not be read",
    "too_short": "very little text was found for this image size",
    "many_illegible": "many words are unreadable",
    "low_agreement": "the two readers disagree a lot, so the text may be wrong",
    "possible_omission": "some text may have been missed",
    "glare": "there is glare on the photo",
}
IGNORED_FLAGS = {"ocr_failed"}  # the cross-check could not run: not a problem with the page itself


@dataclass
class FlaggedPage:
    page_number: int
    quality: float | None
    flags: list[str]
    reasons: list[str]             # plain language, one per flag
    needs_attention: bool          # worth asking the student to resend (for example as a file)


@dataclass
class IngestResult:
    status: str                                    # indexed | duplicate | failed | deferred
    job_id: str
    files: list[str] = field(default_factory=list)  # the sources, as submitted
    subject: str | None = None
    subject_source: str | None = None              # user | auto | fallback
    title: str = ""
    summary: str = ""
    page_count: int = 0
    flagged_pages: list[FlaggedPage] = field(default_factory=list)
    final_path: str | None = None                  # filed file, or the album folder
    pdf_path: str | None = None                    # combined PDF of an album
    doc_id: int | None = None
    source_kinds: list[str] = field(default_factory=list)   # photo | photo-file | pdf per file
    existing: dict | None = None                   # for duplicates: the document already indexed
    error: str | None = None                       # why it failed or was deferred
    transient: bool = False
    retry_in_seconds: float | None = None          # for deferred jobs
    failed_dir: str | None = None                  # where a permanently failed job's files went
    duplicate_dir: str | None = None
    warnings: list[str] = field(default_factory=list)
    attempts: int = 0
    duration_seconds: float = 0.0

    @property
    def ok(self) -> bool:
        return self.status == INDEXED


def flagged_pages(records: Sequence[PageRecord]) -> list[FlaggedPage]:
    out: list[FlaggedPage] = []
    for r in records:
        flags = [f for f in r.flags if f not in IGNORED_FLAGS]
        if not flags and not r.needs_attention:
            continue
        reasons = [(r.error or FLAG_REASONS["error"]) if f == "error" else FLAG_REASONS.get(f, f) for f in flags]
        out.append(FlaggedPage(r.page_number, r.quality, flags, reasons, r.needs_attention))
    return out
