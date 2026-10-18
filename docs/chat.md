# Chat

Code: `src/studybuddy/chat/`. Commands: `python scripts/chat.py [--subject Science] [--ask "question"]`,
`python scripts/calibrate_gate.py`, `python scripts/eval_expansion.py`, `python scripts/eval_overview.py [--timing]`.

## What happens to a question

1. **Follow-up rewrite** (only when there is history): one short model call turns "how does that make them worse?" into a standalone
   search query using the last three turns. An empty, refusing or answer-length rewrite falls back to the question as typed.
2. **Expansion** (on by default, `SB_CHAT_EXPAND_QUERIES=0` or `--no-expand` turns it off): two model-written rephrasings are searched too and
   fused by rank fusion. Raw scores always refer to the original question; a chunk found only through a rephrasing has none, so it cannot inflate the gate.
3. **Retrieve and spread**: hybrid search (pool of 20, optional subject filter), then 5 sources in score order with at most 2 chunks per page and
   3 per document; if that leaves places empty the best skipped chunks fill them.
4. **Gate**: from the raw scores, decide whether the papers answer at all. If not, reply "Your papers don't seem to cover this question, so I won't guess."
   without calling the model, and list the closest pages as *probably not relevant*.
5. **Budget and stream**: fit the prompt in the model's loaded context, stream the answer, validate citations while streaming.
6. **Notes added by the program, not the model**: a cited page that was hard to read ("check the photo" and why), a reminder that handwritten
   names, dates, scores and answer letters may be read wrongly (only when the question asks about such a field), and for an answer that cites nothing either a hint
   naming the source whose wording matches or a caution.

## Citations

Sources are headed `SOURCE n - Subject | Title | file, page p` and the answer cites `[n]`. `[1]`, `[1, 2]`, `[1-3]` and `[1][2]` are understood. A number that names no
source (including 0) is **stripped from the text and recorded** (`invalid_citations`); other brackets (`[sic]`) are left alone. A property test proves the streaming
filter gives the same text as the one-shot cleaner for every split of the text. Worksheets have question numbers of their own: with bare `[n]` headings the model cited
question numbers 6 and 7 as sources in 3 of 13 answers; with `SOURCE n` headings plus "numbers inside a source are NOT source numbers" it was 0 of 13 (and the
answers got full sentences). One follow-up still cited a question number once; that was stripped and the likely page named in a note.

## "Not in your papers"

Calibrated 2026-10-17 on three real worksheet photos (nomic embeddings, `qwen3-vl-8b`), 13 answerable questions and 12 unanswerable ones written for this
(6 far-off topics such as a World Cup or the French Revolution, 6 near-misses that share words with the pages such as "What is the capital of the Philippines?"):

| | raw cosine of the best source |
|---|---|
| answerable (13) | 0.665 - 0.818 |
| far-off, unanswerable (6) | 0.488 - 0.622 |
| near-miss, unanswerable (6) | 0.598 - 0.725 |

The classes overlap between 0.665 and 0.725, so no thresholds separate them. The rule is answerable if `cosine >= 0.68`, or `cosine >= 0.64` **and** at least half of the
question's content words occur in that source. This gate is deliberately lenient (high 0.68, low 0.64, coverage 0.5); the model's own `NOT_IN_PAPERS` reply (prompt rule) handles near-misses.

| | false refusals (answerable refused) | false answers (unanswerable answered) |
|---|---|---|
| gate alone, chosen thresholds | 0 / 13 | 3 / 12 (far 0/6, near 3/6) |
| model alone (gate off) | 0 / 13 | 0 / 12 |
| **full pipeline** | **0 / 13** | **0 / 12** |
| a strict gate (0.732 / 0.564 / 0.80) alone | 2 / 13 | 0 / 12 |

In the full run 9 of the 12 were refused by the gate (no model call) and 3 by the model. n is small and comes from one worksheet series: **re-run
`scripts/calibrate_gate.py` as more pages and questions are added**. The risk of the lenient gate is a near-miss the model answers anyway; the strict gate costs real answers.
Tradeoff chosen: never refuse something the papers do answer, and let the model say no.

## Query expansion

Measured with `scripts/eval_expansion.py` (13 answerable, 12 unanswerable): answer-page rank hit@1 0.77 to 0.92, MRR 0.84 to 0.94, the "which layer allows plates to
move" question went from rank 7 to 5, no question got worse, the gate let through the same 3 of 12 unanswerable. End to end nothing changed (the answer page was already among the
5 sources for 13/13; 13/13 answered, valid citations, 12/12 refused) because spreading puts the page in; the cost is one extra short call (median 1.3 s to 2.2 s). It is on by default for retrieval robustness; turn it off if latency matters more.

## Context budget

LM Studio's loaded context is read from `GET /api/v0/models` (`SB_CHAT_CONTEXT_TOKENS` overrides it). Tokens are estimated at 3 characters each and tightened from the server's
`prompt_tokens`. After the answer reserve (`SB_CHAT_MAX_ANSWER_TOKENS`, 700), history may use at most a quarter of what is left (oldest turns dropped first) and sources the rest (lowest ranked dropped first,
the best one shortened rather than dropped). If not even one shortened source fits, chat fails with the numbers ("the loaded context is 1100 tokens but chat needs at least 1194 ...").
A real question prompt uses 1,400-1,650 tokens. The loaded context was 13,721 tokens when this was first measured and is **about 40,000 (40,223) since the memory split was changed
(2026-10-18)**; everything under "Overview questions" below was measured at 40,223. A 2,400-token context left out 2 sources with a warning. A larger context means whole documents of
roughly 30,000 tokens (about 12 dense pages) fit without any condensing; only longer ones are condensed.

## Overview questions ("summarize the entire paper", "what is this worksheet about", "buod ng papel na ito")

A question about a whole document has no single answering chunk, so it skips retrieval and the gate. `overview.py` detects it, `targets.py` decides which
document(s), `condense.py` fits them into the context, and `session._overview` reads the pages in order with `OVERVIEW_SYSTEM_PROMPT`. A question that is not an overview
request goes down the normal path unchanged; a topic that matches no paper is not treated as an overview, so the normal path refuses it.

**Which document** (inside the subject filter, first rule that applies): a title the question spells out (60% of its words, no tie); the only document left; "this / latest /
just uploaded" and a document added in the last 60 minutes (that one); up to 3 documents (each summarised); otherwise the student is asked, with the 8 newest listed, and may
reply with a number or part of a title. So "summarize this handout" right after an upload means the upload, not every paper.

**How it fits** (per document, from its share of the context after the answer reserve and prompt):

| tier | when | what the model sees | note shown to the student |
|---|---|---|---|
| full | every page fits | the pages as read | "overview of ..., not an answer found on one page" |
| condensed | does not fit, at most 30 pages, at least 150 tokens a page | each page shortened to key points, at most 800 characters, **four pages per model call** | "I summarized each page first ... fine details may be missing" |
| summary | more than 30 pages or less than 150 tokens a page | nothing: the stored summary written at ingest is **shown as it is**, with no model call | "too long to read at once ... uses its stored summary only; details are missing" |

Pages are never silently dropped: the tier is stated in the answer, and a page that could not be condensed is only shortened and counted in a note. The 800-character cap
exists because 2,800-character pages took 22 s a call, so a 14-page document took 5.6 minutes.

**Condensing in groups.** The model gets up to 4 pages (and at most half the context, a page over that goes alone) and must answer with one section per page, each starting with a line
`PAGE n`. The reply is accepted only if it holds exactly the expected page numbers once each, in order, none empty; each page keeps its own condensed text, so sources and citations stay per page.
Otherwise (or if the call fails) that group is redone one page per call. Pages already under the limit are not sent. Progress shows "Summarizing pages 1-4 of 14 ...".
The half-context cap exists because a first version grouped by page count only: with pages of about 10,000 tokens every group prompt filled the whole context, overflowed, and fell back,
giving 18 calls and 931 s for 14 pages.

**Stored-summary tier.** An overview of an existing document never answers "not in your papers" from this tier: the earlier design gave the summary to the model as a source and the model
refused it. Now the answer is `Title: <stored summary> [1]` plus the note, and `[1]` opens page 1. A document **without** a stored summary shows the first 600 characters of page 1 and the note
"has no stored summary, so this shows only the start of page 1" (a blank first page shows "(the first page has no readable text)"). Next to shorter documents the model answers about those, and
the long one's summary is added to the notes under the answer (the model is not given it and cannot cite it).

**Measured** (`scripts/eval_overview.py --timing`, 2026-10-18, three real sample photos ingested into temporary folders, `qwen3-vl-8b`, loaded context 40,223 tokens):

| group | result |
|---|---|
| answerable (13) | 13 / 13 answered |
| unanswerable (12) and near-miss (6) | 12 / 12 and 6 / 6 refused |
| non-overview questions detected as overview | 0 |
| overview (9) | 8 / 9 |
| 14-page document, condensed | 131.2 s with 5 condensing calls + 1 answer (before grouping: 144.5 s with 14 + 1; the broken first grouping: 931.6 s, 18 + 1) |
| 40-page document, stored-summary tier | 2.0 s, answered, 0 model calls (before: the model replied "not in your papers") |

- **Overview 8/9.** "this / ito / nito" questions now expect the latest upload when it is under 60 minutes old (all of up to 3 older documents, or "asks which paper" for more); the program's
  behaviour was not changed. The one miss, "tungkol saan ang worksheet" (11.5 s), was answered but cited no source; it passed in the earlier run, so an uncited answer to an overview
  question is a real, intermittent weakness (the usual "no source was cited" note is shown). Not investigated further.
- **The 14-page case is a stress test.** Its pages are about 4,000 tokens each (the three sample pages repeated until the document overflows 40,223 tokens by half), 56,000+ tokens in all.
  Real handout pages are 500-1,500 tokens, so a real 14-page reviewer (about 15,000 tokens) fits whole and is never condensed. Condensing matters for documents above roughly 30,000 tokens.
  The 131 s is mostly reading that much text: a normal-size group of four pages takes about 14 s, a group of four 29,000-character pages about 54 s. Whole-answer time after condensing is about 30 s.
- With "this" and more than 3 old documents the student is asked which paper; with up to 3 old documents all are summarised together (covered by tests, and by the evaluation's resolved expectations).
- Run it against **temporary folders** with the samples ingested one file at a time (several files in one `ingest.py once` call are rejected unless `--album`, see `docs/ingest.md`). An index holding
  fewer or more documents than the three samples changes the overview results (4+ documents make "summary of the entire paper" ask which paper): the evaluation index must be exactly the samples.

## Reliability notes

- First request after the model has been idle can take 10-60 s (LM Studio loading it); typical question 1.7-2.1 s end to end, first token about the same.
- Answers are deterministic (temperature 0): two full runs gave identical statuses and citations.
- Not verified on real pages: a cited low-quality page (the three samples read well; the note is covered by unit tests), tables, formulas, two-column pages, handwriting-heavy questions, Filipino or mixed-language questions.
- Source text is fenced as data and the prompt says to ignore instructions inside it; a hostile page can still make the model *say* something wrong, but it cannot make a citation valid that names no source.
