# Chat

Code: `src/studybuddy/chat/`. Commands: `python scripts/chat.py [--subject Science] [--ask "question"]`,
`python scripts/calibrate_gate.py`, `python scripts/eval_expansion.py`.

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
A real prompt uses 1,400-1,650 tokens of 13,721 (about 12%); a 2,400-token context left out 2 sources with a warning.

## Reliability notes

- First request after the model has been idle can take 10-60 s (LM Studio loading it); typical question 1.7-2.1 s end to end, first token about the same.
- Answers are deterministic (temperature 0): two full runs gave identical statuses and citations.
- Not verified on real pages: a cited low-quality page (the three samples read well; the note is covered by unit tests), tables, formulas, two-column pages, handwriting-heavy questions, Filipino or mixed-language questions.
- Source text is fenced as data and the prompt says to ignore instructions inside it; a hostile page can still make the model *say* something wrong, but it cannot make a citation valid that names no source.
