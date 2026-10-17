# Reader bake-off findings and reliability notes

Model: `qwen/qwen3-vl-8b` as READER, RapidOCR as the cross-check. Three real handout photos
(Telegram-compressed, 960x1280, two of them with handwriting), proofread by the student.
Sample images and their text stay outside the repo (`SB_SAMPLES_DIR`). Three pages are a smoke
test, not statistics: treat every number below as provisional.

## Results (2026-10-17, prompt hash in `reader-prompt-baseline.json`)

| | READER CER / WER | OCR CER / WER |
|---|---|---|
| all pages (mean) | 0.011 / 0.018 | 0.284 / 0.345 |
| printed text only | 0.022 / 0.029 (form page), 0.000 / 0.000 (worksheet) | 0.17-0.19 / 0.23 |
| handwritten text only | WER 0.233 and 0.308 | WER 0.567 and 1.000 |

- Image cleanup made no measurable difference on these pages (READER CER 0.011 with cleanup, 0.011 without).
  It stays on because it is cheap and should help on shadows and skew that these photos did not have.
- The classic OCR is used only as a cross-check. Its raw CER is poor on page layouts with a
  neighbouring page or show-through text, because its reading order scrambles.

## Handwriting is NOT reliable (for the Telegram bot message)

The READER reads printed text well and handwriting badly. On the real pages it misread a handwritten
surname, a section code, a score, a date and an answer letter. Nothing in the
quality score can detect these, because the OCR is worse at handwriting than the READER.

The bot must therefore say, whenever a page contains handwriting (form headers, answer letters), something like:

> Handwritten names, dates, scores and answer letters may be wrong. Check them against the page.

Never present a handwritten value as certain, and never use one as a fact in an answer without the page citation.

## Known gaps

- **Checkbox labels dropped.** On the form page the READER omitted the labels "Laboratory" and "Others" and
  lost which boxes were ticked. The prompt has no checkbox rule yet. A rule (`[x]`/`[ ]` before each label)
  will be tested together with more varied pages in one bake-off; do not change the prompt before that.
- **Dropped-words flag cannot see small drops.** The coverage check (flag below 0.85) catches a missing
  paragraph, not two missing words: complete transcripts scored 0.91-0.97 coverage because the OCR merges
  and garbles words.
- **Not yet tested on real pages:** tables, two columns, formulas, sideways photos, glare, full-resolution files.

## Quality score design

`quality = coverage x (1 - 3 x illegible share) x 0.5 if too short x 0.8 if glare`, where coverage is the
share of OCR words found in the READER text. The F1 agreement between the two engines is shown but only
raises the `low_agreement` flag (below 0.50): on the real pages the most accurate transcript had the lowest
agreement (0.72) because the OCR was confused, so using it in the score punished good reads.

Thresholds (heuristics, revisit with more samples): `low_agreement` 0.50, `possible_omission` 0.85,
attention below quality 0.60 or any of the flags empty / error / many_illegible / low_agreement /
possible_omission.
