# The search index

Code: `src/studybuddy/index/`. Everything lives in `SB_DATA_DIR/index/` (never in the repo or the library folder).

## Layout

| Part | File | Holds |
|---|---|---|
| SQLite | `index/studybuddy.db` | the **source of truth**: documents, pages, chunks (all text), FTS5 keyword index |
| LanceDB | `index/vectors/` | only `(chunk id, vector, subject key, doc id)`, rebuildable from SQLite |
| Embedding record | `SB_DATA_DIR/embed_meta.json` | model, vector size and document prefix the vectors were built with |

- `documents`: content hash (unique, used for dedup), library `file_path`, source kind, subject, title, summary, page count.
- `pages`: **one row for every input page**, including empty and failed ones (quality, flags, error, the full reader record as JSON).
- `chunks`: ~1,200 characters (hard cap 1,800), never crossing a page, so every citation has an exact page. Each chunk carries
  `image_path`, the original image of its page (for a PDF it is the PDF itself and `page_number` selects the page).
- Chunking is paragraph-aware: Markdown tables stay whole (or split on row boundaries), a `[figure: ...]` line stays with what
  follows, and ~150 characters of overlap carry context across chunk boundaries. The cap exists because the embedding model
  silently truncates over-long input (verified: a 2,500-word input is accepted at a 2,048-token context).
- Writes: chunk, embed (nothing written yet), one SQLite transaction, then the vectors. A vector failure removes the SQLite rows.

## Search

`index.search(query, k, subject=None, mode="hybrid")` or `python scripts/search.py "question" --subject Science --k 5`.

- **vector**: the question is embedded with `embed_queries` (adds `EMBED_QUERY_PREFIX`) and ranked by cosine similarity.
- **keyword**: FTS5 `bm25` over chunk text and title (porter stemming). Each question word is quoted, stopwords dropped, terms OR-joined,
  so words like `AND`, `NEAR` or `*` in a question are plain words.
- **hybrid** (default): reciprocal rank fusion, `sum(weight / (60 + rank))` over both lists.
- The optional subject filter is case-insensitive and applied before ranking in both stores.
- Every hit has the file path, page, original image path, page quality and flags, both ranks, the **raw cosine similarity and bm25 score**.
  Fused scores only order results; the chat step (chunk 5) must use the raw numbers to decide "the sources do not contain the answer".
  With the nomic model on a tiny test set, relevant chunks scored cosine ~0.70-0.80 and unrelated ones ~0.50-0.55: calibrate on real data.

## When the embedding model (or document prefix) changes

Vectors from different models, sizes or document prefixes cannot be mixed. The index refuses to open with a clear message:

```
[FAIL] EMBEDDING MODEL CHANGED: index was built with 'old' (768 dims), now 'new' (1024 dims). ... Run: python scripts/index_admin.py reindex
```

Procedure:

1. Change `SB_EMBED_MODEL` (and, if the new model needs them, `EMBED_DOC_PREFIX` / `EMBED_QUERY_PREFIX`) in `.env`; load the model in LM Studio.
2. `python scripts/index_admin.py status` shows recorded vs configured model and flags the mismatch (exit code 1).
3. `python scripts/index_admin.py reindex` re-embeds every chunk **from the text stored in SQLite** (no re-reading of photos) into a new
   vector table and swaps it in only when every chunk is done. If it fails or is interrupted, the old vectors and the embedding record
   are untouched; just run it again.
4. `python scripts/index_admin.py status` should now report no problem.

Changing only `EMBED_QUERY_PREFIX` needs no re-index (stored vectors are unaffected). Changing `EMBED_DOC_PREFIX` does.

Other commands: `check` (vectors without chunks and chunks without vectors), `repair` (delete orphan vectors, embed chunks that lack one),
`reset --yes` (delete the whole index; without `--yes` it only lists what it would delete). `reset` never touches the library folder or
your original photos; re-ingest to rebuild. `add-text PATH --subject S` indexes `.txt`/`.md` files (one page each) for manual testing.

## Do the nomic prefixes help?

`python scripts/compare_prefixes.py` (needs proofread pairs plus a `questions.json` in `SB_SAMPLES_DIR`; results go to `SB_DATA_DIR/eval/`).
Vector-only retrieval, 13 questions over 12 chunks from three real pages (2026-10-17, nomic-embed-text-v1.5):

| variant | hit@1 | hit@3 | MRR |
|---|---|---|---|
| no prefixes | 0.69 | 0.92 | 0.79 |
| both prefixes | 0.85 | 0.92 | 0.90 |
| document prefix only | 0.92 | 0.92 | 0.94 |
| query prefix only | 0.69 | 0.85 | 0.78 |

The document prefix carries the benefit; the query prefix alone does nothing. Doc-only beating both is within noise at this size.
Keep both prefixes (that is what the model was trained with) and re-check with a larger question set.

Hybrid vs vector on the same set: vector hit@1 0.85, keyword alone 0.54, hybrid with equal weights 0.77. A keyword weight of 0.5 brought
hybrid back to 0.85 here, but 13 questions over 12 chunks is too little to tune on: the default stays 1.0 until there is more data.

## Known limits

- Chunk quality on tables, two-column pages and formulas is untested on real pages.
- Retrieval quality at scale (hundreds of pages) is untested; the experiments above are tiny.
- One writer at a time (SQLite WAL allows concurrent readers); the ingest watcher in chunk 4 should serialize its writes.
