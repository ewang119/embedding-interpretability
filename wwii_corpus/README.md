# WWII source-grouped corpus

This folder prepares the supplied Wikipedia batch for 40-anchor semantic labeling.
It preserves the earlier pilot and keeps incoming keyword scores separate from
training labels. The shared prepared snapshot is in `data/updated_chunks/`.
Local build inputs and intermediates remain under `wwii_corpus/data/` and are Git
ignored. Run commands from the repository root.

The current working pool has 10,698 passages from 300 sources, with 427,920 draft
binary decisions. Every anchor has at least 77 draft positives. These are collection
counts, not human-verified label totals.

The two labeling models disagree on at least one concept in 4,644 passages (43.4%).
There are 5,701 differing binary decisions and 7,317 unique items in the review
queue, including other uncertainty flags and calibration examples. Agreement is
not a measure of correctness; review labels before reporting model accuracy.

## Data flow

1. `prepare.py` copies the supplied JSONL and anchor index into `data/raw/`, records
   their SHA-256 hashes, and creates exact, nonoverlapping spans of two or three
   consecutive sentences, 25–70 words each. It filters obvious reference material,
   incomplete fragments, and repeated sentences, including sentences in the pilot.
   `rejections.jsonl` and `unused_sentences.jsonl` preserve the exclusion record.
2. `select.py` chooses up to 10,000 candidates by round-robin sampling across
   articles, with a deterministic order within each article. Sampling uses no
   keyword hints. This balances sources; it does not estimate natural topic frequency.
3. `label_api.py` presents only passage text and all 40 definitions to the chosen
   model. Every omitted positive code is a draft negative decision. Positives carry
   model-proposed supporting quotes; non-verbatim quotes are flagged for review.
   All-negative historical passages remain eligible. The model separately flags
   fragments, unresolved context, references/captions, and garbled text.
4. `materialize.py` joins completed reviews to their original text and definition
   hashes, excludes flagged unusable passages, and removes near duplicates. It
   prefers existing pilot passages when a near-duplicate component spans both pools.
   It writes draft labels, coverage, a review queue, and aligned training files.

The incoming JSONL contains 10,423 long records from 277 articles. The two Markdown
collections partition those same records; importing both would duplicate the data.
The anchor index supplies keyword-derived suggestions, not approved labels.

## Commands

Install the repository dependencies, including NLTK, NumPy, and pandas. Supply the
original input paths as arguments; no machine-specific paths are built into code.

```bash
python3 -m wwii_corpus.prepare --chunks /path/to/chunks.jsonl \
  --index /path/to/anchor_index.json --setup-nltk
python3 -m wwii_corpus.select
python3 -m wwii_corpus.label_api --model gpt-5-mini-2025-08-07 \
  --budget 4.50 --batch-size 40 --workers 8
python3 -m wwii_corpus.status
python3 -m wwii_corpus.materialize
python3 -m wwii_corpus.audit
python3 -m wwii_corpus.verify_bundle
```

The API command reads `OPENAI_API_KEY` or prompts without echoing. Credentials are
never included in prompts, checkpoints, or output files. The ignored `work/`
directory holds resumable judgments and a SQLite cost ledger. Keep this directory
when resuming: removing the ledger loses both the cache and task cost accounting.
The budget covers all requests recorded in that ledger, including retries and
small calibration runs. Unknown network outcomes retain their conservative maximum
cost reservation. Reported token usage is task accounting, not account-wide billing.
Raw input files and the two large derived candidate files stay local and are Git
ignored. Their manifests and preparation/selection reports preserve the hashes.

Preparation, selection, and materialization refuse to overwrite existing outputs.
Choose a new directory when rebuilding. Labeling resumes completed judgments with
matching model, prompt version, passage text, and definition hashes. Use
`--allow-partial` with materialization only when deliberately exporting an incomplete
run; the report records every unlabeled passage.

An optional second-model pass adds disagreement flags without approving or
silently replacing the primary labels. It shares the same total budget ledger:

```bash
python3 -m wwii_corpus.select \
  --candidates wwii_corpus/data/selected_candidates.jsonl \
  --out wwii_corpus/data/qa_candidates.jsonl --target 2000 --seed 674
python3 -m wwii_corpus.label_api --model gpt-4.1-mini-2025-04-14 \
  --budget 4.50 --batch-size 40 --workers 8 --output-prefix crosscheck \
  --candidates wwii_corpus/data/qa_candidates.jsonl
```

For a deficient anchor, `python3 -m wwii_corpus.enrich --pool /path/to/base_pool`
retrieves extra passages from the unused clean candidates. Hints guide retrieval
only; re-run semantic labeling with `--candidates
wwii_corpus/data/selected_enriched_candidates.jsonl`, then materialize with
`--selection selected_enriched_candidates.jsonl`. Completed primary judgments are
reused from the cache. Enrichment changes topic prevalence and is recorded separately.

The completed run also used embedding similarity to retrieve training candidates
and reused 44 eligible passages from the earlier pilot. All supplements were
re-labeled against every anchor; previous labels were used for retrieval only.
The 22 previously verified source spans retain their saved snapshots. The earlier
pilot remains unchanged. These 44 IDs must not be counted twice when combining pools.

`frozen_source_splits.json` freezes the original source-balanced holdout before
retrieval. Supplementary passages from held-out articles are excluded. All accepted
additions go into training; validation and test membership remain fixed.

## Outputs and review

- `data/cleaned_candidates.jsonl`: exact text spans, provenance, and separate
  parent-level keyword hints; no labels.
- `data/selected_candidates.jsonl`: reproducible source-balanced review selection.
- `data/semantic_reviews.jsonl`: sparse positive codes, uncertainty flags, model
  metadata, and usability decisions for completed requests.
- `data/updated_chunks/passages.jsonl`: complete binary `working_labels`; approved `labels`
  remain null and `review_state` stays `AI draft`.
- `data/updated_chunks/coverage.csv`: positive and negative counts, distinct positive sources,
  and train/validation/test positive counts for each anchor. The 75-positive target
  is a collection planning threshold, not proof of adequate statistical power.
- `data/updated_chunks/review_queue.jsonl`: uncertain decisions plus a deterministic sample
  of five positives, five negatives, and five keyword-candidate negatives per anchor,
  where available. Second-model disagreements and temporal-scope warnings are also
  queued. This is a quality-control sample, not a representative test set.
- `data/updated_chunks/review_queue.csv`: the same review items with passage text, source,
  concept names, model choices, and blank human-review fields. The primary column
  reflects the working draft, including recorded assistant spot corrections.
  Editing this worklist does not automatically update the training export.
- `data/updated_chunks/excluded.jsonl`: exclusions from the selected pool; original inputs
  and cleaned candidates remain available for reassessment.
- `data/spot_checks.jsonl`: text-bound assistant corrections for identified sample
  errors. These supplement raw API judgments and remain AI drafts, with reasons.
- `data/updated_chunks/training/`: aligned `texts.csv`, `labels.csv`, `meta.csv`, and
  `split.json`, plus `embeddings.npy` and its row-ID manifest, with hashes in
  `export.json`. Vectors are generated with `text-embedding-3-small`, 1,536 dimensions.

The exploratory split targets 70% train, 10% validation, and 20% test, leaving
validation data for model selection within the original 80% training allocation.
Whole source groups stay in one split, normalizing article URL fragments and percent encoding.
Actual proportions vary with source group sizes. Source grouping prevents article
leakage; copied or related material across different articles can still bias results.
When combining this corpus with the earlier pilot, create one combined source-group
split. Do not concatenate independently split training and test exports.

## Before reporting accuracy

Model judgments are drafts, not human-reviewed gold. Initial spot checks found
missed overlapping concepts and overly broad assignments; the prompt includes
boundary guidance, but this does not establish label accuracy. Check false negatives
as well as positives. All 40 anchors require review, including the ten provisional
definitions A31–A40. Source spans are verified against the supplied text file;
original webpage revisions were not supplied and are not independently verified.
The verified pilot supplements are identified separately in passage metadata.

Freeze reviewed test labels before tuning or training, check positive support by
anchor and by source, and report unsupported anchors explicitly. Use validation
data for thresholds and hyperparameters. Evaluate held-out performance with
per-anchor precision/recall/F1 and average precision, alongside macro/micro results
and a bag-of-words baseline. AI-label agreement measures imitation of the annotator;
it does not establish human concept accuracy or identify unique semantic dimensions.

The prepared embeddings are committed with the snapshot; temporary API caches
are Git ignored. To regenerate vectors from the
released pool, preserving its existing row order:

```bash
python3 -m wwii_corpus.label_api --mode embed_extra --budget 4.50 \
  --candidates data/updated_chunks/passages.jsonl
python3 -c 'from wwii_corpus.semantic_retrieval import export_embeddings; print(export_embeddings())'
```

No probe training has been run on this pool. For an exploratory run after reviewing
the label limitations, use the exported metadata so the frozen split is respected:

```bash
python3 -m probe.run \
  --embeddings data/updated_chunks/training/embeddings.npy \
  --labels data/updated_chunks/training/labels.csv \
  --meta data/updated_chunks/training/meta.csv \
  --ablation none \
  --out results/wwii_corpus_exploratory
```

```bash
python3 -m pytest wwii_corpus/tests -q
```

The original supplied inputs are archived under `data/updated_chunks/provenance/raw/`.
The shared working records are `data/updated_chunks/passages.jsonl` and
`anchors.json`; the raw keyword tags and AI draft judgments are not human ground truth.
