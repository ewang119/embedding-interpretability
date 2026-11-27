# Updated WWII chunks

Use this directory as the team's canonical **working dataset**. It contains 10,698
passages from 300 source groups, with 40 anchors and 427,920 draft binary decisions.

## Source of truth

- `provenance/raw/` preserves the original supplied files as gzip archives. The
  original byte counts and SHA-256 hashes are in `provenance/raw/manifest.json`.
  These are the input record; their keyword tags are sampling hints, not labels.
- `passages.jsonl` and `anchors.json` define this prepared working snapshot.
  `working_labels` contains AI judgments. The approved `labels` remain null;
  **no passages have human-approved labels**. Ten anchor definitions are provisional.
- CSV files and `training/` are projections of that snapshot. Do not edit a CSV
  and assume the JSONL or other exports changed. Apply reviewed changes together
  and regenerate consistent exports before making a new dataset version.
- Human-reviewed labels against finalized definitions should become the labeling
  ground truth. Neither the supplied keyword tags nor this run's AI labels are
  ground truth. Source-text checks also do not establish historical accuracy.

The original files were rechunked into exact, nonoverlapping two- or three-sentence
spans, 25–70 words each. The initial 10,000-passage selection balanced articles.
Additional retrieval and 44 reused pilot passages supplemented training only.
The earlier pilot is unchanged; do not count those 44 shared IDs twice.

## Files

- `passages.jsonl`: text, provenance, draft/approved labels, and frozen split.
- `anchors.json`: the 40 concept definitions and their status.
- `labeling.csv`: working label matrix for passage-level review.
- `coverage.csv`: positive/negative counts and source support for each anchor.
- `review_queue.csv`: questionable decisions with passage text, model choices,
  and blank human-review fields. Editing it does not update the training export.
- `training/`: aligned texts, labels, metadata, split indices, and embeddings.
- `supplemental_snapshots/`: saved source text supporting 22 verified pilot spans.
- `provenance/`: original inputs, selections, prompts, model judgments, corrections,
  and the frozen source assignments used to build this snapshot.
- `bundle_manifest.json`: byte counts and SHA-256 hashes of committed artifacts.

Every anchor has at least 77 **draft** positives. The two models disagree on at
least one concept in 4,644 passages (43.4%), covering 5,701 binary decisions.
The review queue contains 7,317 unique items, including other uncertainty and
calibration checks. Agreement is not an accuracy measure; review positive and
negative decisions before reporting concept accuracy.

## Splits and embeddings

The frozen split has 7,228 training, 1,223 validation, and 2,247 test passages.
Source groups remain whole. Validation is reserved from the approximate 80%
training allocation; proportions vary with article sizes and training-only additions.
Held-out membership was frozen before embedding-guided retrieval. Different
articles can still contain related material, so source grouping is one leakage check.

`training/embeddings.npy` contains float32 `text-embedding-3-small` vectors with
shape `(10698, 1536)`. Its row IDs and settings are recorded in
`training/embedding_manifest.json`. Preserve row order across vectors, labels,
and metadata. Fit any standardization on training rows only.

From the repository root, verify the committed snapshot without API calls or
private build files:

```bash
python3 -m wwii_corpus.verify_bundle
```

The reusable preparation tools and additional commands are in
[`wwii_corpus/README.md`](../../wwii_corpus/README.md). Local build intermediates
and API checkpoints are Git ignored; this directory is the shared prepared snapshot.
