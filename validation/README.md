# WWII passage collection and validation

This is the input format for the WWII dataset. The validator checks collection
quality and label coverage before embeddings and the final label matrix are built.

The `validation/` folder contains the validator, this guide, and supporting files
in `data/`. Run the commands below from the repository root.

`data/anchors.json` contains the first ten definitions supplied by Pradyum in the team
chat, with stable IDs added. They are a draft, not the complete proposed set of 40.
`data/chunks.jsonl` starts empty: no historical passages or labels have been invented
to populate the dataset. The tiny invented passages used in automated tests are
test fixtures only.

## Collect a passage

Add passages directly to `data/chunks.jsonl`, one JSON object per line, using the
fields below. The optional `data/chunk.template.jsonl` shows a blank entry; the
validator does not read it. Do not validate that blank example as a collected passage.

| Field | Rule |
|---|---|
| `chunk_id` | Unique ID for this passage; keep it stable when correcting labels. |
| `source_id` | Stable ID shared by all passages from the same article or book. |
| `source_url` | HTTP(S) URL where the passage was obtained. |
| `chunk_text` | Three consecutive sentences copied from the source. |
| `anchor_version` | The version of the definitions used to make these judgments. |
| `labels` | Anchor ID to integer `1` (yes), integer `0` (reviewed no), or `null` (unreviewed). |
| `split` | Optional: `train`, `val`, or `test`; provide for every passage or none. |

Judge the passage against every anchor, not only the concept used to find it.
A passage can express several concepts. A missing label key is counted as
unreviewed, never silently converted to zero. Explicit nulls are preferred.
The validator rejects strings like `"yes"`, floating-point values, and booleans.

Keep passages in their original context and record source revisions/section
locations in additional fields when useful. The script checks URL structure;
it does not fetch the page or verify that the text was copied accurately.
Keep any AI-generated challenge examples in a separate file with their generation
details; do not mix them into this collection as purported source excerpts.

## Run the validator

From the repository root:

```bash
python3 -m validation.validate_wwii \
  --anchors validation/data/anchors.json \
  --chunks validation/data/chunks.jsonl \
  --out results/wwii_validation
```

NLTK is already a project dependency. If the English tokenizer data is missing,
install it once with `python3 -m nltk.downloader punkt_tab`. The validator never
downloads resources automatically. Counts use NLTK's English sentence tokenizer;
review abbreviation/quotation boundary disagreements before editing source text.

Outputs:

- `validation_report.json`: errors with input line numbers, warnings, source
  grouping information, anchor/tokenizer versions, and readiness flags.
- `coverage.csv`: yes/no/unreviewed/invalid counts per anchor, the number of
  distinct positive source IDs, and the shortfall against `--min-positive`.
  `missing` is a subset of `unreviewed`, not an extra category of judgments.

The positive target defaults to 75 and is configurable with `--min-positive`.
It is a collection target motivated by a preliminary Reuters/LSA experiment,
not a guarantee that a concept can be learned. Repeated passages from one source
do not provide the same diversity as independent passages from many sources.
Coverage includes parseable passage objects even when a passage has other errors;
fix the validation errors before trusting the totals as usable training support.

Exit codes:

- `0`: structurally valid collection. Unreviewed labels and low support remain
  warnings while collection is in progress.
- `1`: validation errors, or incomplete/under-target data with `--require-ready`.
- `2`: invalid command-line arguments.

For a stricter collection checkpoint, append `--require-ready`. `ready` requires
valid data, complete labels, and every anchor meeting the chosen positive target.
`ready_for_label_export` only requires valid data and complete labels; a small
fully labeled pilot can therefore be exported before reaching the final target.
Neither flag validates the historical accuracy, human agreement, or final approval
of the anchor definitions. The initially empty collection intentionally fails.

The tool flags duplicate IDs and text after Unicode/whitespace/case normalization.
It also flags a source crossing declared splits, including URL aliases that use
different source IDs. Repeated sources without splits produce a warning. This is
not near-duplicate or semantic similarity detection.

## Helping Pradyum review the anchors

For each existing definition, gather real source passages for discussion:

1. A clear positive, with the sentence or relationship that makes it qualify.
2. A hard negative that mentions similar people, places, or terms but fails the
   substantive-coverage rule. Explain why it should be no.
3. A borderline passage to label independently, then compare judgments.

Record the passage/source IDs and the agreed rationale. Discuss what
"substantially about" means and any restrictions on dates, scale, or causality.
For overlapping anchors, decide when both may be yes versus only one. Useful
pairs from the supplied draft are attrition/logistics versus siege warfare,
appeasement versus the Munich Agreement, and Pacific amphibious campaigns versus
the Normandy landings (different theaters despite some shared military language).
These are review questions, not automatic changes to Pradyum's definitions.

The team chat proposes 40 anchors; it does not explicitly request more anchors
from Eric. Finish and review the existing list before expanding it. Add new
definitions only when the team identifies a meaningful gap with adequate examples.

Update `data/anchors.json` as the rest of the agreed definitions arrive. Change
`anchor_version` when definitions or the anchor set change, update the template,
and re-review affected judgments before marking a passage with the new version.
Changing a version string alone is not a substitute for reviewing the labels.

## Handoff to training

The next preparation step will embed passages with one fixed model and export
`embeddings.npy` and `labels.csv` in the same passage order. This validator does
not yet create either file or assign splits. Keep source IDs for that step:
the existing `probe.run` splitter assigns individual rows and must be adapted
to keep multiple passages from the same source together. Use the same source
split for the later bag-of-words comparison.
