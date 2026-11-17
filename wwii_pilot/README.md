# WWII pilot dataset

Everything for this pilot lives in `wwii_pilot/`, separate from the older collection validator and Reuters notebooks. The Python tools use only the standard library.

This is one shared pool, with a separate yes/no judgment for every passage–anchor pair. Labels are multi-label: one passage may support several anchors. The original 12-positive/3-negative collection buckets are retained as provenance, not copied into all other columns as negatives.

`data/anchors.json` contains **40** anchors. The optional 41st anchor, **Evacuation and refugees**, is excluded. The first 30 definitions come from the supplied notes; the final 10 are explicitly marked provisional. Labels made with provisional definitions need another review after those definitions are finalized.

## Files

Open `data/wwii_labeling.xlsx` for the review workbook, or `data/coverage.md` for a readable count table. `data/new_passages.md` is a copyable list of the 550 new excerpts and their draft positive anchors.


- `data/labeling.csv`: editable labels and source information. `1` means yes, `0` means reviewed no, and blank means unreviewed. Current numeric labels are **AI drafts** until a person checks all 40 and marks the row `Reviewed`.
- `data/passages.jsonl`: passage IDs, original collection annotations, draft/working/approved labels, source provenance, and review notes. `labels` contains approved judgments only; `working_labels` drives current counts.
- `data/coverage.csv`: positive, negative, and unreviewed counts for every anchor, plus source coverage and remaining positive targets.
- `data/collection_priorities.csv`: same counts sorted by shortfall, then available source support.
- `data/sources.csv`: provenance and review notes. A recovered candidate URL is not the same as a verified original source.
- `data/changed_target_labels.csv`: original bucket labels changed by the semantic review.
- `data/duplicates.csv` and `data/near_duplicates.csv`: identity/overlap checks. Near matches need joint review and must stay together in any eventual split.
- `data/collection/`: downloaded source snapshots, exact paragraph/sentence locations, and explicit semantic judgments for newly collected passages.
- `data/inputs/`: unchanged copies of the original excerpts and anchor notes.

The pool now contains **1,000 passages: all 450 imported passages plus 550 new exact source excerpts**. Each has 40 AI draft judgments. Every anchor has at least **25 draft positives** after stratified selection. `data/selection_report.json` records how 550 were selected from 678 reviewed new candidates without changing their labels. This is a collection milestone, not a training adequacy claim.

The passage target is 1,000. **75 positives per anchor is an editable planning target inherited from the validator**, not an established minimum for this experiment or a promise that 1,000 passages will meet it. Sampling is directed toward weak anchors, so these counts do not estimate how often concepts occur in an unbiased WWII corpus.

## Refresh after editing labels

Run from the repository root:

```bash
python3 -m wwii_pilot.refresh_shared_pool
```

This reads `data/labeling.csv`, preserves blanks, validates IDs and review states, backs up editable data, and regenerates JSONL and coverage reports. Text changes require a new passage ID. It refuses to silently drop rows. To use an exported replacement CSV:

```bash
python3 -m wwii_pilot.refresh_shared_pool --csv /absolute/path/to/labeling.csv
```

The Excel workbook is a convenient review copy with live coverage formulas. Its anchor columns use A01–A40, matching `data/anchors.json` order. **Saving the workbook does not update the repository CSV automatically.** Import the saved workbook to refresh all repository reports:

```bash
python3 -m wwii_pilot.refresh_shared_pool --xlsx /absolute/path/to/wwii_labeling.xlsx
```

The importer reads the `Labels` sheet, maps A01–A40 back to stable anchor IDs, preserves blanks, and refuses incomplete rows marked `Reviewed`. Keep one authoritative editing copy.

## Collection and review rules

New passages contain 2–3 consecutive source sentences, at most 70 words, with no repeated passages or source sentences and at most three passages per source article or underlying book. Extracted text preserves words; only HTML/citation markers and whitespace are normalized. Each new passage stores the original HTML hash and the exact location in a saved paragraph snapshot. Government-authored prose and openly licensed Wikipedia text are used with source attribution; Wikipedia passages retain the CC BY-SA 4.0 license and revision metadata where available. Do not assume this license covers unrelated existing supplied passages.

Retrieval filters discover candidates only. An explicit passage-level review against all 40 definitions determines numeric labels. A source-topic match alone never creates labels for the other anchors. `source_verified` for new collection means the excerpt matched the saved source text; it does not certify every historical claim or replace human label review.

The original supplied pool includes passages whose original source is still missing, and some recovered URLs have not been rechecked. These are visible in the reports. This working pool is **not ready for training** until source problems, provisional definitions, and draft labels are reviewed. No split is assigned: the eventual 80/20 split must keep whole source works and overlapping passages together, rather than randomly splitting rows.

## Checks

```bash
python3 -m wwii_pilot.audit_shared_pool --report wwii_pilot/data/quality_report.json
```

This checks IDs, binary labels, CSV/JSONL agreement, exact source locations, new-excerpt length, source limits, and repeated sentences. It is read-only apart from the requested report. `data/quality_report.json` records the latest result. The single imported curly-quotation sentence-tokenizer discrepancy was manually read: `a30_01` contains two grammatical sentences.


```bash
python3 -m unittest discover -s wwii_pilot/tests -p test_shared_pool.py -v
```

The older `validation.validate_wwii` command targets the original three-sentence collection format and its separate anchor file. Use this shared-pool workflow for the two-or-three-sentence, 40-anchor draft.

## Included import inputs

`inputs/candidate_sources/` contains only the ten candidate JSONL files needed by the original-pool importer. The unchanged supplied excerpt and anchor notes, plus the explicit AI judgments, are in `data/inputs/` and `data/semantic_reviews_2026-11-17.json`. The 550 selected additions, with explicit all-anchor draft reviews and source locations, are in `data/collection/reviews.jsonl`.

To reconstruct the original 450-row import in a separate directory without overwriting current review edits:

```bash
python3 -m wwii_pilot.build_shared_pool \
  --batch wwii_pilot/data/inputs/excerpts.txt \
  --anchor-notes wwii_pilot/data/inputs/anchor_notes.txt \
  --judgments wwii_pilot/data/semantic_reviews_2026-11-17.json \
  --out /absolute/path/to/rebuilt_wwii_pool
```

To reconstruct the full pool there, copy `wwii_pilot/data/collection/` into that directory, then run `python3 -m wwii_pilot.refresh_shared_pool --out /absolute/path/to/rebuilt_wwii_pool --append /absolute/path/to/rebuilt_wwii_pool/collection/reviews.jsonl`. The checked-in workbook is a review artifact; the refresh command updates CSV/JSON reports but does not rewrite the workbook.
