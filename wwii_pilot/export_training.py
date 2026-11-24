"""Export the pilot as aligned training files with a frozen source-grouped split."""
from __future__ import annotations

import argparse
from collections import Counter
from copy import deepcopy
import csv
import hashlib
import json
import math
from pathlib import Path
import sys
import tempfile
from urllib.parse import urlsplit

import numpy as np
import pandas as pd

from probe.data import SPLIT_NAMES, label_report, save_split, split_from_meta
from .audit_shared_pool import audit
from .build_shared_pool import DEFAULT_OUT, read_jsonl
from .refresh_shared_pool import apply_csv


def _sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _write_csv(path, fields, rows):
    with Path(path).open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _check_labels(values, anchor_ids, passage_id, field, complete=False):
    if not isinstance(values, dict) or set(values) != set(anchor_ids):
        raise ValueError(f"{passage_id}: {field} must contain exactly the anchor IDs")
    if any(value is not None and (type(value) is not int or value not in (0, 1))
           for value in values.values()):
        raise ValueError(f"{passage_id}: {field} must use integer 0/1 or null")
    if complete and any(value is None for value in values.values()):
        raise ValueError(f"{passage_id}: {field} has unreviewed labels")


def _load_pool(pool):
    document = json.loads((pool / "anchors.json").read_text(encoding="utf-8"))
    anchors = document["anchors"]
    anchor_ids = [anchor["id"] for anchor in anchors]
    if not anchor_ids or len(set(anchor_ids)) != len(anchor_ids):
        raise ValueError("anchors must have nonempty, unique IDs")
    if any(not isinstance(aid, str) or not aid.strip() for aid in anchor_ids):
        raise ValueError("anchor IDs must be nonblank strings")
    rows = read_jsonl(pool / "passages.jsonl")
    if not rows:
        raise ValueError("the pilot has no passages")
    ids = [row["chunk_id"] for row in rows]
    if len(set(ids)) != len(ids):
        raise ValueError("duplicate passage IDs")
    for row in rows:
        pid = row["chunk_id"]
        _check_labels(row["working_labels"], anchor_ids, pid, "working_labels")
        _check_labels(row["labels"], anchor_ids, pid, "labels",
                      complete=row["review_state"] == "Reviewed")
        if row["anchor_version"] != document["anchor_version"]:
            raise ValueError(f"{pid}: anchor_version differs from anchors.json")
        if type(row.get("source_verified")) is not bool:
            raise ValueError(f"{pid}: source_verified must be Boolean")
        if row["review_state"] == "Reviewed" and row["labels"] != row["working_labels"]:
            raise ValueError(f"{pid}: approved and working labels disagree; refresh the pilot")
    # Metadata edits in the review CSV must be imported, just like label edits.
    refreshed = apply_csv(deepcopy(rows), pool / "labeling.csv", anchors)
    if refreshed != rows:
        raise ValueError("CSV and JSONL disagree; run wwii_pilot.refresh_shared_pool first")
    report = audit(pool)
    if report["errors"]:
        raise ValueError("pilot audit failed: " + "; ".join(report["errors"][:5]))
    if report["near_duplicate_pairs"]:
        raise ValueError("resolve near-duplicate passages and their split grouping before export")
    return document, rows


def _has_source(row):
    sid, url = row.get("source_id"), row.get("source_url")
    if not isinstance(sid, str) or not sid.strip() or not isinstance(url, str):
        return False
    parsed = urlsplit(url.strip())
    return parsed.scheme in {"http", "https"} and bool(parsed.netloc)


def export(pool, out, *, allow_ai_drafts=False, reviewed_only=False,
           known_sources_only=False, verified_only=False,
           split=(0.7, 0.1, 0.2), seed=0, min_test_pos=5):
    """Read the pilot without changing it; publish one complete export directory.

    Default: require approved labels and complete source metadata for every row.
    Explicit subset flags record every skipped ID. Draft exports use working_labels
    and remain exploratory, even when their file and split checks pass.
    """
    pool, out = Path(pool).resolve(), Path(out).resolve()
    if allow_ai_drafts and reviewed_only:
        raise ValueError("--allow-ai-drafts and --reviewed-only are mutually exclusive")
    if known_sources_only and verified_only:
        raise ValueError("choose only one source subset option")
    if (len(split) != 3 or any(not math.isfinite(x) or x <= 0 for x in split)
            or not math.isclose(sum(split), 1.0, abs_tol=1e-9)):
        raise ValueError("split requires three positive fractions summing to 1")
    if min_test_pos < 1:
        raise ValueError("min_test_pos must be at least 1")
    if out.exists():
        raise ValueError("output path already exists; choose a new directory to avoid stale training files")
    document, all_rows = _load_pool(pool)
    anchors = document["anchors"]
    anchor_ids = [anchor["id"] for anchor in anchors]
    rows, skipped = [], []
    for row in all_rows:
        reasons = []
        if reviewed_only and row["review_state"] != "Reviewed":
            reasons.append("not_reviewed")
        if known_sources_only and not _has_source(row):
            reasons.append("missing_source")
        if verified_only and (not _has_source(row) or not row["source_verified"]):
            reasons.append("source_not_verified")
        if reasons:
            skipped.append({"id": row["chunk_id"], "reasons": reasons})
        else:
            rows.append(row)
    if not rows:
        raise ValueError("no passages remain under the selected review/source policy")
    label_field = "working_labels" if allow_ai_drafts else "labels"
    if not allow_ai_drafts:
        unreviewed = sum(row["review_state"] != "Reviewed" for row in rows)
        if unreviewed:
            raise ValueError(f"{unreviewed} passage(s) are not Reviewed; finish review, select "
                             "--reviewed-only, or use --allow-ai-drafts for an exploratory run")
    else:
        if any(row["review_state"] in {"Conflict", "Unreviewed"} for row in rows):
            raise ValueError("resolve Conflict/Unreviewed passages before exporting working labels")
    for row in rows:
        _check_labels(row[label_field], anchor_ids, row["chunk_id"], label_field, complete=True)
    missing_sources = sum(not _has_source(row) for row in rows)
    if missing_sources:
        raise ValueError(f"{missing_sources} passage(s) lack a source ID or HTTP(S) URL; complete "
                         "provenance or explicitly select --known-sources-only or --verified-only")

    ids = [row["chunk_id"] for row in rows]
    Y = np.asarray([[row[label_field][aid] for aid in anchor_ids] for row in rows], dtype=np.float32)
    meta = pd.DataFrame({
        "id": ids,
        "source_id": [row["source_id"].strip() for row in rows],
        "split": [row.get("split") for row in rows],
    })
    parts, split_mode = split_from_meta(meta, ids, Y, split, seed)
    if any(len(part) == 0 for part in parts):
        raise ValueError("source grouping leaves an empty split; add source groups or revise the split")
    # Recovered source aliases must not let the same article cross splits.
    split_for = {int(i): name for name, part in zip(SPLIT_NAMES, parts) for i in part}
    url_splits = {}
    for i, row in enumerate(rows):
        url_splits.setdefault(row["source_url"].strip().rstrip("/"), set()).add(split_for[i])
    if any(len(names) > 1 for names in url_splits.values()):
        raise ValueError("a source URL crosses splits under different source IDs; reconcile source aliases")
    counts = label_report(Y, parts, anchor_ids)
    below_test = counts.index[counts["test"] < min_test_pos].tolist()
    no_train_positive = counts.index[counts["train"] == 0].tolist()
    provisional = [a["id"] for a in anchors if a["definition_status"] == "provisional_inferred"]
    unverified = sum(not row["source_verified"] for row in rows)
    exploratory_reasons = []
    if allow_ai_drafts:
        exploratory_reasons.append("working_labels_selected")
    if provisional:
        exploratory_reasons.append("provisional_anchor_definitions")
    if unverified:
        exploratory_reasons.append("unverified_source_links")
    if below_test or no_train_positive:
        exploratory_reasons.append("insufficient_per_split_label_support")
    summary = {
        "schema_version": 1,
        "anchor_version": document["anchor_version"],
        "input_sha256": {name: _sha256(pool / name)
                         for name in ("anchors.json", "passages.jsonl", "labeling.csv")},
        "input_passages": len(all_rows),
        "exported": len(rows),
        "anchors": len(anchors),
        "anchor_order": anchor_ids,
        "label_field": label_field,
        "review_states": dict(Counter(row["review_state"] for row in rows)),
        "source_policy": "verified_only" if verified_only else "known_only" if known_sources_only else "all",
        "sources": int(meta["source_id"].nunique()),
        "unverified_source_passages": unverified,
        "skipped": skipped,
        "split_mode": split_mode,
        "split_sizes": {name: len(part) for name, part in zip(SPLIT_NAMES, parts)},
        "requested_split": list(split),
        "seed": seed,
        "min_test_pos": min_test_pos,
        "anchors_below_min_test_positives": below_test,
        "anchors_without_training_positives": no_train_positive,
        "provisional_anchor_ids": provisional,
        "exploratory_only": bool(exploratory_reasons),
        "exploratory_reasons": exploratory_reasons,
        "row_ids_sha256": hashlib.sha256(("\n".join(ids) + "\n").encode()).hexdigest(),
        "split_library_versions": {"numpy": np.__version__, "pandas": pd.__version__},
        "embedding_contract": "Embed texts.csv in its existing row order with one fixed model, pooling, "
                              "and normalization; record those settings and the generated row IDs.",
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".wwii-export-", dir=out.parent) as temp:
        stage = Path(temp) / "bundle"
        stage.mkdir()
        _write_csv(stage / "labels.csv", ["id", *anchor_ids],
                   ({"id": row["chunk_id"], **row[label_field]} for row in rows))
        _write_csv(stage / "texts.csv", ["id", "text"],
                   ({"id": row["chunk_id"], "text": row["chunk_text"]} for row in rows))
        _write_csv(stage / "meta.csv",
                   ["id", "source_id", "split", "source_url", "source_title", "source_license",
                    "anchor_version", "review_state", "source_verified"],
                   ({"id": row["chunk_id"], "source_id": row["source_id"].strip(), "split": split_for[i],
                     "source_url": row["source_url"].strip(), "source_title": row.get("source_title") or "",
                     "source_license": row.get("source_license") or "", "anchor_version": row["anchor_version"],
                     "review_state": row["review_state"], "source_verified": row["source_verified"]}
                    for i, row in enumerate(rows)))
        save_split(stage / "split.json", parts, ids)
        _write_csv(stage / "label_counts.csv", ["anchor_id", "train", "val", "test", "total"],
                   ({"anchor_id": aid, **{name: int(counts.loc[aid, name]) for name in SPLIT_NAMES},
                     "total": int(Y[:, i].sum())} for i, aid in enumerate(anchor_ids)))
        summary["output_sha256"] = {name: _sha256(stage / name) for name in
                                    ("labels.csv", "texts.csv", "meta.csv", "split.json", "label_counts.csv")}
        (stage / "export.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
        stage.rename(out)
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pool", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--out", type=Path, required=True, help="new output directory")
    labels = parser.add_mutually_exclusive_group()
    labels.add_argument("--allow-ai-drafts", action="store_true", help="export working_labels for exploratory runs")
    labels.add_argument("--reviewed-only", action="store_true", help="explicitly exclude passages not marked Reviewed")
    sources = parser.add_mutually_exclusive_group()
    sources.add_argument("--known-sources-only", action="store_true", help="exclude passages missing source IDs/URLs")
    sources.add_argument("--verified-only", action="store_true", help="exclude passages without verified sources")
    parser.add_argument("--split", type=float, nargs=3, default=(0.7, 0.1, 0.2), metavar=("TRAIN", "VAL", "TEST"))
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--min-test-pos", type=int, default=5)
    args = parser.parse_args(argv)
    try:
        result = export(args.pool, args.out, allow_ai_drafts=args.allow_ai_drafts,
                        reviewed_only=args.reviewed_only, known_sources_only=args.known_sources_only,
                        verified_only=args.verified_only, split=tuple(args.split), seed=args.seed,
                        min_test_pos=args.min_test_pos)
    except (ValueError, OSError, KeyError, TypeError, IndexError) as error:
        sys.exit(f"not exported: {error}")
    print(f"Exported {result['exported']}/{result['input_passages']} passages, "
          f"{result['anchors']} anchors, {result['sources']} source groups -> {args.out}")
    print(f"Frozen split: {result['split_sizes']}; labels: {result['label_field']}")
    if result["exploratory_only"]:
        print("Exploratory only: " + ", ".join(result["exploratory_reasons"]))
    if result["anchors_below_min_test_positives"]:
        print(f"{len(result['anchors_below_min_test_positives'])} anchors have fewer than "
              f"{result['min_test_pos']} test positives; probe.run drops these at the same threshold.")
    return result


if __name__ == "__main__":
    main()
