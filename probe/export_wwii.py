"""Turn validated WWII passages into the files the probe and baseline read.

    python3 -m probe.export_wwii \
        --anchors validation/data/anchors.json \
        --chunks validation/data/chunks.jsonl \
        --out data/wwii

Runs Eric's validator (validation/validate_wwii.py) first and refuses to
export if it finds errors or unreviewed (null) labels, because an
unreviewed label is unknown, not "no". Below-target anchors are allowed,
so a small fully labeled pilot can be exported early.

Writes, all in the same passage order as chunks.jsonl:
  labels.csv    id + one 0/1 column per anchor, in anchors.json order (probe.run --labels)
  texts.csv     id + passage text (input for the embedding step and probe.baseline)
  meta.csv      id, source_id, split, source_url, anchor_version (probe.run --meta)
  export.json   counts, anchor version, and which chunks were skipped

Next: embed texts.csv in row order with one fixed model into
embeddings.npy, then
  python -m probe.run --embeddings data/wwii/embeddings.npy \
      --labels data/wwii/labels.csv --meta data/wwii/meta.csv --out results/wwii_run1

--reviewed-only exports just the passages whose labels are all reviewed,
skipping the rest (they are listed in export.json). Use it for pilot runs
while labeling is still in progress.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import pandas as pd

from validation.validate_wwii import validate_dataset


def read_rows(chunks_path: Path) -> list[dict]:
    return [json.loads(line) for line in chunks_path.read_text(encoding="utf-8").splitlines() if line.strip()]


def export(anchors_path, chunks_path, out, reviewed_only: bool = False, min_positive: int = 75) -> dict:
    anchors_path, chunks_path, out = Path(anchors_path), Path(chunks_path), Path(out)
    report, coverage = validate_dataset(anchors_path, chunks_path, min_positive)
    if not report["valid"]:
        first = "; ".join(f"{e['code']}: {e['message']}" + (f" (line {e['line']})" if "line" in e else "")
                          for e in report["errors"][:5])
        raise ValueError(f"validator found {len(report['errors'])} error(s); fix them first. {first}")

    anchor_ids = [row["anchor_id"] for row in coverage]
    rows = read_rows(chunks_path)
    complete = [r for r in rows if all(r["labels"].get(a) in (0, 1) for a in anchor_ids)]
    skipped = [r["chunk_id"] for r in rows if r not in complete]
    if skipped and not reviewed_only:
        raise ValueError(f"{len(skipped)} passage(s) still have unreviewed labels "
                         f"({report['unreviewed_judgments']} judgments). Finish labeling, or pass "
                         f"--reviewed-only to export just the fully reviewed passages.")
    if not complete:
        raise ValueError("no fully reviewed passages to export")

    ids = [r["chunk_id"] for r in complete]
    labels = pd.DataFrame([[int(r["labels"][a]) for a in anchor_ids] for r in complete], columns=anchor_ids)
    labels.insert(0, "id", ids)
    texts = pd.DataFrame({"id": ids, "text": [r["chunk_text"] for r in complete]})
    meta = pd.DataFrame({
        "id": ids,
        "source_id": [r["source_id"] for r in complete],
        "split": [r.get("split") or "" for r in complete],
        "source_url": [r["source_url"] for r in complete],
        "anchor_version": [r["anchor_version"] for r in complete],
    })

    out.mkdir(parents=True, exist_ok=True)
    labels.to_csv(out / "labels.csv", index=False)
    texts.to_csv(out / "texts.csv", index=False)
    meta.to_csv(out / "meta.csv", index=False)
    yes = labels[anchor_ids].sum()
    summary = {
        "anchor_version": report["anchor_version"],
        "chunks_file_sha256": hashlib.sha256(chunks_path.read_bytes()).hexdigest(),
        "exported": len(ids),
        "skipped_unreviewed": skipped,
        "sources": int(meta.source_id.nunique()),
        "anchors": len(anchor_ids),
        "yes_per_anchor": {a: int(yes[a]) for a in anchor_ids},
        "anchors_below_target": [a for a in anchor_ids if yes[a] < min_positive],
        "min_positive": min_positive,
        "split_given": bool((meta.split != "").all()),
    }
    (out / "export.json").write_text(json.dumps(summary, indent=1))
    return summary


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--anchors", required=True, type=Path)
    ap.add_argument("--chunks", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--reviewed-only", action="store_true",
                    help="skip passages with unreviewed labels instead of refusing")
    ap.add_argument("--min-positive", type=int, default=75)
    args = ap.parse_args(argv)
    try:
        s = export(args.anchors, args.chunks, args.out, args.reviewed_only, args.min_positive)
    except ValueError as e:
        sys.exit(f"not exported: {e}")
    print(f"exported {s['exported']} passages from {s['sources']} sources, {s['anchors']} anchors -> {args.out}")
    if s["skipped_unreviewed"]:
        print(f"skipped {len(s['skipped_unreviewed'])} passages with unreviewed labels")
    if s["anchors_below_target"]:
        print(f"{len(s['anchors_below_target'])} anchors below {s['min_positive']} yes-labels "
              f"(fine for a pilot, not for final results)")
    return s


if __name__ == "__main__":
    main()
