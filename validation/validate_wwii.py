"""Check collected WWII passages and report label coverage before training.

Run from the repository root:
    python3 -m validation.validate_wwii --anchors validation/data/anchors.json \
        --chunks validation/data/chunks.jsonl --out results/wwii_validation
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import unicodedata
from collections import defaultdict
from pathlib import Path
from urllib.parse import urlsplit, urlunsplit

import nltk
from nltk.tokenize import sent_tokenize


COVERAGE_FIELDS = [
    "anchor_id", "name", "yes", "no", "unreviewed", "missing", "invalid",
    "positive_sources", "target_positive", "positive_shortfall",
]


def _unique_object(pairs):
    # A repeated JSON key must not silently replace an earlier label.
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"Repeated JSON key: {key!r}")
        result[key] = value
    return result


def _decode(text):
    return json.loads(text, object_pairs_hook=_unique_object)


def _nonempty(value):
    return isinstance(value, str) and bool(value.strip())


def _normalized_text(text):
    return " ".join(unicodedata.normalize("NFKC", text).split()).casefold()


def _source_url(value):
    if not _nonempty(value):
        return None
    try:
        parts = urlsplit(value.strip())
        if parts.scheme.lower() not in {"http", "https"} or not parts.hostname:
            return None
        if any(char.isspace() for char in value):
            return None
        # Sections of the same article still belong to the same source.
        return urlunsplit((parts.scheme.lower(), parts.netloc.lower(),
                           parts.path.rstrip("/"), parts.query, ""))
    except ValueError:
        return None


def validate_dataset(anchors_path, chunks_path, min_positive=75):
    """Return a JSON-ready validation report and one coverage row per anchor."""
    if type(min_positive) is not int or min_positive < 1:
        raise ValueError("min_positive must be a positive integer")
    anchors_path, chunks_path = Path(anchors_path), Path(chunks_path)
    errors, warnings = [], []

    def issue(code, message, *, warning=False, **context):
        (warnings if warning else errors).append(
            {"code": code, "message": message, **context})

    try:
        payload = _decode(anchors_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, ValueError) as exc:
        issue("anchors_input", str(exc))
        payload = {}
    if not isinstance(payload, dict):
        issue("anchors_schema", "Anchor file must be a JSON object.")
        payload = {}
    if type(payload.get("schema_version")) is not int or payload.get("schema_version") != 1:
        issue("schema_version", "Anchor file needs schema_version: 1.")
    version = payload.get("anchor_version")
    if not _nonempty(version):
        issue("anchor_version", "Anchor file needs a nonempty anchor_version.")
    anchor_rows = payload.get("anchors")
    if not isinstance(anchor_rows, list) or not anchor_rows:
        issue("anchors_empty", "Provide a nonempty anchors list.")
        anchor_rows = []

    anchors = {}
    for index, anchor in enumerate(anchor_rows, 1):
        if not isinstance(anchor, dict):
            issue("anchor_schema", "Each anchor must be an object.", anchor_row=index)
            continue
        anchor_id = anchor.get("id")
        if (not isinstance(anchor_id, str) or
                not re.fullmatch(r"[a-z][a-z0-9_]*", anchor_id) or anchor_id == "id"):
            issue("anchor_id", "Use a lowercase snake_case ID other than 'id'.", anchor_row=index)
            continue
        if anchor_id in anchors:
            issue("duplicate_anchor_id", "Anchor IDs must be unique.", anchor_id=anchor_id)
            continue
        anchors[anchor_id] = anchor
        for field in ("name", "definition"):
            if not _nonempty(anchor.get(field)):
                issue("anchor_definition", f"Anchor needs a nonempty {field}.", anchor_id=anchor_id)

    coverage = {
        key: dict(anchor_id=key, name=value.get("name", key), yes=0, no=0,
                  unreviewed=0, missing=0, invalid=0, positive_sources=0,
                  target_positive=min_positive, positive_shortfall=min_positive)
        for key, value in anchors.items()
    }
    positive_sources = defaultdict(set)
    seen_ids, seen_text = {}, {}
    source_rows, url_rows = defaultdict(list), defaultdict(list)
    rows = []
    tokenizer_available = True
    try:
        lines = chunks_path.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeError) as exc:
        issue("chunks_input", str(exc))
        lines = []
    input_rows = 0
    for line_number, line in enumerate(lines, 1):
        if not line.strip():
            continue
        input_rows += 1
        try:
            row = _decode(line)
        except ValueError as exc:
            issue("chunk_json", str(exc), line=line_number)
            continue
        if not isinstance(row, dict):
            issue("chunk_schema", "Each JSONL row must be an object.", line=line_number)
            continue
        rows.append(row)
        chunk_id = row.get("chunk_id")
        context = {"line": line_number}
        if _nonempty(chunk_id):
            context["chunk_id"] = chunk_id
            if chunk_id in seen_ids:
                issue("duplicate_chunk_id", f"ID first appeared on line {seen_ids[chunk_id]}.", **context)
            seen_ids[chunk_id] = line_number
        else:
            issue("chunk_id", "Provide a nonempty chunk_id.", **context)

        source_id = row.get("source_id")
        if not _nonempty(source_id):
            issue("source_id", "Provide a stable source/article ID for grouping.", **context)
        url = _source_url(row.get("source_url"))
        if url is None:
            issue("source_url", "Provide an HTTP(S) source URL; URLs are checked locally, not fetched.", **context)
        if row.get("anchor_version") != version or not _nonempty(version):
            issue("chunk_anchor_version", "Chunk anchor_version must match the anchor file.", **context)

        split = row.get("split")
        if split is not None and split not in ("train", "val", "test"):
            issue("split", "Optional split must be train, val, or test.", **context)
        source_entry = (line_number, chunk_id, source_id, split)
        if _nonempty(source_id):
            source_rows[source_id].append(source_entry)
        if url:
            url_rows[url].append(source_entry)

        text = row.get("chunk_text")
        if not _nonempty(text):
            issue("chunk_text", "Provide nonempty passage text.", **context)
        else:
            normalized = _normalized_text(text)
            if normalized in seen_text:
                issue("duplicate_text", f"Same normalized text as line {seen_text[normalized]}.", **context)
            else:
                seen_text[normalized] = line_number
            if tokenizer_available:
                try:
                    count = len(sent_tokenize(text, language="english"))
                    if count != 3:
                        issue("sentence_count", f"Expected 3 sentences; tokenizer found {count}.", **context)
                except LookupError:
                    tokenizer_available = False
                    issue("tokenizer_data", "Install tokenizer data with: python3 -m nltk.downloader punkt_tab")

        labels = row.get("labels")
        if not isinstance(labels, dict):
            issue("labels_schema", "Provide a labels object; use null for unreviewed judgments.", **context)
            for counts in coverage.values():
                counts["invalid"] += 1
            continue
        unknown = sorted(set(labels) - set(anchors))
        if unknown:
            issue("unknown_anchors", f"Unknown label IDs: {', '.join(unknown)}.", **context)
        for anchor_id, counts in coverage.items():
            if anchor_id not in labels:
                counts["missing"] += 1
                counts["unreviewed"] += 1
                continue
            value = labels[anchor_id]
            if value is None:
                counts["unreviewed"] += 1
            elif type(value) is int and value in (0, 1):
                counts["yes" if value == 1 else "no"] += 1
                if value == 1 and _nonempty(source_id):
                    positive_sources[anchor_id].add(source_id)
            else:
                counts["invalid"] += 1
                issue("label_value", "Labels must be integer 0, integer 1, or null.",
                      anchor_id=anchor_id, **context)

    if not rows:
        issue("chunks_empty", "No passage objects were found.")

    assigned = [row.get("split") is not None for row in rows]
    if any(assigned) and not all(assigned):
        issue("partial_splits", "Provide splits for every passage or omit them for all passages.")
    for source_id, entries in source_rows.items():
        splits = {entry[3] for entry in entries if isinstance(entry[3], str)}
        if len(splits) > 1:
            issue("source_split_leakage", f"Source occurs in multiple splits: {sorted(splits)}.", source_id=source_id)
    for url, entries in url_rows.items():
        ids = {entry[2] for entry in entries if _nonempty(entry[2])}
        splits = {entry[3] for entry in entries if isinstance(entry[3], str)}
        if len(ids) > 1:
            issue("source_alias", "The same source URL uses multiple source IDs.",
                  warning=True, source_url=url, source_ids=sorted(ids))
        if len(splits) > 1:
            issue("source_url_split_leakage", "The same source URL occurs in multiple splits.", source_url=url)

    shared_sources = {key: len(value) for key, value in source_rows.items() if len(value) > 1}
    if shared_sources:
        issue("shared_sources", "Multiple chunks share a source. Keep each source in one split; probe.run currently splits rows.",
              warning=True)
    for anchor_id, counts in coverage.items():
        counts["positive_sources"] = len(positive_sources[anchor_id])
        counts["positive_shortfall"] = max(0, min_positive - counts["yes"])
    unreviewed = sum(row["unreviewed"] for row in coverage.values())
    missing = sum(row["missing"] for row in coverage.values())
    if unreviewed:
        issue("unreviewed_labels", f"{unreviewed} judgments are unreviewed, including {missing} omitted keys. They remain unknown, not zero.", warning=True)
    below_target = [key for key, counts in coverage.items() if counts["positive_shortfall"]]
    if below_target:
        issue("positive_target", f"{len(below_target)} anchors have fewer than {min_positive} positive passages.",
              warning=True, anchor_ids=below_target)

    complete = bool(rows and coverage) and not unreviewed and not any(
        row["invalid"] for row in coverage.values())
    report = {
        "schema_version": 1,
        "anchors_file": str(anchors_path.resolve()),
        "chunks_file": str(chunks_path.resolve()),
        "anchor_version": version,
        "sentence_tokenizer": {"name": "nltk.sent_tokenize/english", "nltk_version": nltk.__version__},
        "input_rows": input_rows, "n_chunks": len(rows), "n_anchors": len(coverage),
        "n_sources": len(source_rows), "sources_with_multiple_chunks": shared_sources,
        "min_positive": min_positive, "unreviewed_judgments": unreviewed,
        "missing_judgments": missing, "valid": not errors, "labels_complete": complete,
        "ready_for_label_export": not errors and complete,
        "meets_positive_target": bool(coverage) and not below_target,
        "ready": not errors and complete and bool(coverage) and not below_target,
        "errors": errors, "warnings": warnings,
    }
    return report, list(coverage.values())


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--anchors", required=True, type=Path)
    parser.add_argument("--chunks", required=True, type=Path)
    parser.add_argument("--out", type=Path, default=Path("results/wwii_validation"))
    parser.add_argument("--min-positive", type=int, default=75,
                        help="collection target per anchor, not a statistical guarantee (default: 75)")
    parser.add_argument("--require-ready", action="store_true",
                        help="also fail on unreviewed labels or insufficient positive coverage")
    args = parser.parse_args(argv)
    if args.min_positive < 1:
        parser.error("--min-positive must be at least 1")
    report_path, coverage_path = args.out / "validation_report.json", args.out / "coverage.csv"
    inputs = {args.anchors.resolve(), args.chunks.resolve()}
    if report_path.resolve() in inputs or coverage_path.resolve() in inputs:
        parser.error("Report paths must not overwrite an input file.")
    report, coverage = validate_dataset(args.anchors, args.chunks, args.min_positive)
    args.out.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    with coverage_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=COVERAGE_FIELDS)
        writer.writeheader()
        writer.writerows(coverage)
    print(f"{report['n_chunks']} chunks, {report['n_anchors']} anchors, {report['n_sources']} sources")
    print(f"Valid: {report['valid']}; labels complete: {report['labels_complete']}; "
          f"positive target met: {report['meets_positive_target']}")
    print(f"{len(report['errors'])} errors; {len(report['warnings'])} warnings")
    print(f"Reports: {args.out.resolve()}")
    return int(not report["valid"] or (args.require_ready and not report["ready"]))


if __name__ == "__main__":
    raise SystemExit(main())
