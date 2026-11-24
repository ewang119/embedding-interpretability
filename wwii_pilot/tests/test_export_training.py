"""Training handoff checks using invented fixtures, never historical passages."""
from copy import deepcopy
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from probe.data import load, split_from_meta
from wwii_pilot.build_shared_pool import passage_id, write_csv
from wwii_pilot.export_training import export, main


ANCHORS = {
    "anchor_version": "fixture-v1",
    "anchors": [
        {"id": "industry", "code": "A01", "name": "Industry", "definition_status": "user_supplied"},
        {"id": "convoys", "code": "A02", "name": "Convoys", "definition_status": "user_supplied"},
    ],
}


def fixture_rows(draft=False):
    rows = []
    for i in range(9):
        text = (f"Alpha{i} Bravo{i} Charlie{i}. Delta{i} Echo{i} Foxtrot{i}. "
                f"Golf{i} Hotel{i} India{i}.")
        labels = {"convoys": i % 2, "industry": 1 - i % 2}
        rows.append({
            "chunk_id": passage_id(text), "chunk_text": text,
            "anchor_version": "fixture-v1", "review_state": "AI draft" if draft else "Reviewed",
            "source_id": f"fixture-source-{i // 3}",
            "source_url": f"https://example.org/fixtures/{i // 3}",
            "source_title": f"Fixture source {i // 3}", "source_license": "Synthetic test fixture",
            "source_verified": True, "source_status": "manual_source_review",
            "working_labels": labels,
            "labels": {key: None for key in labels} if draft else dict(labels),
            "split": ("train", "val", "test")[i // 3],
        })
    return rows


def write_pool(tmp_path, rows=None, anchors=None):
    pool = tmp_path / "pool"
    pool.mkdir()
    document = deepcopy(anchors or ANCHORS)
    rows = rows if rows is not None else fixture_rows()
    (pool / "anchors.json").write_text(json.dumps(document))
    (pool / "passages.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows))
    aids = [a["id"] for a in document["anchors"]]
    fields = ["passage_id", "passage_text", "review_state", "source_id", "source_url", "source_verified", *aids]
    write_csv(pool / "labeling.csv", fields, [
        {"passage_id": row["chunk_id"], "passage_text": row["chunk_text"],
         "review_state": row["review_state"], "source_id": row.get("source_id") or "",
         "source_url": row.get("source_url") or "",
         "source_verified": "yes" if row["source_verified"] else "no", **row["working_labels"]}
        for row in rows
    ])
    return pool


def test_aligned_files_preserve_passage_and_anchor_order(tmp_path):
    rows = fixture_rows()
    pool = write_pool(tmp_path, rows)
    before = {p.name: p.read_bytes() for p in pool.iterdir()}
    out = tmp_path / "export"
    report = export(pool, out, min_test_pos=1)
    labels, texts, meta = [pd.read_csv(out / name) for name in ("labels.csv", "texts.csv", "meta.csv")]
    assert labels.columns.tolist() == ["id", "industry", "convoys"]
    assert labels.id.tolist() == texts.id.tolist() == meta.id.tolist() == [r["chunk_id"] for r in rows]
    assert texts.text.tolist() == [r["chunk_text"] for r in rows]
    assert meta.source_title.tolist() == [r["source_title"] for r in rows]
    assert meta.source_license.tolist() == [r["source_license"] for r in rows]
    assert labels[["industry", "convoys"]].values.tolist() == [
        [r["labels"]["industry"], r["labels"]["convoys"]] for r in rows
    ]
    assert not report["exploratory_only"] and report["label_field"] == "labels"
    assert report["split_sizes"] == {"train": 3, "val": 3, "test": 3}
    for name, expected in report["output_sha256"].items():
        assert hashlib.sha256((out / name).read_bytes()).hexdigest() == expected
    assert {p.name: p.read_bytes() for p in pool.iterdir()} == before


def test_export_loads_in_probe_and_reuses_frozen_split(tmp_path):
    rows = fixture_rows()
    out = tmp_path / "export"
    export(write_pool(tmp_path, rows), out)
    np.save(tmp_path / "vectors.npy", np.zeros((len(rows), 4), dtype=np.float32))
    data = load(tmp_path / "vectors.npy", out / "labels.csv")
    meta = pd.read_csv(out / "meta.csv")
    parts, mode = split_from_meta(meta.sample(frac=1, random_state=3), data.ids, data.Y)
    assert mode == "given in meta"
    saved = json.loads((out / "split.json").read_text())
    for name, part in zip(("train", "val", "test"), parts):
        assert saved[name] == [data.ids[i] for i in part]
    source_sets = [set(meta.set_index("id").loc[saved[name], "source_id"]) for name in saved]
    assert all(not source_sets[i] & source_sets[j] for i in range(3) for j in range(i))


def test_draft_labels_require_explicit_option(tmp_path):
    pool = write_pool(tmp_path, fixture_rows(draft=True))
    with pytest.raises(ValueError, match="not Reviewed"):
        export(pool, tmp_path / "rejected")
    assert not (tmp_path / "rejected").exists()
    result = export(pool, tmp_path / "draft", allow_ai_drafts=True)
    assert result["label_field"] == "working_labels"
    assert result["exploratory_only"]
    assert "working_labels_selected" in result["exploratory_reasons"]


def test_reviewed_subset_lists_skipped_passages(tmp_path):
    rows = fixture_rows()
    rows[0]["review_state"] = "AI draft"
    rows[0]["labels"] = {"industry": None, "convoys": None}
    result = export(write_pool(tmp_path, rows), tmp_path / "export", reviewed_only=True)
    assert result["exported"] == 8
    assert result["skipped"] == [{"id": rows[0]["chunk_id"], "reasons": ["not_reviewed"]}]


def test_missing_sources_require_explicit_subset_and_are_reported(tmp_path):
    rows = fixture_rows()
    rows[0].update(source_id=None, source_url=None, source_verified=False)
    pool = write_pool(tmp_path, rows)
    with pytest.raises(ValueError, match="lack a source ID"):
        export(pool, tmp_path / "rejected")
    result = export(pool, tmp_path / "known", known_sources_only=True)
    assert result["exported"] == 8 and result["source_policy"] == "known_only"
    assert result["skipped"] == [{"id": rows[0]["chunk_id"], "reasons": ["missing_source"]}]


def test_verified_subset_excludes_known_but_unverified_sources(tmp_path):
    rows = fixture_rows()
    rows[0]["source_verified"] = False
    pool = write_pool(tmp_path, rows)
    result = export(pool, tmp_path / "verified", verified_only=True)
    assert result["exported"] == 8 and result["unverified_source_passages"] == 0
    assert result["skipped"][0]["reasons"] == ["source_not_verified"]


@pytest.mark.parametrize("bad", [None, True, 1.0, "1"])
def test_blank_or_noninteger_labels_are_never_converted_to_zero(tmp_path, bad):
    rows = fixture_rows(draft=True)
    rows[0]["working_labels"]["industry"] = bad
    pool = write_pool(tmp_path, rows)
    with pytest.raises(ValueError, match="unreviewed labels|integer 0/1"):
        export(pool, tmp_path / "export", allow_ai_drafts=True)
    assert not (tmp_path / "export").exists()


@pytest.mark.parametrize("column,value", [("source_id", "edited-source"), ("review_state", "Needs review")])
def test_unimported_csv_metadata_edits_are_rejected(tmp_path, column, value):
    pool = write_pool(tmp_path)
    path = pool / "labeling.csv"
    with path.open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        fields = reader.fieldnames
        edits = list(reader)
    edits[0][column] = value
    write_csv(path, fields, edits)
    with pytest.raises(ValueError, match="CSV and JSONL disagree"):
        export(pool, tmp_path / "export")


def test_invalid_passage_identity_fails_audit_before_writing(tmp_path):
    rows = fixture_rows()
    rows[0]["chunk_id"] = "changed-id"
    with pytest.raises(ValueError, match="text/ID mismatch"):
        export(write_pool(tmp_path, rows), tmp_path / "export")
    assert not (tmp_path / "export").exists()


def test_partial_split_is_rejected(tmp_path):
    rows = fixture_rows()
    rows[0]["split"] = None
    with pytest.raises(ValueError, match="partially filled"):
        export(write_pool(tmp_path, rows), tmp_path / "export")


def test_source_alias_cannot_cross_splits(tmp_path):
    rows = fixture_rows()
    rows[3]["source_url"] = rows[0]["source_url"]
    with pytest.raises(ValueError, match="source URL crosses splits"):
        export(write_pool(tmp_path, rows), tmp_path / "export")


def test_generated_split_is_reproducible_and_preserves_sources(tmp_path):
    rows = fixture_rows()
    for row in rows:
        row["split"] = None
    pool = write_pool(tmp_path, rows)
    for name in ("first", "second"):
        export(pool, tmp_path / name, split=(1 / 3, 1 / 3, 1 / 3), seed=7)
    first, second = tmp_path / "first", tmp_path / "second"
    assert {p.name: p.read_bytes() for p in first.iterdir()} == {p.name: p.read_bytes() for p in second.iterdir()}
    meta = pd.read_csv(first / "meta.csv")
    assert (meta.groupby("source_id")["split"].nunique() == 1).all()


def test_provisional_definitions_and_weak_test_support_are_explicit(tmp_path):
    anchors = deepcopy(ANCHORS)
    anchors["anchors"][0]["definition_status"] = "provisional_inferred"
    result = export(write_pool(tmp_path, anchors=anchors), tmp_path / "export")
    assert result["provisional_anchor_ids"] == ["industry"]
    assert result["exploratory_only"]
    assert result["anchors_below_min_test_positives"] == ["industry", "convoys"]
    assert "insufficient_per_split_label_support" in result["exploratory_reasons"]


def test_existing_output_is_not_overwritten_or_mixed_with_embeddings(tmp_path):
    pool = write_pool(tmp_path)
    out = tmp_path / "existing"
    out.mkdir()
    (out / "embeddings.npy").write_bytes(b"existing-vector-file")
    with pytest.raises(ValueError, match="output path already exists"):
        export(pool, out)
    assert list(out.iterdir()) == [out / "embeddings.npy"]
    assert (out / "embeddings.npy").read_bytes() == b"existing-vector-file"


@pytest.mark.parametrize("split", [(0.8, 0, 0.2), (-0.1, 0.3, 0.8), (0.7, 0.1, 0.3), (float("nan"), 0.1, 0.9)])
def test_invalid_split_fractions_are_rejected(tmp_path, split):
    with pytest.raises(ValueError, match="positive fractions summing to 1"):
        export(write_pool(tmp_path), tmp_path / "export", split=split)


def test_cli_reports_draft_requirement_without_writing_output(tmp_path):
    pool = write_pool(tmp_path, fixture_rows(draft=True))
    with pytest.raises(SystemExit, match="not exported:.*not Reviewed"):
        main(["--pool", str(pool), "--out", str(tmp_path / "export")])
    assert not (tmp_path / "export").exists()
