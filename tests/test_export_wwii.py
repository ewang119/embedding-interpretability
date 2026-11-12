"""Tests for probe/export_wwii.py, using tiny invented fixtures (not training
passages), like tests/test_wwii_validation.py. Run: python -m pytest tests -q"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest
from nltk.tokenize.punkt import PunktParameters, PunktSentenceTokenizer

from probe.data import split_from_meta
from probe.export_wwii import export, main
from validation import validate_wwii as validator


@pytest.fixture(autouse=True)
def offline_tokenizer(monkeypatch):
    params = PunktParameters()
    params.abbrev_types = {"dr", "u.s", "gen"}
    tok = PunktSentenceTokenizer(params)
    monkeypatch.setattr(validator, "sent_tokenize", lambda text, language: tok.tokenize(text))


ANCHORS = {
    "schema_version": 1, "anchor_version": "test-v1",
    "anchors": [
        {"id": "convoys", "name": "Convoys", "definition": "Substantially about convoy protection."},
        {"id": "bombing", "name": "Bombing", "definition": "Substantially about bombing campaigns."},
    ],
}


def row(cid, src, convoys, bombing, split=None):
    r = {"chunk_id": cid, "source_id": src, "source_url": f"https://example.org/{src}",
         "chunk_text": f"Passage {cid} starts here. It has a middle. It ends now.",
         "anchor_version": "test-v1", "labels": {"convoys": convoys, "bombing": bombing}}
    if split:
        r["split"] = split
    return r


def write(tmp_path, rows):
    a, c = tmp_path / "anchors.json", tmp_path / "chunks.jsonl"
    a.write_text(json.dumps(ANCHORS))
    c.write_text("".join(json.dumps(r) + "\n" for r in rows))
    return a, c


def test_exports_three_aligned_files(tmp_path):
    a, c = write(tmp_path, [row("c1", "s1", 1, 0), row("c2", "s1", 0, 1), row("c3", "s2", 1, 1)])
    s = export(a, c, tmp_path / "out", min_positive=1)
    labels = pd.read_csv(tmp_path / "out" / "labels.csv")
    texts = pd.read_csv(tmp_path / "out" / "texts.csv")
    meta = pd.read_csv(tmp_path / "out" / "meta.csv")
    assert list(labels.columns) == ["id", "convoys", "bombing"]          # anchors.json order
    assert labels.id.tolist() == texts.id.tolist() == meta.id.tolist() == ["c1", "c2", "c3"]
    assert labels[["convoys", "bombing"]].values.tolist() == [[1, 0], [0, 1], [1, 1]]
    assert meta.source_id.tolist() == ["s1", "s1", "s2"]
    assert s["exported"] == 3 and s["sources"] == 2 and s["anchors_below_target"] == []


def test_refuses_unreviewed_labels_by_default(tmp_path):
    a, c = write(tmp_path, [row("c1", "s1", 1, 0), row("c2", "s2", None, 1)])
    with pytest.raises(ValueError, match="unreviewed"):
        export(a, c, tmp_path / "out")
    assert not (tmp_path / "out" / "labels.csv").exists()


def test_reviewed_only_skips_and_lists_incomplete_passages(tmp_path):
    a, c = write(tmp_path, [row("c1", "s1", 1, 0), row("c2", "s2", None, 1), row("c3", "s3", 0, 1)])
    s = export(a, c, tmp_path / "out", reviewed_only=True, min_positive=75)
    assert s["skipped_unreviewed"] == ["c2"]
    assert pd.read_csv(tmp_path / "out" / "labels.csv").id.tolist() == ["c1", "c3"]
    assert set(s["anchors_below_target"]) == {"convoys", "bombing"}


def test_refuses_when_validator_finds_errors(tmp_path):
    bad = row("c1", "s1", 1, 0)
    bad["chunk_text"] = "Only one sentence here."
    a, c = write(tmp_path, [bad, row("c2", "s2", 0, 1)])
    with pytest.raises(ValueError, match="sentence_count"):
        export(a, c, tmp_path / "out")


def test_given_split_flows_into_probe_split(tmp_path):
    rows = [row("c1", "s1", 1, 0, "train"), row("c2", "s1", 0, 1, "train"),
            row("c3", "s2", 1, 1, "val"), row("c4", "s3", 0, 1, "test")]
    a, c = write(tmp_path, rows)
    s = export(a, c, tmp_path / "out", min_positive=1)
    assert s["split_given"]
    meta = pd.read_csv(tmp_path / "out" / "meta.csv")
    labels = pd.read_csv(tmp_path / "out" / "labels.csv")
    parts, mode = split_from_meta(meta, labels.id.tolist(), labels[["convoys", "bombing"]].to_numpy())
    assert mode == "given in meta"
    assert [p.tolist() for p in parts] == [[0, 1], [2], [3]]


def test_cli_exits_with_message(tmp_path):
    a, c = write(tmp_path, [row("c1", "s1", None, 0)])
    with pytest.raises(SystemExit, match="not exported"):
        main(["--anchors", str(a), "--chunks", str(c), "--out", str(tmp_path / "out")])
