"""Collection checks use tiny invented fixtures, never training passages."""

import csv
import json

import pytest
from nltk.tokenize.punkt import PunktParameters, PunktSentenceTokenizer

from validation import validate_wwii as validator


@pytest.fixture(autouse=True)
def offline_tokenizer(monkeypatch):
    # Exercise sentence checks without downloading an English model in tests.
    params = PunktParameters()
    params.abbrev_types = {"dr", "u.s", "gen"}
    tokenizer = PunktSentenceTokenizer(params)
    monkeypatch.setattr(validator, "sent_tokenize",
                        lambda text, language: tokenizer.tokenize(text))


def anchor_file():
    return {
        "schema_version": 1, "anchor_version": "test-v1",
        "anchors": [
            {"id": "convoys", "name": "Convoys", "definition": "Substantially about convoy protection."},
            {"id": "bombing", "name": "Bombing", "definition": "Substantially about bombing campaigns."},
        ],
    }


def chunks():
    return [
        {"chunk_id": "c1", "source_id": "article-1", "source_url": "https://example.org/article-1",
         "chunk_text": "Ships carried supplies. Escorts guarded them. The cargo arrived.",
         "anchor_version": "test-v1", "labels": {"convoys": 1, "bombing": 0}},
        {"chunk_id": "c2", "source_id": "article-2", "source_url": "https://example.org/article-2",
         "chunk_text": "Planes crossed the coast. Crews identified a target. The operation continued.",
         "anchor_version": "test-v1", "labels": {"convoys": 0, "bombing": 1}},
    ]


def write_inputs(tmp_path, rows=None, anchors=None):
    a, c = tmp_path / "anchors.json", tmp_path / "chunks.jsonl"
    a.write_text(json.dumps(anchor_file() if anchors is None else anchors), encoding="utf-8")
    rows = chunks() if rows is None else rows
    c.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    return a, c


def error_codes(report):
    return {issue["code"] for issue in report["errors"]}


def test_complete_collection_and_row_order(tmp_path):
    report, coverage = validator.validate_dataset(*write_inputs(tmp_path), min_positive=1)
    assert report["valid"] and report["ready"] and report["ready_for_label_export"]
    assert report["n_chunks"] == 2 and report["n_sources"] == 2
    assert [row["anchor_id"] for row in coverage] == ["convoys", "bombing"]
    assert [(row["yes"], row["no"], row["positive_sources"]) for row in coverage] == [(1, 1, 1), (1, 1, 1)]


def test_missing_and_null_labels_stay_unknown(tmp_path):
    rows = chunks()
    rows[0]["labels"] = {"convoys": None}
    report, coverage = validator.validate_dataset(*write_inputs(tmp_path, rows), min_positive=1)
    assert report["valid"] and not report["labels_complete"] and not report["ready"]
    assert report["unreviewed_judgments"] == 2 and report["missing_judgments"] == 1
    assert coverage[0]["no"] == 1 and coverage[0]["unreviewed"] == 1
    assert coverage[1]["no"] == 0 and coverage[1]["missing"] == 1


@pytest.mark.parametrize("value", [True, False, 1.0, "1", "yes", -1, 2, [], {}])
def test_invalid_labels_are_never_coerced(tmp_path, value):
    rows = chunks()
    rows[0]["labels"]["convoys"] = value
    report, coverage = validator.validate_dataset(*write_inputs(tmp_path, rows), min_positive=1)
    assert "label_value" in error_codes(report) and not report["ready_for_label_export"]
    assert coverage[0]["invalid"] == 1 and coverage[0]["yes"] == 0


def test_unknown_anchor_and_definition_version(tmp_path):
    rows = chunks()
    rows[0]["labels"]["typo"] = 1
    rows[1]["anchor_version"] = "older-version"
    report, _ = validator.validate_dataset(*write_inputs(tmp_path, rows))
    assert {"unknown_anchors", "chunk_anchor_version"} <= error_codes(report)


def test_duplicate_ids_and_normalized_text(tmp_path):
    rows = chunks()
    rows[1]["chunk_id"] = rows[0]["chunk_id"]
    rows[1]["chunk_text"] = "  " + rows[0]["chunk_text"].upper().replace(" ", "\n  ")
    report, _ = validator.validate_dataset(*write_inputs(tmp_path, rows))
    assert {"duplicate_chunk_id", "duplicate_text"} <= error_codes(report)
    assert next(e for e in report["errors"] if e["code"] == "duplicate_text")["line"] == 2


def test_sentence_counts_and_abbreviations(tmp_path):
    rows = chunks()
    rows[0]["chunk_text"] = "Dr. Lane boarded the ship. The U.S. escorts followed. Supplies arrived."
    report, _ = validator.validate_dataset(*write_inputs(tmp_path, rows))
    assert "sentence_count" not in error_codes(report)
    rows[0]["chunk_text"] = "One sentence. Another sentence."
    report, _ = validator.validate_dataset(*write_inputs(tmp_path, rows))
    assert "sentence_count" in error_codes(report)


@pytest.mark.parametrize("field,value,code", [
    ("source_id", "", "source_id"),
    ("source_url", "", "source_url"),
    ("source_url", "file:///invented", "source_url"),
    ("source_url", "https://bad host/page", "source_url"),
    ("chunk_text", "", "chunk_text"),
    ("labels", [], "labels_schema"),
    ("split", "training", "split"),
])
def test_invalid_passage_fields(tmp_path, field, value, code):
    rows = chunks()
    rows[0][field] = value
    report, _ = validator.validate_dataset(*write_inputs(tmp_path, rows))
    assert code in error_codes(report) and not report["ready"]


def test_shared_sources_and_positive_source_counts(tmp_path):
    rows = chunks()
    rows[1]["source_id"] = rows[0]["source_id"]
    rows[1]["source_url"] = rows[0]["source_url"]
    rows[1]["labels"]["convoys"] = 1
    report, coverage = validator.validate_dataset(*write_inputs(tmp_path, rows))
    assert report["valid"] and report["sources_with_multiple_chunks"] == {"article-1": 2}
    assert "shared_sources" in {w["code"] for w in report["warnings"]}
    assert coverage[0]["yes"] == 2 and coverage[0]["positive_sources"] == 1


def test_source_cannot_cross_declared_splits(tmp_path):
    rows = chunks()
    rows[1]["source_id"] = rows[0]["source_id"]
    rows[0]["split"], rows[1]["split"] = "train", "test"
    report, _ = validator.validate_dataset(*write_inputs(tmp_path, rows))
    assert "source_split_leakage" in error_codes(report)


def test_url_aliases_cannot_hide_source_leakage(tmp_path):
    rows = chunks()
    rows[1]["source_url"] = rows[0]["source_url"] + "#another-section"
    rows[0]["split"], rows[1]["split"] = "train", "test"
    report, _ = validator.validate_dataset(*write_inputs(tmp_path, rows))
    assert "source_url_split_leakage" in error_codes(report)
    assert "source_alias" in {w["code"] for w in report["warnings"]}


def test_partial_splits_are_rejected(tmp_path):
    rows = chunks()
    rows[0]["split"] = "train"
    report, _ = validator.validate_dataset(*write_inputs(tmp_path, rows))
    assert "partial_splits" in error_codes(report)


def test_bad_json_and_repeated_keys_are_reported_with_lines(tmp_path):
    a, c = write_inputs(tmp_path)
    with c.open("a", encoding="utf-8") as handle:
        handle.write('not json\n{"labels":{"convoys":1,"convoys":0}}\n[]\n')
    report, _ = validator.validate_dataset(a, c)
    assert report["input_rows"] == 5 and report["n_chunks"] == 2
    assert {e["line"] for e in report["errors"] if e["code"] == "chunk_json"} == {3, 4}
    assert "chunk_schema" in error_codes(report)


@pytest.mark.parametrize("problem", ["duplicate", "reserved_id", "empty_definition", "wrong_schema", "empty"])
def test_invalid_anchor_files(tmp_path, problem):
    anchors = anchor_file()
    if problem == "duplicate":
        anchors["anchors"].append(anchors["anchors"][0])
    elif problem == "reserved_id":
        anchors["anchors"][0]["id"] = "id"
    elif problem == "empty_definition":
        anchors["anchors"][0]["definition"] = " "
    elif problem == "wrong_schema":
        anchors["schema_version"] = True
    else:
        anchors["anchors"] = []
    report, _ = validator.validate_dataset(*write_inputs(tmp_path, anchors=anchors))
    assert not report["valid"] and not report["ready"]


def test_missing_files_and_empty_collection(tmp_path):
    report, coverage = validator.validate_dataset(tmp_path / "missing-a", tmp_path / "missing-c")
    assert {"anchors_input", "chunks_input", "chunks_empty"} <= error_codes(report)
    assert not coverage and not report["ready"]
    report, _ = validator.validate_dataset(*write_inputs(tmp_path, rows=[]))
    assert "chunks_empty" in error_codes(report)


def test_missing_tokenizer_data_has_actionable_error(tmp_path, monkeypatch):
    def missing(*args, **kwargs):
        raise LookupError("no tokenizer data")
    monkeypatch.setattr(validator, "sent_tokenize", missing)
    report, _ = validator.validate_dataset(*write_inputs(tmp_path))
    errors = [e for e in report["errors"] if e["code"] == "tokenizer_data"]
    assert len(errors) == 1 and "punkt_tab" in errors[0]["message"]


def test_cli_writes_reports_and_controls_exit_status(tmp_path):
    a, c = write_inputs(tmp_path)
    out = tmp_path / "reports"
    args = ["--anchors", str(a), "--chunks", str(c), "--out", str(out)]
    assert validator.main(args) == 0  # Under target, but valid during collection.
    report = json.loads((out / "validation_report.json").read_text())
    assert report["ready_for_label_export"] and not report["ready"]
    with (out / "coverage.csv").open(newline="") as handle:
        table = list(csv.DictReader(handle))
    assert table[0]["positive_shortfall"] == "74"
    assert validator.main(args + ["--require-ready"]) == 1
    assert validator.main(args + ["--require-ready", "--min-positive", "1"]) == 0


def test_cli_invalid_data_and_arguments(tmp_path):
    a, c = write_inputs(tmp_path, rows=[])
    args = ["--anchors", str(a), "--chunks", str(c), "--out", str(tmp_path / "reports")]
    assert validator.main(args) == 1
    with pytest.raises(SystemExit) as exc:
        validator.main(args + ["--min-positive", "0"])
    assert exc.value.code == 2


def test_cli_never_overwrites_an_input_with_a_report(tmp_path):
    a, c = write_inputs(tmp_path)
    c = c.rename(tmp_path / "validation_report.json")
    before = c.read_bytes()
    with pytest.raises(SystemExit) as exc:
        validator.main(["--anchors", str(a), "--chunks", str(c), "--out", str(tmp_path)])
    assert exc.value.code == 2 and c.read_bytes() == before
