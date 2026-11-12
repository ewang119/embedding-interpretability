"""Tests for probe/baseline.py. Run: python -m pytest tests -q"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from probe.baseline import compare, main, run_baseline
from probe.data import stratified_split

WORDS = {
    "siege": "siege encircled leningrad stalingrad starvation garrison",
    "atlantic": "uboat convoy atlantic merchant escort torpedo",
    "munich": "munich chamberlain sudeten czechoslovakia daladier conference",
}
FILLER = "the war army men year front report history said many".split()


def make_corpus(n=300, seed=0):
    rng = np.random.default_rng(seed)
    names = list(WORDS)
    Y = (rng.random((n, 3)) < 0.25).astype(int)
    texts = []
    for row in Y:
        words = list(rng.choice(FILLER, 8))
        for c, on in enumerate(row):
            if on:
                words += list(rng.choice(WORDS[names[c]].split(), 3))
        rng.shuffle(words)
        texts.append(" ".join(words))
    return texts, Y, names


def test_finds_keyword_anchors():
    texts, Y, names = make_corpus()
    table, summ, info = run_baseline(texts, Y, names)
    assert summ["macro_avg_precision"] > 0.95
    assert info["dropped"] == []
    assert set(table.index) == set(names)


def test_uses_same_test_chunks_as_probe_run():
    _, Y, _ = make_corpus()
    texts, _, names = make_corpus()
    _, _, info = run_baseline(texts, Y, names, seed=3)
    tr, va, te = stratified_split(Y, (0.7, 0.1, 0.2), seed=3)
    assert info["split_sizes"] == [len(tr), len(va), len(te)]


def test_drops_concepts_with_too_few_test_positives():
    texts, Y, names = make_corpus()
    Y[:, 2] = 0
    Y[:3, 2] = 1
    _, _, info = run_baseline(texts, Y, names, min_test_pos=5)
    assert info["dropped"] == ["munich"]


def test_compare_marks_winner():
    bow = pd.DataFrame({"f1": [0.5, 0.9], "avg_precision": [0.50, 0.90]}, index=["a", "b"])
    probe = pd.DataFrame({"f1": [0.8, 0.9], "avg_precision": [0.80, 0.905]}, index=["a", "b"])
    c = compare(bow, probe)
    assert c.loc["a", "winner"] == "probe"
    assert c.loc["b", "winner"] == "tie"


def test_cli_matches_rows_by_id_and_compares(tmp_path):
    texts, Y, names = make_corpus()
    ids = [f"c{i}" for i in range(len(texts))]
    pd.DataFrame({"id": ids[::-1], "text": texts[::-1]}).to_csv(tmp_path / "t.csv", index=False)  # shuffled order
    lab = pd.DataFrame(Y, columns=names)
    lab.insert(0, "id", ids)
    lab.to_csv(tmp_path / "l.csv", index=False)
    pd.DataFrame({"f1": 1.0, "avg_precision": 1.0}, index=names).to_csv(tmp_path / "probe.csv")
    res = main(["--texts", str(tmp_path / "t.csv"), "--labels", str(tmp_path / "l.csv"),
                "--out", str(tmp_path / "out"), "--compare", str(tmp_path / "probe.csv")])
    assert res["summary"]["macro_avg_precision"] > 0.95      # would fail if rows were mismatched
    assert (tmp_path / "out" / "compare.csv").exists()


def test_cli_stops_when_text_is_missing(tmp_path):
    pd.DataFrame({"id": ["a"], "text": ["x"]}).to_csv(tmp_path / "t.csv", index=False)
    pd.DataFrame({"id": ["a", "b"], "k": [1, 0]}).to_csv(tmp_path / "l.csv", index=False)
    with pytest.raises(SystemExit):
        main(["--texts", str(tmp_path / "t.csv"), "--labels", str(tmp_path / "l.csv"), "--out", str(tmp_path / "o")])


def test_split_from_reuses_probe_run_split(tmp_path):
    """--split-from must score the baseline on exactly probe.run's test chunks."""
    import json
    texts, Y, names = make_corpus()
    ids = [f"c{i}" for i in range(len(texts))]
    pd.DataFrame({"id": ids, "text": texts}).to_csv(tmp_path / "t.csv", index=False)
    lab = pd.DataFrame(Y, columns=names)
    lab.insert(0, "id", ids)
    lab.to_csv(tmp_path / "l.csv", index=False)
    rng = np.random.default_rng(5)
    order = rng.permutation(len(ids))
    split = {"train": [ids[i] for i in order[:200]], "val": [ids[i] for i in order[200:240]],
             "test": [ids[i] for i in order[240:]]}
    (tmp_path / "split.json").write_text(json.dumps(split))
    res = main(["--texts", str(tmp_path / "t.csv"), "--labels", str(tmp_path / "l.csv"),
                "--split-from", str(tmp_path / "split.json"), "--out", str(tmp_path / "out")])
    assert res["split_sizes"] == [200, 40, len(ids) - 240]
    table = pd.read_csv(tmp_path / "out" / "per_concept.csv", index_col=0)
    test_rows = [ids.index(i) for i in split["test"]]
    assert table.loc["siege", "support"] == Y[test_rows, 0].sum()


def test_split_from_rejects_unknown_ids(tmp_path):
    import json
    texts, Y, names = make_corpus(n=50)
    ids = [f"c{i}" for i in range(50)]
    pd.DataFrame({"id": ids, "text": texts}).to_csv(tmp_path / "t.csv", index=False)
    lab = pd.DataFrame(Y, columns=names)
    lab.insert(0, "id", ids)
    lab.to_csv(tmp_path / "l.csv", index=False)
    (tmp_path / "s.json").write_text(json.dumps({"train": ["nope"], "val": [], "test": []}))
    with pytest.raises(SystemExit, match="split-from"):
        main(["--texts", str(tmp_path / "t.csv"), "--labels", str(tmp_path / "l.csv"),
              "--split-from", str(tmp_path / "s.json"), "--out", str(tmp_path / "out")])
