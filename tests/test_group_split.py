"""Splitting with source groups (WWII passages from the same article stay
together). Run: python -m pytest tests -q"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from probe.data import split_from_meta, stratified_split
from probe.run import main as run_main


def grouped_labels(n_sources=120, per_source=(1, 6), D=4, seed=0):
    """Chunks from sources of varying size; chunks of one source tend to share labels."""
    rng = np.random.default_rng(seed)
    groups, rows = [], []
    for s in range(n_sources):
        k = rng.integers(*per_source)
        topic = rng.random(D) < 0.25
        for _ in range(k):
            groups.append(f"src-{s}")
            rows.append(topic & (rng.random(D) < 0.8))
    return np.array(rows, dtype=int), np.array(groups)


def test_no_source_crosses_splits():
    Y, g = grouped_labels()
    parts = stratified_split(Y, (0.7, 0.1, 0.2), seed=0, groups=g)
    where = {}
    for s, idx in enumerate(parts):
        for i in idx:
            assert where.setdefault(g[i], s) == s, f"{g[i]} is in two splits"


def test_grouped_split_covers_every_chunk_once_and_keeps_proportions():
    Y, g = grouped_labels(n_sources=300)
    tr, va, te = stratified_split(Y, (0.7, 0.1, 0.2), seed=1, groups=g)
    assert sorted(np.concatenate([tr, va, te]).tolist()) == list(range(len(Y)))
    assert abs(len(te) / len(Y) - 0.2) < 0.05
    share = Y[te].sum(0) / Y.sum(0)
    assert np.all(np.abs(share - 0.2) < 0.08), share


def test_one_chunk_per_group_matches_ungrouped_split():
    Y, _ = grouped_labels()
    a = stratified_split(Y, seed=4)
    b = stratified_split(Y, seed=4, groups=[f"chunk{i}" for i in range(len(Y))])
    assert all(np.array_equal(x, y) for x, y in zip(a, b))


def test_group_key_spelling_does_not_change_split():
    Y, g = grouped_labels()
    renamed = np.array([f"zz-{x[::-1]}" for x in g])
    a = stratified_split(Y, seed=2, groups=g)
    b = stratified_split(Y, seed=2, groups=renamed)
    assert all(np.array_equal(x, y) for x, y in zip(a, b))


def test_wrong_number_of_groups_rejected():
    with pytest.raises(ValueError):
        stratified_split(np.zeros((5, 2)), groups=["a", "b"])


# ---------------------------------------------------------------------------
# split_from_meta
# ---------------------------------------------------------------------------

def meta_for(ids, groups=None, split=None):
    m = pd.DataFrame({"id": ids})
    if groups is not None:
        m["source_id"] = groups
    if split is not None:
        m["split"] = split
    return m


def test_meta_with_full_split_column_is_used_as_given():
    ids = [f"c{i}" for i in range(6)]
    split = ["train", "train", "train", "val", "test", "test"]
    parts, mode = split_from_meta(meta_for(ids[::-1], split=split[::-1]), ids, np.zeros((6, 1)))
    assert mode == "given in meta"
    assert [p.tolist() for p in parts] == [[0, 1, 2], [3], [4, 5]]


def test_meta_split_rejects_source_in_two_splits():
    ids = ["a", "b", "c", "d"]
    m = meta_for(ids, groups=["s1", "s1", "s2", "s3"], split=["train", "test", "val", "test"])
    with pytest.raises(ValueError, match="more than one split"):
        split_from_meta(m, ids, np.zeros((4, 1)))


def test_meta_rejects_unknown_split_name_and_missing_ids():
    ids = ["a", "b", "c"]
    with pytest.raises(ValueError, match="train/val/test"):
        split_from_meta(meta_for(ids, split=["train", "dev", "test"]), ids, np.zeros((3, 1)))
    with pytest.raises(ValueError, match="missing from meta"):
        split_from_meta(meta_for(["a", "b"]), ids, np.zeros((3, 1)))


def test_meta_without_split_groups_by_source():
    Y, g = grouped_labels()
    ids = [f"c{i}" for i in range(len(Y))]
    parts, mode = split_from_meta(meta_for(ids, groups=g, split=[None] * len(Y)), ids, Y)
    assert mode == "stratified, sources kept whole"
    assert [p.tolist() for p in parts] == [p.tolist() for p in stratified_split(Y, groups=g)]


# ---------------------------------------------------------------------------
# probe.run --meta
# ---------------------------------------------------------------------------

def test_run_keeps_sources_together(tmp_path):
    Y, g = grouped_labels(n_sources=200)
    rng = np.random.default_rng(0)
    X = rng.normal(size=(len(Y), 8)).astype(np.float32)
    X[:, :Y.shape[1]] += 2.0 * Y
    ids = [f"c{i}" for i in range(len(Y))]
    np.save(tmp_path / "e.npy", X)
    lab = pd.DataFrame(Y, columns=[f"k{j}" for j in range(Y.shape[1])])
    lab.insert(0, "id", ids)
    lab.to_csv(tmp_path / "l.csv", index=False)
    meta_for(ids, groups=g).to_csv(tmp_path / "m.csv", index=False)
    res = run_main(["--embeddings", str(tmp_path / "e.npy"), "--labels", str(tmp_path / "l.csv"),
                    "--meta", str(tmp_path / "m.csv"), "--out", str(tmp_path / "out"),
                    "--ablation", "none", "--n-resamples", "0", "--ks", "2"])
    assert res["split_mode"] == "stratified, sources kept whole"
    split = json.loads((tmp_path / "out" / "split.json").read_text())
    src = dict(zip(ids, g))
    train_sources = {src[i] for i in split["train"]}
    assert not train_sources & {src[i] for i in split["test"]}


def test_run_reports_bad_meta_clearly(tmp_path):
    np.save(tmp_path / "e.npy", np.zeros((3, 2), dtype=np.float32))
    pd.DataFrame({"id": ["a", "b", "c"], "k": [1, 0, 1]}).to_csv(tmp_path / "l.csv", index=False)
    pd.DataFrame({"id": ["a", "b"]}).to_csv(tmp_path / "m.csv", index=False)
    with pytest.raises(SystemExit, match="--meta"):
        run_main(["--embeddings", str(tmp_path / "e.npy"), "--labels", str(tmp_path / "l.csv"),
                  "--meta", str(tmp_path / "m.csv"), "--out", str(tmp_path / "out")])
