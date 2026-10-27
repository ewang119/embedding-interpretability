"""End-to-end check on planted data. Run: python -m pytest tests -q"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd

from probe import positions
from probe.data import stratified_split
from probe.run import main
from tests.synthetic import make_synthetic


def _write(tmp_path, **kw):
    X, Y, concepts, planted = make_synthetic(**kw)
    np.save(tmp_path / "emb.npy", X)
    pd.DataFrame(Y.astype(int), columns=concepts).to_csv(tmp_path / "labels.csv", index=False)
    return concepts, planted


def test_split_keeps_rare_concepts_in_every_split():
    rng = np.random.default_rng(0)
    Y = np.zeros((1000, 5))
    for c, n_pos in enumerate([10, 20, 40, 80, 160]):
        Y[rng.choice(1000, n_pos, replace=False), c] = 1
    tr, va, te = stratified_split(Y, (0.7, 0.1, 0.2), seed=0)
    assert len(tr) + len(va) + len(te) == 1000
    assert not (set(tr) & set(te)) and not (set(tr) & set(va)) and not (set(va) & set(te))
    for c in range(5):
        assert abs(Y[te, c].sum() - 0.2 * Y[:, c].sum()) <= 1
        assert Y[va, c].sum() >= 1


def test_pipeline_recovers_planted_positions(tmp_path):
    concepts, planted = _write(tmp_path)
    res = main(["--embeddings", str(tmp_path / "emb.npy"), "--labels", str(tmp_path / "labels.csv"),
                "--out", str(tmp_path / "out"), "--ks", "3", "6", "--n-random", "10", "--n-resamples", "3"])
    out = tmp_path / "out"
    table = pd.read_csv(out / "per_concept.csv", index_col=0)
    pos = json.loads((out / "positions.json").read_text())

    # probe works, shuffled-label control does not
    # planted signal is ~2.6 SD of separation, so even a perfect probe tops out near 0.9 AP
    assert res["summary"]["macro_avg_precision"] > 0.8
    assert res["summary"]["control_macro_avg_precision"] < 0.4

    # local concepts: top-3 positions are exactly the planted ones, and stable
    for c, name in enumerate(concepts[:-1]):
        assert set(pos[name]["top_positions"][:3]) == set(planted[c][0])
        assert table.loc[name, "stability_top3"] > 0.9
    # redundant concept: top-6 covers both planted groups
    assert set(pos["redundant"]["top_positions"][:6]) == set(sum(planted[len(concepts) - 1], []))

    abl = pd.read_csv(out / "ablation.csv").set_index(["mode", "concept", "k"])
    for name in concepts[:-1]:
        # removing the planted positions hurts far more than random, in both modes
        for mode in ("frozen", "retrain"):
            row = abl.loc[(mode, name, 3)]
            assert row.top_drop > row.random_drop_mean + 3 * row.random_drop_std
            assert row.p_value < 0.1

    # redundant concept: its information sits in 6 positions, so ablating its
    # top 3 barely matters (the other copy still carries it), while ablating
    # all 6 removes it. This is the "spread out vs concentrated" signal: a
    # small drop at small k is not proof the concept isn't in those positions.
    for mode in ("frozen", "retrain"):
        half, full = abl.loc[(mode, "redundant", 3)].top_drop, abl.loc[(mode, "redundant", 6)].top_drop
        assert full > 0.3
        assert half < 0.4 * full

    spec = np.load(out / "specificity_top3.npy")
    off_diag = spec[~np.eye(len(spec), dtype=bool)]
    assert np.all(np.diag(spec)[:-1] > 5 * np.abs(off_diag).max())


def test_random_jaccard_matches_simulation():
    rng = np.random.default_rng(0)
    d, k = 768, 50
    sims = [positions.jaccard(rng.choice(d, k, replace=False), rng.choice(d, k, replace=False)) for _ in range(2000)]
    assert abs(np.mean(sims) - positions.random_jaccard(d, k)) < 0.01
