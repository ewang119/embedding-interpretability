"""Unit tests for each piece of the probe pipeline.

test_synthetic.py checks the whole pipeline end to end. These check the
parts one at a time, so a bug in (say) the split or the ablation shows up
as a failing test with that part's name on it.

Run: python -m pytest tests -q
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import torch

from probe import ablation, positions
from probe.data import ProbeData, Standardizer, label_report, load, stratified_split
from probe.evaluate import average_precision
from probe.model import LinearProbe, MLPProbe, build
from probe.run import main as run_main
from probe.train import TrainConfig, pos_weights, predict_proba, train_probe


def toy(n=300, d=8, D=3, seed=0, prevalence=0.2):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, d)).astype(np.float32)
    Y = (rng.random((n, D)) < prevalence).astype(np.float32)
    X[:, :D] += 2.0 * Y          # concept c shifts position c
    return X, Y


# ---------------------------------------------------------------------------
# data.py
# ---------------------------------------------------------------------------

class TestProbeData:
    def test_rejects_row_mismatch(self):
        with pytest.raises(ValueError, match="rows"):
            ProbeData(X=np.zeros((5, 3)), Y=np.zeros((4, 2)), concepts=["a", "b"])

    def test_rejects_non_binary_labels(self):
        with pytest.raises(ValueError, match="0/1"):
            ProbeData(X=np.zeros((2, 3)), Y=np.array([[0, 2], [1, 0]]), concepts=["a", "b"])

    def test_rejects_wrong_number_of_concept_names(self):
        with pytest.raises(ValueError, match="concept names"):
            ProbeData(X=np.zeros((2, 3)), Y=np.zeros((2, 2)), concepts=["a"])

    def test_default_ids_and_shapes(self):
        p = ProbeData(X=np.zeros((4, 3)), Y=np.zeros((4, 2)), concepts=["a", "b"])
        assert p.ids == ["0", "1", "2", "3"]
        assert (p.n, p.d, p.n_concepts) == (4, 3, 2)
        assert p.X.dtype == np.float32


def test_load_reads_id_column_and_keeps_row_order(tmp_path):
    X, Y = toy(n=10)
    np.save(tmp_path / "e.npy", X)
    df = pd.DataFrame(Y.astype(int), columns=["x", "y", "z"])
    df.insert(0, "id", [f"doc{i}" for i in range(10)])
    df.to_csv(tmp_path / "l.csv", index=False)
    p = load(tmp_path / "e.npy", tmp_path / "l.csv")
    assert p.concepts == ["x", "y", "z"]
    assert p.ids[3] == "doc3"
    np.testing.assert_array_equal(p.Y, Y)
    np.testing.assert_allclose(p.X, X)


class TestStratifiedSplit:
    def test_disjoint_and_complete(self):
        _, Y = toy(n=500, D=6)
        parts = stratified_split(Y, (0.7, 0.1, 0.2), seed=1)
        joined = np.concatenate(parts)
        assert len(joined) == 500 and len(set(joined)) == 500

    def test_sizes_follow_fractions(self):
        _, Y = toy(n=1000, D=4)
        tr, va, te = stratified_split(Y, (0.7, 0.1, 0.2), seed=0)
        assert abs(len(tr) - 700) <= 5 and abs(len(va) - 100) <= 5 and abs(len(te) - 200) <= 5

    def test_same_seed_same_split_different_seed_different_split(self):
        _, Y = toy(n=300)
        a, b, c = (stratified_split(Y, seed=s) for s in (3, 3, 4))
        assert all(np.array_equal(x, y) for x, y in zip(a, b))
        assert not np.array_equal(a[2], c[2])

    def test_rare_concept_still_reaches_test(self):
        Y = np.zeros((400, 2))
        Y[:5, 0] = 1               # only 5 positives
        Y[100:200, 1] = 1
        _, _, te = stratified_split(Y, seed=0)
        assert Y[te, 0].sum() >= 1

    def test_label_report_counts(self):
        Y = np.array([[1, 0], [1, 1], [0, 1], [0, 0]])
        r = label_report(Y, [np.array([0, 1]), np.array([2]), np.array([3])], ["a", "b"])
        assert r.loc["a"].tolist() == [2, 0, 0]
        assert r.loc["b"].tolist() == [1, 1, 0]


class TestStandardizer:
    def test_train_split_becomes_mean0_std1(self):
        X = np.random.default_rng(0).normal(5, 3, size=(200, 4)).astype(np.float32)
        Z = Standardizer.fit(X).transform(X)
        np.testing.assert_allclose(Z.mean(0), 0, atol=1e-5)
        np.testing.assert_allclose(Z.std(0), 1, atol=1e-4)

    def test_inverse_round_trip(self):
        X = np.random.default_rng(1).normal(size=(50, 6)).astype(np.float32)
        s = Standardizer.fit(X)
        np.testing.assert_allclose(s.inverse(s.transform(X)), X, atol=1e-5)

    def test_constant_position_does_not_divide_by_zero(self):
        X = np.ones((20, 3), dtype=np.float32)
        Z = Standardizer.fit(X).transform(X)
        assert np.isfinite(Z).all()


# ---------------------------------------------------------------------------
# ablation.py
# ---------------------------------------------------------------------------

def test_ablate_zeros_only_chosen_positions_and_copies():
    Z = np.ones((4, 5), dtype=np.float32)
    out = ablation.ablate(Z, [1, 3])
    assert (out[:, [1, 3]] == 0).all() and (out[:, [0, 2, 4]] == 1).all()
    assert (Z == 1).all(), "input must not be modified"


def test_raw_and_standardized_ablation_agree():
    """The reconstruction step ablates raw vectors; the probe ablates
    standardized ones. Both must remove exactly the same thing."""
    X = np.random.default_rng(2).normal(3, 2, size=(30, 6)).astype(np.float32)
    s = Standardizer.fit(X)
    pos = [0, 4]
    via_raw = s.transform(ablation.ablate_raw(X, pos, s.mean))
    via_std = ablation.ablate(s.transform(X), pos)
    np.testing.assert_allclose(via_raw, via_std, atol=1e-5)


def test_empirical_p_bounds():
    rand = np.array([0.1, 0.2, 0.3, 0.4])
    assert ablation._empirical_p(1.0, rand) == pytest.approx(1 / 5)   # beats every draw
    assert ablation._empirical_p(0.0, rand) == pytest.approx(1.0)     # beaten by every draw


def test_specificity_diagonal_dominates_when_concepts_are_separate():
    X, Y = toy(n=600, d=8, D=3)
    s = Standardizer.fit(X)
    Z = s.transform(X)
    m, _ = train_probe(Z[:400], Y[:400], Z[400:500], Y[400:500], TrainConfig(max_epochs=500))
    base = average_precision(predict_proba(m, Z[500:]), Y[500:])
    spec = ablation.specificity(m, Z[500:], Y[500:], positions.top_positions(m.weight_matrix(), 1), base)
    assert spec.shape == (3, 3)
    off = spec[~np.eye(3, dtype=bool)]
    assert np.diag(spec).min() > 5 * np.abs(off).max()


# ---------------------------------------------------------------------------
# positions.py
# ---------------------------------------------------------------------------

def test_top_positions_by_absolute_weight():
    W = np.array([[0.1, -5.0, 2.0, 0.0]])
    assert positions.top_positions(W, 2).tolist() == [[1, 2]]


def test_concentration_extremes():
    one_hot = np.array([[0.0, 3.0, 0.0, 0.0]])
    even = np.ones((1, 8))
    assert positions.concentration(one_hot, 1)[0] == pytest.approx(1.0)
    assert positions.concentration(even, 2)[0] == pytest.approx(2 / 8)


def test_stability_identical_runs_is_one():
    W = np.random.default_rng(0).normal(size=(3, 20))
    np.testing.assert_allclose(positions.stability([W, W.copy(), W.copy()], 5), 1.0)


def test_random_jaccard_simple_case():
    # two random 1-of-2 picks overlap fully half the time -> expected Jaccard ~ 1/3 by formula
    assert positions.random_jaccard(2, 1) == pytest.approx((1 / 2) / (2 - 1 / 2))


# ---------------------------------------------------------------------------
# train.py / model.py
# ---------------------------------------------------------------------------

def test_pos_weights_ratio_and_cap():
    Y = np.zeros((100, 3))
    Y[:10, 0] = 1           # 90 / 10 = 9
    Y[:1, 1] = 1            # 99 / 1 -> capped
    w = pos_weights(Y, cap=50)  # column 2 has no positives -> treated as 1 positive, capped
    assert w[0] == pytest.approx(9.0)
    assert w[1] == pytest.approx(50.0)
    assert w[2] == pytest.approx(50.0)
    assert (w >= 1).all()


def test_training_is_deterministic_for_a_seed():
    X, Y = toy()
    cfg = TrainConfig(max_epochs=200, seed=7)
    a, _ = train_probe(X[:200], Y[:200], X[200:], Y[200:], cfg)
    b, _ = train_probe(X[:200], Y[:200], X[200:], Y[200:], cfg)
    np.testing.assert_allclose(a.weight_matrix(), b.weight_matrix())


def test_early_stopping_keeps_best_epoch():
    X, Y = toy()
    _, h = train_probe(X[:200], Y[:200], X[200:], Y[200:], TrainConfig(max_epochs=3000, patience=50))
    assert h["best_epoch"] < h["epochs_run"] <= 3000
    # a new best needs to beat the old one by > 1e-6, so allow that much slack
    assert h["best_val_loss"] == pytest.approx(min(h["val_loss"]), abs=1e-5)


def test_probe_learns_the_planted_position():
    X, Y = toy(n=600)
    m, _ = train_probe(X[:450], Y[:450], X[450:], Y[450:], TrainConfig())
    W = m.weight_matrix()
    assert positions.top_positions(W, 1).ravel().tolist() == [0, 1, 2]
    assert np.nanmin(average_precision(predict_proba(m, X[450:]), Y[450:])) > 0.7


def test_model_shapes_and_mlp_has_no_position_weights():
    lin, mlp = build("linear", 8, 3), build("mlp", 8, 3, hidden=16)
    assert isinstance(lin, LinearProbe) and isinstance(mlp, MLPProbe)
    x = torch.zeros(5, 8)
    assert lin(x).shape == (5, 3) and mlp(x).shape == (5, 3)
    assert lin.weight_matrix().shape == (3, 8)
    with pytest.raises(NotImplementedError):
        mlp.weight_matrix()
    with pytest.raises(ValueError):
        build("transformer", 8, 3)


# ---------------------------------------------------------------------------
# run.py guards
# ---------------------------------------------------------------------------

def _write(tmp_path, X, Y, names):
    np.save(tmp_path / "e.npy", X)
    pd.DataFrame(Y.astype(int), columns=names).to_csv(tmp_path / "l.csv", index=False)
    return ["--embeddings", str(tmp_path / "e.npy"), "--labels", str(tmp_path / "l.csv"),
            "--out", str(tmp_path / "out"), "--ablation", "none", "--n-resamples", "0", "--ks", "2"]


def test_run_drops_concepts_with_too_few_test_positives(tmp_path, capsys):
    X, Y = toy(n=400, D=3)
    Y[:, 2] = 0
    Y[:6, 2] = 1                # ~1 test positive: should be dropped
    res = run_main(_write(tmp_path, X, Y, ["a", "b", "rare"]))
    assert res["n_concepts"] == 2
    assert "rare" in capsys.readouterr().err
    assert "rare" not in pd.read_csv(tmp_path / "out" / "per_concept.csv", index_col=0).index


def test_run_stops_when_no_concept_is_usable(tmp_path):
    X, _ = toy(n=100, D=2)
    Y = np.zeros((100, 2))
    Y[0, :] = 1
    with pytest.raises(SystemExit):
        run_main(_write(tmp_path, X, Y, ["a", "b"]))
