"""Tests for probe/evaluate.py: thresholds, average precision, and the
per-concept scoring table.

Run: python -m pytest tests -q
"""

from __future__ import annotations

import numpy as np
import pytest

from probe.evaluate import average_precision, per_concept, summary, tune_thresholds


# ---------------------------------------------------------------------------

class TestThresholds:
    def test_threshold_separates_a_clean_split(self):
        probs = np.array([[0.9], [0.8], [0.3], [0.2]])
        Y = np.array([[1], [1], [0], [0]])
        t = tune_thresholds(probs, Y)
        assert 0.3 < t[0] < 0.8
        assert ((probs >= t) == Y).all()

    def test_concept_without_positives_keeps_default(self):
        t = tune_thresholds(np.array([[0.9], [0.1]]), np.array([[0], [0]]), default=0.5)
        assert t[0] == 0.5


def test_average_precision_perfect_and_undefined():
    probs = np.array([[0.9, 0.4], [0.1, 0.6]])
    Y = np.array([[1, 0], [0, 0]])
    ap = average_precision(probs, Y)
    assert ap[0] == pytest.approx(1.0)
    assert np.isnan(ap[1]), "no positives -> undefined, not 0"


def test_scores_match_sklearn_with_several_concepts():
    from sklearn.metrics import f1_score
    rng = np.random.default_rng(0)
    Y = (rng.random((200, 4)) < 0.3).astype(int)
    probs = np.clip(Y * 0.4 + rng.random((200, 4)) * 0.6, 0, 1)
    t = per_concept(probs, Y, np.full(4, 0.5), list("abcd"))
    pred = (probs >= 0.5).astype(int)
    np.testing.assert_allclose(t["f1"], f1_score(Y, pred, average=None))
    assert summary(t, probs, Y)["micro_f1"] == pytest.approx(f1_score(Y, pred, average="micro"))


def test_single_concept_is_scored_as_one_concept():
    """Regression: with one concept, sklearn scored "yes" and "no" separately."""
    probs = np.array([[0.9], [0.2], [0.7], [0.1]])
    Y = np.array([[1], [0], [0], [0]])
    t = per_concept(probs, Y, np.array([0.5]), ["c"])
    assert len(t) == 1
    assert t.loc["c", "precision"] == pytest.approx(0.5)   # 1 of 2 predicted yes is right
    assert summary(t, probs, Y)["micro_f1"] == pytest.approx(2 / 3)   # not accuracy (0.75)


def test_per_concept_table_columns():
    probs = np.array([[0.9], [0.2], [0.7], [0.1]])
    Y = np.array([[1], [0], [1], [0]])
    t = per_concept(probs, Y, np.array([0.5]), ["c"])
    assert t.loc["c", "support"] == 2
    assert t.loc["c", "f1"] == pytest.approx(1.0)
    assert t.loc["c", "prevalence"] == pytest.approx(0.5)
