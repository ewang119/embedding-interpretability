"""Metrics.

Two kinds of score per concept:
- F1 / precision / recall at a decision threshold. The threshold for each
  concept is tuned on the validation split, never on test.
- Average precision (area under the precision-recall curve). It needs no
  threshold, so it is the better number for ablation curves: an ablation can
  shift every score down a bit and wreck F1 at a fixed threshold even when the
  probe still ranks chunks correctly.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score


def tune_thresholds(probs: np.ndarray, Y: np.ndarray, default: float = 0.5) -> np.ndarray:
    """Per-concept threshold that maximizes F1 on the given (validation) split."""
    thresholds = np.full(Y.shape[1], default, dtype=np.float32)
    for c in range(Y.shape[1]):
        y = Y[:, c]
        if y.sum() == 0:
            continue
        order = np.argsort(-probs[:, c])
        p_sorted, y_sorted = probs[order, c], y[order]
        tp = np.cumsum(y_sorted)
        fp = np.cumsum(1 - y_sorted)
        f1 = 2 * tp / (tp + fp + y.sum())
        best = int(np.argmax(f1))
        nxt = p_sorted[best + 1] if best + 1 < len(p_sorted) else 0.0
        thresholds[c] = (p_sorted[best] + nxt) / 2   # midpoint between kept and dropped scores
    return thresholds


def average_precision(probs: np.ndarray, Y: np.ndarray) -> np.ndarray:
    ap = np.full(Y.shape[1], np.nan)
    for c in range(Y.shape[1]):
        if 0 < Y[:, c].sum() < len(Y):
            ap[c] = average_precision_score(Y[:, c], probs[:, c])
    return ap


def _counts(pred: np.ndarray, Y: np.ndarray):
    """True/false positives and false negatives per concept (column).

    Done by hand rather than with sklearn: with a single concept sklearn
    reads an (N, 1) label array as one binary problem and scores both the
    "yes" and the "no" class, which broke the table and turned micro F1
    into accuracy.
    """
    Y = np.asarray(Y) > 0
    pred = np.asarray(pred) > 0
    return (pred & Y).sum(0), (pred & ~Y).sum(0), (~pred & Y).sum(0)


def _f1(tp, fp, fn):
    tp, fp, fn = (np.asarray(v, dtype=float) for v in (tp, fp, fn))
    with np.errstate(divide="ignore", invalid="ignore"):
        p = np.where(tp + fp > 0, tp / (tp + fp), 0.0)
        r = np.where(tp + fn > 0, tp / (tp + fn), 0.0)
        f1 = np.where(p + r > 0, 2 * p * r / (p + r), 0.0)
    return p, r, f1


def per_concept(probs: np.ndarray, Y: np.ndarray, thresholds: np.ndarray, concepts) -> pd.DataFrame:
    tp, fp, fn = _counts(probs >= thresholds, Y)
    p, r, f1 = _f1(tp, fp, fn)
    support = tp + fn
    return pd.DataFrame(
        {
            "support": support.astype(int),
            "prevalence": Y.mean(0),
            "precision": p,
            "recall": r,
            "f1": f1,
            "avg_precision": average_precision(probs, Y),
            "threshold": thresholds,
        },
        index=list(concepts),
    )


def summary(table: pd.DataFrame, probs: np.ndarray, Y: np.ndarray) -> dict:
    tp, fp, fn = _counts(probs >= table["threshold"].to_numpy(), Y)
    _, _, micro_f1 = _f1(tp.sum(), fp.sum(), fn.sum())
    scored = table[table["support"] > 0]
    return {
        "macro_f1": float(scored["f1"].mean()),
        "micro_f1": float(micro_f1),
        "macro_avg_precision": float(scored["avg_precision"].mean()),
        # what a probe that ignores the embedding would get for average precision
        "chance_avg_precision": float(scored["prevalence"].mean()),
        "n_concepts_scored": int(len(scored)),
    }
