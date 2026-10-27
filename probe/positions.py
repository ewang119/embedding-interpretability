"""Reading candidate positions out of the linear probe's weights.

This produces the hypothesis ("concept c lives in positions p1, p2, ...").
ablation.py is what tests it.

Weights are only comparable across positions because the inputs were
standardized first (see data.Standardizer).
"""

from __future__ import annotations

from itertools import combinations

import numpy as np


def top_positions(W: np.ndarray, k: int) -> np.ndarray:
    """(D, k) array: for each concept, the k positions with the largest |weight|, largest first."""
    return np.argsort(-np.abs(W), axis=1)[:, :k]


def concentration(W: np.ndarray, k: int) -> np.ndarray:
    """Share of each concept's squared weight that sits in its top-k positions.

    1.0 means the probe uses only those k positions; k/d means the weight is
    spread evenly. Correlated positions split weight between them, so a low
    number can mean "spread out" or "duplicated across positions"; ablation
    with retraining tells those two apart.
    """
    w2 = W ** 2
    top = np.sort(w2, axis=1)[:, ::-1][:, :k]
    return top.sum(1) / np.maximum(w2.sum(1), 1e-12)


def jaccard(a, b) -> float:
    a, b = set(map(int, a)), set(map(int, b))
    return len(a & b) / max(len(a | b), 1)


def stability(weight_mats: list[np.ndarray], k: int) -> np.ndarray:
    """Mean pairwise Jaccard overlap of each concept's top-k across resampled runs.

    If retraining on a slightly different sample of chunks gives a different
    top-k, the position list is noise and should not be reported as a finding.
    Two random top-k sets out of d positions overlap by about k / (2d - k).
    """
    tops = [top_positions(W, k) for W in weight_mats]
    pairs = list(combinations(range(len(tops)), 2))
    if not pairs:
        return np.full(weight_mats[0].shape[0], np.nan)
    return np.array(
        [np.mean([jaccard(tops[i][c], tops[j][c]) for i, j in pairs]) for c in range(tops[0].shape[0])]
    )


def random_jaccard(d: int, k: int) -> float:
    """Expected Jaccard overlap of two random k-subsets of d positions (baseline for stability)."""
    inter = k * k / d
    return inter / (2 * k - inter)
