"""Ablation: does removing a concept's top-weighted positions hurt that concept
more than removing the same number of random positions?

"Remove" = set the position to 0 in standardized space, i.e. to its training
mean. (For the reconstruction experiment the inverter needs raw vectors; use
`ablate_raw`, which does the same thing in raw space.)

Two modes, and they answer different questions:

frozen   Keep the trained probe, ablate the test embeddings, re-score.
         Cheap. Partly circular for a linear probe: zeroing the inputs with
         the biggest weights will hurt that probe more than random ones
         almost by construction. Treat it as a sanity check.

retrain  Ablate train/val/test, train a fresh probe, re-score. The fresh probe
         can reweight whatever positions are left, so a drop here means the
         rest of the vector really lacks the information. This is the number
         to report as "how much of the concept lives there".

The two modes mostly agree when positions are uncorrelated and the probe is
L2-regularized. They come apart when positions are correlated with each other
(real embeddings: likely) or when the probe is sparse (L1), because then the
frozen probe leaned on positions a retrained probe can route around.

Scores are average precision (threshold-free); see evaluate.py for why.
The random control is drawn many times and reported as mean, std and an
empirical p-value: the share of random draws that hurt at least as much as
the top-k positions did.
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pandas as pd

from .evaluate import average_precision
from .train import TrainConfig, predict_proba, train_probe


def ablate(Z: np.ndarray, positions) -> np.ndarray:
    Z = Z.copy()
    Z[:, np.asarray(positions, dtype=int)] = 0.0
    return Z


def ablate_raw(X: np.ndarray, positions, train_mean: np.ndarray) -> np.ndarray:
    """Same ablation for raw (unstandardized) vectors, e.g. before feeding an inverter."""
    X = X.copy()
    pos = np.asarray(positions, dtype=int)
    X[:, pos] = train_mean[pos]
    return X


def _empirical_p(top_drop: float, random_drops: np.ndarray) -> float:
    return float((1 + np.sum(random_drops >= top_drop)) / (1 + len(random_drops)))


def run(splits, top: dict[int, np.ndarray], base_ap: np.ndarray, concepts, cfg: TrainConfig,
        mode: str = "retrain", model=None, n_random: int = 20, seed: int = 0) -> pd.DataFrame:
    """Ablation table, one row per (concept, k).

    splits : (Ztr, Ytr, Zva, Yva, Zte, Yte), already standardized
    top    : {k: (D, k) array of top positions per concept}
    base_ap: (D,) test average precision of the unablated probe (same mode)
    model  : the trained probe, needed for mode="frozen"
    """
    Ztr, Ytr, Zva, Yva, Zte, Yte = splits
    rng = np.random.default_rng(seed)
    d, D = Ztr.shape[1], Ytr.shape[1]
    rows = []

    for k, top_k in top.items():
        # Random control. Random positions don't depend on the concept, so one
        # draw can be scored for every concept at once (one retrain per draw).
        rand_ap = np.empty((n_random, D))
        for r in range(n_random):
            pos = rng.choice(d, size=k, replace=False)
            rand_ap[r] = _score(splits, pos, mode, model, cfg, cols=None)

        for c in range(D):
            if np.isnan(base_ap[c]):
                continue
            ap = _score(splits, top_k[c], mode, model, cfg, cols=[c])[0]
            top_drop = base_ap[c] - ap
            rand_drops = base_ap[c] - rand_ap[:, c]
            rows.append({
                "concept": concepts[c], "k": k, "mode": mode,
                "base_ap": base_ap[c], "top_ap": ap, "top_drop": top_drop,
                "random_drop_mean": rand_drops.mean(), "random_drop_std": rand_drops.std(),
                "p_value": _empirical_p(top_drop, rand_drops),
            })
    return pd.DataFrame(rows)


def _score(splits, positions, mode, model, cfg, cols):
    """Test average precision after ablating `positions`, for concepts `cols` (None = all)."""
    Ztr, Ytr, Zva, Yva, Zte, Yte = splits
    cols_ = slice(None) if cols is None else cols
    if mode == "frozen":
        probs = predict_proba(model, ablate(Zte, positions))[:, cols_]
    elif mode == "retrain":
        m, _ = train_probe(ablate(Ztr, positions), Ytr[:, cols_], ablate(Zva, positions), Yva[:, cols_],
                           replace(cfg, probe="linear"))
        probs = predict_proba(m, ablate(Zte, positions))
    else:
        raise ValueError(mode)
    return average_precision(probs, Yte[:, cols_])


def specificity(model, Zte, Yte, top_k: np.ndarray, base_ap: np.ndarray) -> np.ndarray:
    """(D, D) matrix: row c = drop in every concept's AP when concept c's top-k are ablated (frozen probe).

    A specific result has a large diagonal and small off-diagonal. Large
    off-diagonal entries mean concepts share positions, which is expected for
    semantically close concepts (e.g. soy-oil and soy-meal).
    """
    D = top_k.shape[0]
    out = np.full((D, D), np.nan)
    for c in range(D):
        out[c] = base_ap - average_precision(predict_proba(model, ablate(Zte, top_k[c])), Yte)
    return out
