"""Data contract, multi-label stratified split, and standardization.

The training code only needs two things, so it does not depend on which
topics or which embedding model the team picks:

    X : float array (N, d)   one embedding per chunk, same model/settings for all
    Y : 0/1 array   (N, D)   one column per concept, from the labeler

Row i of X and row i of Y must be the same chunk.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd


@dataclass
class ProbeData:
    X: np.ndarray                 # (N, d) float32
    Y: np.ndarray                 # (N, D) float32 in {0, 1}
    concepts: list[str]
    ids: list[str] = field(default_factory=list)

    def __post_init__(self):
        self.X = np.asarray(self.X, dtype=np.float32)
        self.Y = np.asarray(self.Y, dtype=np.float32)
        if self.X.ndim != 2 or self.Y.ndim != 2:
            raise ValueError("X and Y must both be 2-D")
        if len(self.X) != len(self.Y):
            raise ValueError(f"X has {len(self.X)} rows but Y has {len(self.Y)}")
        if self.Y.shape[1] != len(self.concepts):
            raise ValueError("number of concept names must match Y columns")
        if not np.isin(self.Y, (0, 1)).all():
            raise ValueError("Y must be 0/1")
        if not self.ids:
            self.ids = [str(i) for i in range(len(self.X))]

    @property
    def n(self) -> int:
        return self.X.shape[0]

    @property
    def d(self) -> int:
        return self.X.shape[1]

    @property
    def n_concepts(self) -> int:
        return self.Y.shape[1]


def load(embeddings_path: str | Path, labels_path: str | Path) -> ProbeData:
    """Load embeddings (.npy) and labels (.csv).

    labels.csv: an optional `id` column, then one 0/1 column per concept,
    rows in the same order as the embeddings file.
    """
    X = np.load(embeddings_path)
    labels = pd.read_csv(labels_path)
    ids = labels.pop("id").astype(str).tolist() if "id" in labels.columns else []
    return ProbeData(X=X, Y=labels.to_numpy(), concepts=list(labels.columns), ids=ids)


# ---------------------------------------------------------------------------
# Split
# ---------------------------------------------------------------------------

def stratified_split(Y: np.ndarray, fracs=(0.7, 0.1, 0.2), seed: int = 0) -> list[np.ndarray]:
    """Multi-label iterative stratification (Sechidis et al., 2011, simplified).

    A plain random split can leave a rare concept with zero positives in the
    test set, which makes its F1 undefined. This handles the rarest concept
    first and sends each of its chunks to whichever split still needs the
    most positives for it, so every split gets close to its share of every
    concept. Returns index arrays in the order of `fracs` (train, val, test).
    """
    rng = np.random.default_rng(seed)
    Y = np.asarray(Y) > 0
    n, n_concepts = Y.shape
    fracs = np.asarray(fracs, dtype=float) / np.sum(fracs)
    n_splits = len(fracs)

    want_label = fracs[:, None] * Y.sum(0)[None, :]   # (S, D) positives each split still wants
    want_total = fracs * n                            # (S,) chunks each split still wants
    assign = np.full(n, -1)

    def place(i: int, s: int):
        assign[i] = s
        want_label[s] -= Y[i]
        want_total[s] -= 1

    while True:
        open_ = assign < 0
        counts = Y[open_].sum(0)
        if counts.max() == 0:
            break
        c = int(np.argmin(np.where(counts > 0, counts, np.iinfo(np.int64).max)))
        rows = np.flatnonzero(open_ & Y[:, c])
        rng.shuffle(rows)
        for i in rows:
            best = np.flatnonzero(want_label[:, c] == want_label[:, c].max())
            if len(best) > 1:
                best = best[want_total[best] == want_total[best].max()]
            place(i, int(rng.choice(best)))

    rest = np.flatnonzero(assign < 0)       # chunks with no positive label at all
    rng.shuffle(rest)
    for i in rest:
        place(i, int(np.argmax(want_total)))

    return [np.sort(np.flatnonzero(assign == s)) for s in range(n_splits)]


# ---------------------------------------------------------------------------
# Standardization
# ---------------------------------------------------------------------------

@dataclass
class Standardizer:
    """Per-position z-scoring, fit on the training split only.

    Two reasons this matters for the project:
    1. Weight magnitudes are only comparable across positions if the positions
       are on the same scale. Without this, a position with a tiny range gets a
       huge weight and looks "important" for the wrong reason.
    2. After standardizing, setting a position to 0 means setting it to its
       training mean ("mean ablation"), which removes the position's
       information without pushing the vector far off the data distribution.
    """
    mean: np.ndarray
    std: np.ndarray

    @classmethod
    def fit(cls, X: np.ndarray, eps: float = 1e-6) -> "Standardizer":
        return cls(mean=X.mean(0), std=np.maximum(X.std(0), eps))

    def transform(self, X: np.ndarray) -> np.ndarray:
        return ((X - self.mean) / self.std).astype(np.float32)

    def inverse(self, Z: np.ndarray) -> np.ndarray:
        return (Z * self.std + self.mean).astype(np.float32)

    def save(self, path: str | Path):
        np.savez(path, mean=self.mean, std=self.std)


def label_report(Y: np.ndarray, splits: list[np.ndarray], concepts: list[str],
                 names=("train", "val", "test")) -> pd.DataFrame:
    """Positives per concept per split. Check this before trusting any F1."""
    rows = {name: Y[idx].sum(0).astype(int) for name, idx in zip(names, splits)}
    return pd.DataFrame(rows, index=concepts)


def save_split(path: str | Path, splits: list[np.ndarray], ids: list[str]):
    names = ("train", "val", "test")
    Path(path).write_text(json.dumps({k: [ids[i] for i in idx] for k, idx in zip(names, splits)}))
