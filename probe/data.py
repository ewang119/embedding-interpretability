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

def stratified_split(Y: np.ndarray, fracs=(0.7, 0.1, 0.2), seed: int = 0,
                     groups=None) -> list[np.ndarray]:
    """Multi-label iterative stratification (Sechidis et al., 2011, simplified).

    A plain random split can leave a rare concept with zero positives in the
    test set, which makes its F1 undefined. This handles the rarest concept
    first and sends each of its chunks to whichever split still needs the
    most positives for it, so every split gets close to its share of every
    concept. Returns index arrays in the order of `fracs` (train, val, test).

    groups: optional group key per chunk (e.g. the WWII `source_id`). All
    chunks of a group go to the same split, so passages from one article
    never end up in both train and test. Groups are placed whole, weighted by
    how many chunks and positives they hold, so proportions are approximate
    when a few groups are large. Without groups every chunk is its own group
    and the result is the same as before groups were supported.
    """
    rng = np.random.default_rng(seed)
    Y = np.asarray(Y) > 0
    n = Y.shape[0]
    if groups is None:
        member = np.arange(n)
    else:
        groups = np.asarray(groups)
        if len(groups) != n:
            raise ValueError(f"{len(groups)} group keys for {n} chunks")
        _, first, member = np.unique(groups, return_index=True, return_inverse=True)
        # renumber groups by first appearance so the result doesn't depend on key spelling
        order = np.argsort(np.argsort(first))
        member = order[member]
    n_groups = member.max() + 1
    L = np.zeros((n_groups, Y.shape[1]), dtype=np.int64)       # positives per group and concept
    np.add.at(L, member, Y.astype(np.int64))
    size = np.bincount(member, minlength=n_groups)              # chunks per group

    fracs = np.asarray(fracs, dtype=float) / np.sum(fracs)
    n_splits = len(fracs)
    want_label = fracs[:, None] * L.sum(0)[None, :]   # (S, D) positives each split still wants
    want_total = fracs * n                            # (S,) chunks each split still wants
    assign = np.full(n_groups, -1)

    def place(g: int, s: int):
        assign[g] = s
        want_label[s] -= L[g]
        want_total[s] -= size[g]

    while True:
        open_ = assign < 0
        counts = L[open_].sum(0)
        if counts.max() == 0:
            break
        c = int(np.argmin(np.where(counts > 0, counts, np.iinfo(np.int64).max)))
        rows = np.flatnonzero(open_ & (L[:, c] > 0))
        rng.shuffle(rows)
        for g in rows:
            best = np.flatnonzero(want_label[:, c] == want_label[:, c].max())
            if len(best) > 1:
                best = best[want_total[best] == want_total[best].max()]
            place(g, int(rng.choice(best)))

    rest = np.flatnonzero(assign < 0)       # groups with no positive label at all
    rng.shuffle(rest)
    for g in rest:
        place(g, int(np.argmax(want_total)))

    chunk_split = assign[member]
    return [np.sort(np.flatnonzero(chunk_split == s)) for s in range(n_splits)]


SPLIT_NAMES = ("train", "val", "test")


def split_from_meta(meta: pd.DataFrame, ids: list[str], Y: np.ndarray, fracs=(0.7, 0.1, 0.2),
                    seed: int = 0) -> tuple[list[np.ndarray], str]:
    """Split using a per-chunk metadata table (columns: id, optional source_id, optional split).

    - every chunk has a `split` (train/val/test) -> use it as given
    - otherwise, `source_id` present          -> stratified split, sources kept whole
    - otherwise                               -> plain stratified split by chunk
    Returns (train, val, test index arrays, description of what was done).
    """
    meta = meta.copy()
    meta["id"] = meta["id"].astype(str)
    if meta["id"].duplicated().any():
        raise ValueError("meta has duplicate ids")
    missing = set(ids) - set(meta["id"])
    if missing:
        raise ValueError(f"{len(missing)} chunk ids missing from meta, e.g. {sorted(missing)[:3]}")
    meta = meta.set_index("id").loc[list(ids)]

    if "split" in meta.columns and meta["split"].notna().all() and (meta["split"].astype(str) != "").all():
        bad = set(meta["split"]) - set(SPLIT_NAMES)
        if bad:
            raise ValueError(f"split values must be train/val/test, got {sorted(bad)}")
        parts = [np.flatnonzero(meta["split"].to_numpy() == s) for s in SPLIT_NAMES]
        if any(len(p) == 0 for p in parts):
            raise ValueError("given split leaves train, val or test empty")
        if "source_id" in meta.columns:
            src = meta["source_id"].to_numpy()
            crossing = {s for s in src if len({meta['split'].iloc[i] for i in np.flatnonzero(src == s)}) > 1}
            if crossing:
                raise ValueError(f"{len(crossing)} source(s) appear in more than one split, e.g. {sorted(crossing)[:3]}")
        return parts, "given in meta"
    if "source_id" in meta.columns and meta["source_id"].notna().all():
        return stratified_split(Y, fracs, seed, groups=meta["source_id"].astype(str).to_numpy()), \
            "stratified, sources kept whole"
    return stratified_split(Y, fracs, seed), "stratified by chunk"


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
