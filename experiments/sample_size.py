"""How many labeled chunks does each concept need?

Trains the probe on growing random fractions of the training set (test set
fixed), several seeds each, and records per concept:
  - test average precision (can the probe detect the concept?)
  - top-k position overlap between seeds (are the positions it picks stable?)
against the number of training positives that concept had.

    python experiments/build_reuters_lsa.py --per-topic 150 --out experiments/data/reuters_lsa_big
    python -m experiments.sample_size --data experiments/data/reuters_lsa_big
"""

from __future__ import annotations

import argparse
from itertools import combinations
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from probe.data import Standardizer, load, stratified_split
from probe.evaluate import average_precision
from probe.positions import jaccard, random_jaccard, top_positions
from probe.train import TrainConfig, predict_proba, train_probe, tune_regularization

BINS = [0, 10, 20, 30, 50, 75, 100, 1000]
BIN_LABELS = ["<10", "10-19", "20-29", "30-49", "50-74", "75-99", "100+"]


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="experiments/data/reuters_lsa_big")
    ap.add_argument("--out", default="experiments/data/sample_size")
    ap.add_argument("--findings", default="experiments/findings")
    ap.add_argument("--fracs", type=float, nargs="+", default=(0.1, 0.2, 0.35, 0.5, 0.65, 0.8))
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--k", type=int, default=10)
    args = ap.parse_args(argv)
    out, findings = Path(args.out), Path(args.findings)
    out.mkdir(parents=True, exist_ok=True)

    d = load(Path(args.data) / "embeddings.npy", Path(args.data) / "labels.csv")
    tr, va, te = stratified_split(d.Y, seed=0)
    Z = Standardizer.fit(d.X[tr]).transform(d.X)   # scaler from the full train split, held fixed
    Zva, Yva, Zte, Yte = Z[va], d.Y[va], Z[te], d.Y[te]
    cfg, _ = tune_regularization(Z[tr], d.Y[tr], Zva, Yva, TrainConfig())
    print(f"{d.n} chunks, {len(tr)} train, {len(te)} test; l2={cfg.l2} l1={cfg.l1}")

    rows = []
    for frac in args.fracs:
        tops = []
        for s in range(args.seeds):
            # fracs stay below 1.0 so each seed trains on a different sample;
            # at 1.0 every seed sees the same data and "stability" is trivially 1
            sub = np.random.default_rng(100 + s).choice(tr, size=int(round(frac * len(tr))), replace=False)
            m, _ = train_probe(Z[sub], d.Y[sub], Zva, Yva, cfg.__class__(**{**cfg.to_dict(), "seed": s}))
            ap_ = average_precision(predict_proba(m, Zte), Yte)
            tops.append(top_positions(m.weight_matrix(), args.k))
            for c, name in enumerate(d.concepts):
                rows.append({"frac": frac, "seed": s, "concept": name,
                             "train_pos": int(d.Y[sub, c].sum()), "test_ap": ap_[c]})
        # stability: mean pairwise overlap of top-k across seeds, per concept
        stab = [np.mean([jaccard(tops[i][c], tops[j][c]) for i, j in combinations(range(len(tops)), 2)])
                for c in range(d.n_concepts)]
        for r in rows[-args.seeds * d.n_concepts:]:
            r["stability"] = stab[d.concepts.index(r["concept"])]
        print(f"frac {frac:.2f}: median AP {np.nanmedian([r['test_ap'] for r in rows[-args.seeds * d.n_concepts:]]):.3f}")

    df = pd.DataFrame(rows)
    df.to_csv(out / "runs.csv", index=False)
    df["bin"] = pd.cut(df.train_pos, BINS, right=False, labels=BIN_LABELS)
    table = df.groupby("bin", observed=True).agg(
        n_points=("test_ap", "size"),
        median_ap=("test_ap", "median"), ap_q25=("test_ap", lambda x: x.quantile(0.25)),
        ap_q75=("test_ap", lambda x: x.quantile(0.75)),
        median_stability=("stability", "median"),
    ).round(3)
    table.to_csv(findings / "sample_size.csv")
    print(table)

    chance = float(np.median(Yte.mean(0)))
    base = random_jaccard(d.d, args.k)
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(10, 3.8))
    x = np.arange(len(table))
    a1.errorbar(x, table.median_ap, yerr=[table.median_ap - table.ap_q25, table.ap_q75 - table.median_ap],
                fmt="o-", capsize=3, color="#2a6fdb")
    a1.axhline(chance, ls="--", color="grey", lw=1)
    a1.text(len(table) - 1, chance + 0.02, "chance", color="grey", fontsize=9, ha="right")
    a1.set(title="Detecting the concept", xlabel="training positives for the concept",
           ylabel="test average precision (median, IQR)", ylim=(0, 1.02))
    a2.plot(x, table.median_stability, "o-", color="#d9822b")
    a2.axhline(base, ls="--", color="grey", lw=1)
    a2.text(len(table) - 1, base + 0.02, "random lists", color="grey", fontsize=9, ha="right")
    a2.set(title=f"Picking the same top-{args.k} positions", xlabel="training positives for the concept",
           ylabel=f"top-{args.k} overlap across seeds (median)", ylim=(0, 1.02))
    for a in (a1, a2):
        a.set_xticks(x, table.index.astype(str))
        a.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(findings / "sample_size.png", dpi=150)


if __name__ == "__main__":
    main()
