"""How much labeling error can the probe absorb?

Our labels will come from an LLM and then be vetted. This flips a share of
the train/val labels and measures what that does, so we know how careful
the vetting has to be. Two kinds of error:

  missed yes  a true "yes" is labeled "no" (the LLM was too strict)
  false yes   a "no" is labeled "yes" (too loose); the number added equals
              the given share of the anchor's real yes-labels

For each kind and rate (0 to 30%), 3 random noise draws:
  - test average precision against CLEAN test labels (what the probe
    really learned) and against equally NOISY test labels (what we would
    measure if the test set were not vetted either)
  - overlap of the top-10 positions with the clean-label probe's top 10

    python experiments/build_reuters_lsa.py --per-topic 150 --out experiments/data/reuters_lsa_big
    python -m experiments.label_noise --data experiments/data/reuters_lsa_big
"""

from __future__ import annotations

import argparse
from dataclasses import replace
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


def add_noise(Y: np.ndarray, kind: str, rate: float, rng) -> np.ndarray:
    """Return a copy of Y with `rate` label error of the given kind, per anchor."""
    Y = Y.copy()
    for c in range(Y.shape[1]):
        pos = np.flatnonzero(Y[:, c] == 1)
        neg = np.flatnonzero(Y[:, c] == 0)
        n_flip = int(round(rate * len(pos)))
        if kind == "missed yes":
            Y[rng.choice(pos, size=min(n_flip, len(pos)), replace=False), c] = 0
        elif kind == "false yes":
            Y[rng.choice(neg, size=min(n_flip, len(neg)), replace=False), c] = 1
        else:
            raise ValueError(kind)
    return Y


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="experiments/data/reuters_lsa_big")
    ap.add_argument("--findings", default="experiments/findings")
    ap.add_argument("--rates", type=float, nargs="+", default=(0.0, 0.05, 0.1, 0.2, 0.3))
    ap.add_argument("--draws", type=int, default=3)
    ap.add_argument("--k", type=int, default=10)
    args = ap.parse_args(argv)
    findings = Path(args.findings)

    d = load(Path(args.data) / "embeddings.npy", Path(args.data) / "labels.csv")
    tr, va, te = stratified_split(d.Y, seed=0)
    Z = Standardizer.fit(d.X[tr]).transform(d.X)
    cfg, _ = tune_regularization(Z[tr], d.Y[tr], Z[va], d.Y[va], TrainConfig())
    clean, _ = train_probe(Z[tr], d.Y[tr], Z[va], d.Y[va], cfg)
    clean_top = top_positions(clean.weight_matrix(), args.k)
    print(f"{d.n} chunks, {d.n_concepts} anchors; l2={cfg.l2} l1={cfg.l1}")

    rows = []
    for kind in ("missed yes", "false yes"):
        for rate in args.rates:
            for draw in range(args.draws):
                rng = np.random.default_rng(1000 * draw + int(rate * 100))
                Yn = add_noise(d.Y, kind, rate, rng)          # train, val and (for the noisy score) test
                m, _ = train_probe(Z[tr], Yn[tr], Z[va], Yn[va], replace(cfg, seed=draw))
                p = predict_proba(m, Z[te])
                top = top_positions(m.weight_matrix(), args.k)
                rows.append({
                    "kind": kind, "rate": rate, "draw": draw,
                    "ap_clean_test": float(np.nanmedian(average_precision(p, d.Y[te]))),
                    "ap_noisy_test": float(np.nanmedian(average_precision(p, Yn[te]))),
                    "top_overlap_with_clean": float(np.median(
                        [jaccard(top[c], clean_top[c]) for c in range(d.n_concepts)])),
                })
            print(f"{kind:10s} {rate:.2f}: AP {np.mean([r['ap_clean_test'] for r in rows[-args.draws:]]):.3f}")

    # Reference: clean labels, but a random 5% / 10% of training chunks left out.
    # Shows how much the top-k moves from ordinary sampling, with no label error.
    for drop in (0.05, 0.10):
        for draw in range(args.draws):
            keep = np.random.default_rng(500 + draw).choice(tr, size=int(round((1 - drop) * len(tr))), replace=False)
            m, _ = train_probe(Z[keep], d.Y[keep], Z[va], d.Y[va], replace(cfg, seed=draw))
            top = top_positions(m.weight_matrix(), args.k)
            ap_ = float(np.nanmedian(average_precision(predict_proba(m, Z[te]), d.Y[te])))
            rows.append({"kind": "reference: drop chunks", "rate": drop, "draw": draw,
                         "ap_clean_test": ap_, "ap_noisy_test": ap_,
                         "top_overlap_with_clean": float(np.median(
                             [jaccard(top[c], clean_top[c]) for c in range(d.n_concepts)]))})

    df = pd.DataFrame(rows)
    table = df.groupby(["kind", "rate"]).mean(numeric_only=True).drop(columns="draw").round(3)
    table.to_csv(findings / "label_noise.csv")
    print(table)

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(10, 3.8))
    colors = {"missed yes": "#2a6fdb", "false yes": "#d9822b"}
    ref = table.loc["reference: drop chunks"]
    for kind, g in table.drop(index="reference: drop chunks", level="kind").groupby(level="kind"):
        r = g.index.get_level_values("rate") * 100
        a1.plot(r, g.ap_clean_test, "o-", color=colors[kind], label=f"{kind}, clean test")
        a1.plot(r, g.ap_noisy_test, "o--", color=colors[kind], alpha=0.6, label=f"{kind}, noisy test")
        a2.plot(r, g.top_overlap_with_clean, "o-", color=colors[kind], label=kind)
    a2.plot(ref.index * 100, ref.top_overlap_with_clean, "s", color="grey",
            label="no label error, 5% / 10% of chunks left out")
    base = random_jaccard(d.d, args.k)
    a2.axhline(base, ls="--", color="grey", lw=1)
    a2.text(max(args.rates) * 100, base + 0.02, "random lists", color="grey", fontsize=9, ha="right")
    a1.set(title="Detecting the anchor", xlabel="labels wrong (%)", ylabel="median test average precision",
           ylim=(0, 1.02))
    a2.set(title=f"Same top-{args.k} positions as clean labels", xlabel="labels wrong (%)",
           ylabel=f"top-{args.k} overlap with clean probe", ylim=(0, 1.02))
    a1.legend(fontsize=8, loc="lower left")
    a2.legend(fontsize=8, loc="lower left")
    for a in (a1, a2):
        a.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(findings / "label_noise.png", dpi=150)


if __name__ == "__main__":
    main()
