"""Rotation control: are individual positions meaningful, or just one of
many equally good bases?

A random rotation (X @ Q, Q orthogonal) keeps every distance and all of the
information in the embeddings, but mixes each original position into all
of the new ones. So:
  - probe accuracy should be about the same in both bases;
  - if the ORIGINAL positions line up with concepts, concentration,
    stability and top-k ablation should all be clearly stronger there than
    in the rotated basis;
  - if they are about equal, "concept X lives in positions 12, 40, 301" is
    not a finding: any basis would give a list just as good.

    python -m experiments.rotation_control --data experiments/data/reuters_lsa
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from probe.run import main as run_probe


def random_rotation(d: int, seed: int) -> np.ndarray:
    q, r = np.linalg.qr(np.random.default_rng(seed).normal(size=(d, d)))
    return q * np.sign(np.diag(r))          # sign fix -> uniform over rotations


def summarize(out: Path, ks) -> dict:
    t = pd.read_csv(out / "per_concept.csv", index_col=0)
    a = pd.read_csv(out / "ablation.csv")
    s = {"median_avg_precision": t.avg_precision.median()}
    for k in ks:
        s[f"concentration_top{k}"] = t[f"concentration_top{k}"].median()
        s[f"stability_top{k}"] = t[f"stability_top{k}"].median()
        ak = a[(a["mode"] == "retrain") & (a.k == k)]
        s[f"retrain_drop_top{k}"] = ak.top_drop.median()
        s[f"retrain_drop_random{k}"] = ak.random_drop_mean.median()
    return s


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="experiments/data/reuters_lsa")
    ap.add_argument("--out", default="experiments/data/rotation")
    ap.add_argument("--findings", default="experiments/findings")
    ap.add_argument("--ks", type=int, nargs="+", default=(10, 50))
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args(argv)
    data, out = Path(args.data), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    X = np.load(data / "embeddings.npy")
    np.save(out / "rotated.npy", (X @ random_rotation(X.shape[1], args.seed)).astype(np.float32))

    common = ["--labels", str(data / "labels.csv"), "--ablation", "retrain", "--n-random", "10",
              "--ks", *map(str, args.ks), "--seed", str(args.seed)]
    run_probe(["--embeddings", str(data / "embeddings.npy"), "--out", str(out / "original"), *common])
    run_probe(["--embeddings", str(out / "rotated.npy"), "--out", str(out / "rotated"), *common])

    table = pd.DataFrame({"original": summarize(out / "original", args.ks),
                          "rotated": summarize(out / "rotated", args.ks)}).round(3)
    table["ratio"] = (table.original / table.rotated).round(2)
    d = X.shape[1]
    table["note"] = ""
    for k in args.ks:
        table.loc[f"concentration_top{k}", "note"] = f"even spread = {k / d:.3f}"
    table.to_csv(Path(args.findings) / "rotation_control.csv")
    print(table.to_string())
    (out / "summary.json").write_text(json.dumps(table.to_dict(), indent=1))


if __name__ == "__main__":
    main()
