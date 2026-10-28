"""End-to-end probe experiment.

    python -m probe.run --embeddings data/embeddings.npy --labels data/labels.csv --out results/run1

Steps:
 1. load + check data, stratified train/val/test split, standardize on train
 2. train the linear probe, tune thresholds on val, score on test
 3. control: same probe trained on shuffled labels (should be near chance)
 4. resample the training set a few times, check the top-k positions are stable
 5. ablation of top-k vs random positions (frozen and/or retrain)
 6. specificity matrix
Everything is written to --out.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np

from . import ablation, positions
from .data import Standardizer, label_report, load, save_split, stratified_split
from .evaluate import per_concept, summary, tune_thresholds
from .train import TrainConfig, predict_proba, train_probe, tune_regularization


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--embeddings", required=True, help=".npy, shape (N, d)")
    ap.add_argument("--labels", required=True, help=".csv, optional id column + one 0/1 column per concept")
    ap.add_argument("--out", required=True)
    ap.add_argument("--split", type=float, nargs=3, default=(0.7, 0.1, 0.2), metavar=("TRAIN", "VAL", "TEST"))
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--min-test-pos", type=int, default=5,
                    help="concepts with fewer test positives are dropped (their F1 means nothing)")
    ap.add_argument("--l1", type=float, default=None, help="fix L1 instead of tuning on val")
    ap.add_argument("--l2", type=float, default=None, help="fix L2 instead of tuning on val")
    ap.add_argument("--lr", type=float, default=1e-2)
    ap.add_argument("--ks", type=int, nargs="+", default=(10, 50, 100))
    ap.add_argument("--n-random", type=int, default=20)
    ap.add_argument("--ablation", choices=("frozen", "retrain", "both", "none"), default="both")
    ap.add_argument("--n-resamples", type=int, default=5, help="training-set resamples for position stability")
    ap.add_argument("--resample-frac", type=float, default=0.8)
    args = ap.parse_args(argv)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    cfg = TrainConfig(lr=args.lr, seed=args.seed)

    # 1. data ---------------------------------------------------------------
    data = load(args.embeddings, args.labels)
    splits_idx = stratified_split(data.Y, args.split, seed=args.seed)
    counts = label_report(data.Y, splits_idx, data.concepts)
    counts.to_csv(out / "label_counts.csv")

    keep = counts["test"].to_numpy() >= args.min_test_pos
    if not keep.all():
        dropped = counts.index[~keep].tolist()
        print(f"[warn] dropping {len(dropped)} concepts with < {args.min_test_pos} test positives: {dropped}",
              file=sys.stderr)
    if not keep.any():
        sys.exit("no concept has enough positives to evaluate; fix the topic list / labels first")
    concepts = [c for c, k in zip(data.concepts, keep) if k]
    Y = data.Y[:, keep]

    tr, va, te = splits_idx
    scaler = Standardizer.fit(data.X[tr])
    scaler.save(out / "standardizer.npz")      # the reconstruction step needs this to un-standardize
    save_split(out / "split.json", splits_idx, data.ids)
    Z = scaler.transform(data.X)
    S = (Z[tr], Y[tr], Z[va], Y[va], Z[te], Y[te])
    Ztr, Ytr, Zva, Yva, Zte, Yte = S
    k_max = min(max(args.ks), data.d)
    ks = [k for k in args.ks if k < data.d]

    # 2. main probe ---------------------------------------------------------
    tuning = []
    if args.l1 is None and args.l2 is None:
        cfg, tuning = tune_regularization(Ztr, Ytr, Zva, Yva, cfg)
    else:
        cfg = replace(cfg, l1=args.l1 or 0.0, l2=args.l2 or 0.0)
    model, hist = train_probe(Ztr, Ytr, Zva, Yva, cfg)
    thr = tune_thresholds(predict_proba(model, Zva), Yva)
    p_te = predict_proba(model, Zte)
    table = per_concept(p_te, Yte, thr, concepts)
    W = model.weight_matrix()
    np.save(out / "weights.npy", W)

    # 3. shuffled-label control ---------------------------------------------
    rng = np.random.default_rng(args.seed)
    perm = rng.permutation(len(Ytr))
    ctrl, _ = train_probe(Ztr, Ytr[perm], Zva, Yva, cfg)
    ctrl_table = per_concept(predict_proba(ctrl, Zte), Yte, tune_thresholds(predict_proba(ctrl, Zva), Yva), concepts)
    table["control_avg_precision"] = ctrl_table["avg_precision"]

    # 4. positions + stability ----------------------------------------------
    resampled = []
    for r in range(args.n_resamples):
        sub = np.random.default_rng(args.seed + 1000 + r).choice(
            len(Ztr), size=int(args.resample_frac * len(Ztr)), replace=False)
        m, _ = train_probe(Ztr[sub], Ytr[sub], Zva, Yva, replace(cfg, seed=args.seed + r))
        resampled.append(m.weight_matrix())

    top = {k: positions.top_positions(W, k) for k in ks}
    for k in ks:
        table[f"concentration_top{k}"] = positions.concentration(W, k)
        if resampled:
            table[f"stability_top{k}"] = positions.stability(resampled, k)
    table.to_csv(out / "per_concept.csv")

    pos_out = {
        c: {
            "top_positions": positions.top_positions(W, k_max)[i].tolist(),
            "top_weights": W[i, positions.top_positions(W, k_max)[i]].round(4).tolist(),
        }
        for i, c in enumerate(concepts)
    }
    (out / "positions.json").write_text(json.dumps(pos_out, indent=1))

    # 5. ablation -----------------------------------------------------------
    abl = []
    if args.ablation in ("frozen", "both"):
        abl.append(ablation.run(S, top, table["avg_precision"].to_numpy(), concepts, cfg,
                                mode="frozen", model=model, n_random=args.n_random, seed=args.seed))
    if args.ablation in ("retrain", "both"):
        abl.append(ablation.run(S, top, table["avg_precision"].to_numpy(), concepts, cfg,
                                mode="retrain", n_random=args.n_random, seed=args.seed))
    if abl:
        import pandas as pd
        pd.concat(abl).to_csv(out / "ablation.csv", index=False)

    # 6. specificity --------------------------------------------------------
    if ks:
        spec = ablation.specificity(model, Zte, Yte, top[ks[0]], table["avg_precision"].to_numpy())
        np.save(out / f"specificity_top{ks[0]}.npy", spec)

    # summary ---------------------------------------------------------------
    summ = summary(table, p_te, Yte)
    summ["control_macro_avg_precision"] = float(table["control_avg_precision"].mean())
    if resampled:
        summ.update({f"random_jaccard_top{k}": positions.random_jaccard(data.d, k) for k in ks})
    result = {
        "n_chunks": data.n, "d": data.d, "n_concepts": len(concepts),
        "split_sizes": [len(tr), len(va), len(te)],
        "config": cfg.to_dict(), "regularization_search": tuning, "training": {k: hist[k] for k in ("best_epoch", "epochs_run", "best_val_loss")},
        "summary": summ,
    }
    (out / "metrics.json").write_text(json.dumps(result, indent=1, default=float))
    print(json.dumps(result["summary"], indent=1))
    return result


if __name__ == "__main__":
    main()
