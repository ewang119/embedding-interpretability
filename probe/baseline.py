"""Bag-of-words baseline: can plain word counts find each anchor?

    python -m probe.baseline --texts data/wwii/texts.csv --labels data/wwii/labels.csv \
        --split-from results/wwii_run1/split.json \
        --out results/bow --compare results/wwii_run1/per_concept.csv

The proposal's bar: the embedding probe should do at least as well as a
model that only sees which words appear. If an anchor is just as easy for
bag-of-words, finding it in the embedding says little about the embedding.

Same footing as probe.run, so the two are directly comparable:
  - same split: --split-from reuses the exact train/val/test ids probe.run
    saved (split.json), including keeping each source in one split; without
    it, the same stratified row split as probe.run with no --meta
  - same filter on concepts with too few test positives
  - same metrics (per-anchor F1 at a val-tuned threshold, average precision)
Model: TF-IDF (unigrams + bigrams, fit on train text only) and one
class-balanced logistic regression per anchor, strength C picked on val.

Inputs: texts.csv (id, text) and labels.csv (id + 0/1 anchor columns), as
written by `python -m probe.check_dataset`. Rows are matched by id.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

from .data import label_report, stratified_split
from .evaluate import average_precision, per_concept, summary, tune_thresholds

C_GRID = (0.1, 1.0, 10.0)


def fit_predict(Xtr, ytr, Xval, Xte, C):
    if ytr.sum() == 0:
        return np.zeros(Xval.shape[0]), np.zeros(Xte.shape[0])
    m = LogisticRegression(C=C, class_weight="balanced", max_iter=2000).fit(Xtr, ytr)
    return m.predict_proba(Xval)[:, 1], m.predict_proba(Xte)[:, 1]


def run_baseline(texts: list[str], Y: np.ndarray, concepts: list[str], split=(0.7, 0.1, 0.2),
                 seed: int = 0, min_test_pos: int = 5, c_grid=C_GRID, parts=None):
    """parts: optional (train, val, test) row indices to use instead of making a split."""
    tr, va, te = parts if parts is not None else stratified_split(Y, split, seed=seed)
    counts = label_report(Y, [tr, va, te], concepts)
    keep = counts["test"].to_numpy() >= min_test_pos
    dropped = counts.index[~keep].tolist()
    concepts = [c for c, k in zip(concepts, keep) if k]
    Y = Y[:, keep]
    if not concepts:
        raise ValueError("no concept has enough test positives")

    vec = TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True, stop_words="english")
    Xtr = vec.fit_transform([texts[i] for i in tr])
    Xva = vec.transform([texts[i] for i in va])
    Xte = vec.transform([texts[i] for i in te])

    p_va = np.zeros((len(va), len(concepts)))
    p_te = np.zeros((len(te), len(concepts)))
    chosen_C = []
    for c in range(len(concepts)):
        best = None
        for C in c_grid:
            pv, pt = fit_predict(Xtr, Y[tr, c], Xva, Xte, C)
            ap = average_precision(pv[:, None], Y[va][:, [c]])[0]
            score = -1.0 if np.isnan(ap) else ap
            if best is None or score > best[0]:
                best = (score, C, pv, pt)
        _, C, p_va[:, c], p_te[:, c] = best
        chosen_C.append(C)

    thr = tune_thresholds(p_va, Y[va])
    table = per_concept(p_te, Y[te], thr, concepts)
    table["C"] = chosen_C
    return table, summary(table, p_te, Y[te]), {"dropped": dropped, "vocab_size": len(vec.vocabulary_),
                                                "split_sizes": [len(tr), len(va), len(te)]}


def compare(bow: pd.DataFrame, probe: pd.DataFrame) -> pd.DataFrame:
    both = bow[["f1", "avg_precision"]].join(probe[["f1", "avg_precision"]], lsuffix="_bow", rsuffix="_probe",
                                             how="inner")
    both["ap_gain"] = both.avg_precision_probe - both.avg_precision_bow
    both["winner"] = np.select([both.ap_gain > 0.02, both.ap_gain < -0.02], ["probe", "bow"], "tie")
    return both.sort_values("ap_gain")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--texts", required=True, help="CSV with id, text")
    ap.add_argument("--labels", required=True, help="CSV with id + one 0/1 column per anchor")
    ap.add_argument("--out", required=True)
    ap.add_argument("--compare", help="probe.run's per_concept.csv, to put the two side by side")
    ap.add_argument("--split", type=float, nargs=3, default=(0.7, 0.1, 0.2))
    ap.add_argument("--split-from", help="split.json written by probe.run; reuse its exact train/val/test ids")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--min-test-pos", type=int, default=5)
    args = ap.parse_args(argv)

    labels = pd.read_csv(args.labels)
    texts = pd.read_csv(args.texts).set_index("id")["text"]
    missing = set(labels["id"]) - set(texts.index)
    if missing:
        sys.exit(f"{len(missing)} label ids have no text, e.g. {sorted(missing)[:3]}")
    concepts = [c for c in labels.columns if c != "id"]
    parts = None
    if args.split_from:
        given = json.loads(Path(args.split_from).read_text())
        pos = {str(i): k for k, i in enumerate(labels["id"])}
        unknown = [i for name in ("train", "val", "test") for i in given[name] if str(i) not in pos]
        if unknown:
            sys.exit(f"--split-from: {len(unknown)} ids not in labels, e.g. {unknown[:3]}")
        parts = [np.array(sorted(pos[str(i)] for i in given[name]), dtype=int) for name in ("train", "val", "test")]
        if sum(map(len, parts)) != len(labels):
            sys.exit(f"--split-from covers {sum(map(len, parts))} of {len(labels)} chunks")

    table, summ, info = run_baseline(texts.loc[labels["id"]].fillna("").tolist(), labels[concepts].to_numpy(),
                                     concepts, tuple(args.split), args.seed, args.min_test_pos, parts=parts)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    table.to_csv(out / "per_concept.csv")
    result = {"summary": summ, **info}
    if args.compare:
        cmp = compare(table, pd.read_csv(args.compare, index_col=0))
        cmp.to_csv(out / "compare.csv")
        result["compare"] = {
            "median_ap_bow": float(cmp.avg_precision_bow.median()),
            "median_ap_probe": float(cmp.avg_precision_probe.median()),
            "probe_wins": int((cmp.winner == "probe").sum()),
            "bow_wins": int((cmp.winner == "bow").sum()),
            "ties": int((cmp.winner == "tie").sum()),
        }
    (out / "metrics.json").write_text(json.dumps(result, indent=1, default=float))
    print(json.dumps(result, indent=1, default=float))
    return result


if __name__ == "__main__":
    main()
