# Real-text check: does the pipeline work on actual articles?

**Yes.** On 1,005 Reuters articles and 40 topics, the probe reaches 0.91
median average precision (chance 0.04, shuffled-label control 0.07), and
each topic's top-weighted positions matter far more than random ones.

## Setup

- Data: `python experiments/build_reuters_lsa.py` (Reuters-21578 via NLTK).
  40 topics = the 3rd to 42nd largest (`earn`, `acq` skipped). 32 to 131
  articles per topic, one article per chunk.
- Embeddings: **LSA stand-in** (TF-IDF, then SVD to 256 positions). Not the
  model we will use. It was picked because it needs no download, and
  it gives real text, real topic overlap, and correlated positions.
- Run: `python -m probe.run --embeddings experiments/data/reuters_lsa/embeddings.npy --labels experiments/data/reuters_lsa/labels.csv --out experiments/data/run_lsa`
  (defaults; about 2 minutes on a laptop CPU).

## Results

| Check | Result |
|---|---|
| Probe, test set | median F1 0.78, median average precision 0.91 |
| Chance / shuffled-label control | 0.04 / 0.07 average precision |
| Share of weight in top 10 of 256 positions | median 0.27 (an even spread would be 0.04) |
| Top-10 overlap across 5 retrains (stability) | median 0.48 (random lists: 0.02) |

Ablation, median drop in average precision (`real_text_ablation_summary.csv`):

| k removed | top-k, retrained probe | k random, retrained probe |
|---|---|---|
| 10 | 0.046 | 0.001 |
| 50 | 0.138 | 0.005 |
| 100 | 0.347 | 0.033 |

## What this tells us

1. The pipeline runs end to end on real data and the controls behave.
2. The top positions are real, not noise: removing them costs 10 to 40
   times more than removing random ones, even when the probe is retrained.
3. Topics are **spread out**. Removing the top 10 costs little; it takes
   50 to 100 positions to cut a topic's score by a third. The retrained
   probe recovering part of the loss (0.35 vs 0.57 frozen at k = 100)
   means some information is duplicated in other positions.
4. Per-topic numbers are noisy: with 1,005 chunks the test set has only
   7 to 27 positives per topic. See the sample-size check.

Per-topic table: `real_text_per_concept.csv`.
