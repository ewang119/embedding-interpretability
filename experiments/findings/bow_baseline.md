# Bag-of-words baseline

**The baseline is strong on long articles and much weaker on short
chunks.** On full Reuters articles word counts reach 0.90 median average
precision. On 3-sentence chunks, the length of our WWII passages, they
reach 0.86; on single sentences they drop to 0.64. Short passages are where
the real embeddings have room to beat it.

## Setup

- `python -m probe.baseline`: TF-IDF (words and word pairs, fit on train
  only) plus one logistic regression per anchor. Same split, same concept
  filter and same metrics as `probe.run`, so per-anchor numbers line up.
- Data: the 1,005 Reuters articles and 40 topics from the real-text check.
- Short chunks: each article cut to its first 1, 2 or 3 sentences, same
  labels.

## Results

| Chunk | Median words | Bag-of-words median avg. precision | Hardest topics |
|---|---|---|---|
| Full article | 126 | 0.90 | |
| First 3 sentences | 73 | 0.86 | soybean 0.42, ship 0.56, oilseed 0.57 |
| First 2 sentences | 53 | 0.83 | soybean 0.47, oilseed 0.55, ship 0.59 |
| First sentence | 32 | 0.64 | soybean 0.22, gnp 0.32, oilseed 0.39 |

Probe on the LSA stand-in embeddings (full articles): 0.905, against 0.904
for the baseline. 17 of 40 topics tie, 10 go to the probe, 13 to the
baseline. That is expected: LSA is built from word counts, so it cannot
know more than they do. This comparison only becomes meaningful with a
neural embedding model.

## What this means for us

1. Run this on the WWII data next to `probe.run`, with `--split-from` and
   `--compare`. It is the "beats bag of words" number in the proposal.
2. Short chunks make the bar lower and the test fairer. An anchor that
   bag-of-words already gets at 0.95 says little about the embedding.
3. The hardest topics are the ones that overlap (soybean and oilseed share
   most of their articles). Expect the same for close WW2 anchors, such as
   appeasement and the Munich Agreement.
4. Reuters' sentences are long (three of them average 73 words). If WWII
   passages are shorter, the baseline should come out lower still.
