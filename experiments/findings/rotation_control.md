# Rotation control: are individual positions meaningful?

**Partly. Ablation passes, the weight-based numbers do not.** Removing a
concept's top positions hurts 3 to 5 times more in the original axes than
in a randomly rotated copy of the same embeddings. But concentration and
stability are almost as high after rotation, so on their own they say
little about positions.

## Why this test

A random rotation keeps all the information in the embeddings but mixes
every original position into all the new ones. If our results came out
just as strong in the rotated basis, "concept X lives in positions 12, 40,
301" would not be a finding, since any basis would give an equally good list.

## Setup

Same 1,005 Reuters articles, 40 topics and LSA stand-in embeddings as the
real-text check. The pipeline is run twice with identical settings: once
on the embeddings, once on the embeddings times a random orthogonal
matrix. Retrain-mode ablation, 10 random draws.
Run: `python -m experiments.rotation_control` (about 3 minutes).

## Results (`rotation_control.csv`, medians over 40 topics)

| Measure | Original axes | Rotated axes | Original / rotated |
|---|---|---|---|
| Test average precision | 0.905 | 0.908 | 1.00 |
| Weight share in top 10 (even spread = 0.04) | 0.267 | 0.224 | 1.19 |
| Top-10 overlap across retrains | 0.482 | 0.431 | 1.12 |
| AP drop, top 10 removed (retrained) | 0.046 | 0.010 | 4.6 |
| AP drop, top 50 removed (retrained) | 0.138 | 0.046 | 3.0 |

Random-position drops are about 0.001 (k = 10) and 0.005 to 0.009 (k = 50)
in both bases.

## What this means for us

1. Accuracy is identical, as it should be: rotation loses no information.
2. **Ablation is the evidence.** In the original axes, concept information
   sits in fewer positions than in an arbitrary basis.
3. **Concentration and stability need a baseline.** Even in a random
   basis the probe puts 22% of its weight on 10 of 256 positions. A
   regularized probe always does this to some degree. So in the paper,
   report these next to the rotated value, never alone.
4. Run this control on the real model's embeddings before we interpret
   any position list. If the ablation ratio drops to about 1 there, the
   project's framing has to change from "which positions" to "which
   directions".

Caveat: LSA axes are SVD components, ordered by variance, so they may be
more aligned with topics than a neural model's positions are.
