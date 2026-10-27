# Probe training pipeline

Trains the per-concept probe from the proposal, reads candidate positions out
of its weights, and tests them with top-k vs random ablation.

It does not depend on which topics or which embedding model we pick. It only
needs two files, in the same row order:

| file | shape | what |
|---|---|---|
| `embeddings.npy` | (N, d) | one vector per chunk, one model, same settings for every chunk |
| `labels.csv` | N rows | optional `id` column, then one 0/1 column per concept |

```bash
pip install -r requirements.txt
python -m probe.run --embeddings data/embeddings.npy --labels data/labels.csv --out results/run1
python -m pytest tests -q        # synthetic end-to-end check, ~15 s
```

A full run at our scale (1,000 chunks, d = 768, 40 concepts, 20 random draws)
takes about a minute on a laptop CPU.

## What a run does

1. **Split** 70/10/20 train/val/test with multi-label stratification, so every
   concept keeps its share of positives in every split. Concepts with fewer
   than 5 test positives are dropped with a warning (`--min-test-pos`), since
   their F1 means nothing.
2. **Standardize** each position using train-set mean and std. Needed so weight
   sizes can be compared across positions, and it means "zero out a position"
   is the same as "set it to its average" (mean ablation).
3. **Pick regularization** (L2 or L1 strength) on the val split. This matters
   a lot: with ~50 positives per concept and 768 positions the probe is short
   on data, and in our synthetic test the setting moved the score by more
   than 2x. Pass `--l1` / `--l2` to fix it by hand.
4. **Train** `nn.Linear(d, D)` with per-concept BCE (one sigmoid neuron per
   concept, D x d weights), weighted for the many "no" labels, early stopping
   on val loss. Tune one decision threshold per concept on val, score on test.
5. **Shuffled-label control.** Same probe trained on shuffled labels. It should
   score near chance; if it doesn't, something is leaking.
6. **Positions.** Top-k positions per concept by |weight|, plus:
   - *concentration*: share of the concept's squared weight in its top k
   - *stability*: retrain on 5 random 80% subsets of train and measure how
     much the top-k lists overlap (Jaccard). Compare against
     `random_jaccard_topk` in `metrics.json`. A top-k list that changes
     when the training data changes slightly is noise.
7. **Ablation** at k = 10, 50, 100, against 20 random k-position draws:
   - *frozen*: same probe, ablated test vectors. Cheap but partly circular for
     a linear probe, since removing its largest-weight inputs hurts it almost
     by construction.
   - *retrain*: ablate everything, train a fresh probe. The fresh probe can
     use whatever is left, so a drop here means the rest of the vector really
     lacks the information. **This is the number to report.**
   - `p_value` = share of random draws that hurt at least as much as the top-k
     did. With 20 draws the smallest possible value is 1/21 ≈ 0.048; use more
     draws (`--n-random`) for the final paper numbers.
8. **Specificity**: D x D matrix, row c = how much every concept's score drops
   when concept c's top-k positions are ablated.

Scores for ablation are **average precision** (no threshold). Ablation can
shift every probability a little and break a fixed threshold even when the
probe still ranks chunks correctly, which would make F1 drops misleading.

## Outputs (`--out`)

| file | contents |
|---|---|
| `metrics.json` | summary scores, chosen settings, regularization search, split sizes |
| `per_concept.csv` | per concept: support, P/R/F1, avg precision, control AP, concentration, stability |
| `positions.json` | per concept: top positions and their weights |
| `ablation.csv` | per concept x k x mode: AP drop for top-k vs random |
| `specificity_top{k}.npy` | D x D cross-concept ablation matrix |
| `weights.npy` | (D, d) probe weights |
| `standardizer.npz` | train mean/std per position; the reconstruction step needs this |
| `split.json`, `label_counts.csv` | which chunk went where, positives per split |

## Hooks for the other pieces

- **Reconstruction / inverter**: the inverter needs raw vectors. Use
  `probe.ablation.ablate_raw(X, positions, train_mean)` with the mean from
  `standardizer.npz` so the ablation matches the probe's exactly.
- **Bag-of-words baseline**: reuse `split.json` and `probe.evaluate` so the
  baseline is scored on the same chunks with the same metrics.
- **Deep probe**: `TrainConfig(probe="mlp")` trains it, but its weights are
  not per-position scores, so for that probe positions come from ablation
  only.

## Known limits

- Early stopping uses the combined val loss over all concepts. Per-concept
  stopping would be slightly better for concepts that learn at very different
  speeds.
- Synthetic tests plant signal in a few positions. Real embeddings probably
  spread concepts out, so expect lower concentration and stability, and
  expect frozen and retrain ablation to disagree more.
