"""Training loop for the probe.

Choices and why:
- Full-batch training. ~1000 chunks fit in memory easily, and full batch makes
  runs deterministic and easy to compare across ablation settings.
- pos_weight per concept (n_neg / n_pos). Most labels are "no", so without
  this the probe learns to say "no" to everything and still gets high accuracy.
- L2 by default, optional L1. L1 pushes most weights to exactly 0, which gives
  a cleaner list of candidate positions, but when two positions carry the same
  information it keeps one and drops the other more or less at random. Worth
  running both and comparing the position lists.
- Early stopping on validation loss, keeping the best weights.
"""

from __future__ import annotations

import copy
from dataclasses import asdict, dataclass

import numpy as np
import torch
from torch import nn

from .model import build


@dataclass
class TrainConfig:
    probe: str = "linear"          # "linear" or "mlp"
    lr: float = 1e-2
    l2: float = 1e-3               # penalty on weights (not bias)
    l1: float = 0.0
    max_epochs: int = 2000
    patience: int = 100            # epochs without val-loss improvement before stopping
    pos_weight_cap: float = 100.0
    seed: int = 0
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

    def to_dict(self):
        return asdict(self)


def pos_weights(Y: np.ndarray, cap: float) -> np.ndarray:
    pos = Y.sum(0)
    neg = len(Y) - pos
    return np.clip(neg / np.maximum(pos, 1), 1.0, cap).astype(np.float32)


def train_probe(Xtr, Ytr, Xval, Yval, cfg: TrainConfig) -> tuple[nn.Module, dict]:
    torch.manual_seed(cfg.seed)
    dev = torch.device(cfg.device)
    xt, yt = torch.from_numpy(Xtr).to(dev), torch.from_numpy(Ytr).to(dev)
    xv, yv = torch.from_numpy(Xval).to(dev), torch.from_numpy(Yval).to(dev)

    model = build(cfg.probe, Xtr.shape[1], Ytr.shape[1]).to(dev)
    loss_fn = nn.BCEWithLogitsLoss(
        pos_weight=torch.from_numpy(pos_weights(Ytr, cfg.pos_weight_cap)).to(dev)
    )
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr)

    best_val, best_state, best_epoch, since_best = float("inf"), None, 0, 0
    history = {"train_loss": [], "val_loss": []}

    for epoch in range(cfg.max_epochs):
        model.train()
        opt.zero_grad()
        data_loss = loss_fn(model(xt), yt)
        W = model.penalized_weights()
        loss = data_loss + cfg.l2 * W.pow(2).sum() + cfg.l1 * W.abs().sum()
        loss.backward()
        opt.step()

        model.eval()
        with torch.no_grad():
            val_loss = loss_fn(model(xv), yv).item()
        history["train_loss"].append(data_loss.item())
        history["val_loss"].append(val_loss)

        if val_loss < best_val - 1e-6:
            best_val, best_state, best_epoch, since_best = val_loss, copy.deepcopy(model.state_dict()), epoch, 0
        else:
            since_best += 1
            if since_best >= cfg.patience:
                break

    model.load_state_dict(best_state)
    model.eval()
    history.update(best_epoch=best_epoch, best_val_loss=best_val, epochs_run=epoch + 1)
    return model, history


DEFAULT_GRID = [  # (l2, l1)
    (1e-3, 0.0), (1e-2, 0.0), (1e-1, 0.0),
    (0.0, 3e-4), (0.0, 1e-3), (0.0, 3e-3),
]


def tune_regularization(Xtr, Ytr, Xval, Yval, cfg: TrainConfig, grid=DEFAULT_GRID) -> tuple[TrainConfig, list]:
    """Pick (l2, l1) by mean validation average precision.

    With ~50 positives per concept and hundreds of positions, the probe is
    data-starved and this setting moves results a lot, so it is chosen on
    validation data rather than fixed. Test data is never touched here.
    """
    from dataclasses import replace

    from .evaluate import average_precision

    results = []
    for l2, l1 in grid:
        c = replace(cfg, l2=l2, l1=l1)
        m, _ = train_probe(Xtr, Ytr, Xval, Yval, c)
        ap = float(np.nanmean(average_precision(predict_proba(m, Xval), Yval)))
        results.append({"l2": l2, "l1": l1, "val_avg_precision": ap})
    best = max(results, key=lambda r: r["val_avg_precision"])
    return replace(cfg, l2=best["l2"], l1=best["l1"]), results


@torch.no_grad()
def predict_proba(model: nn.Module, X: np.ndarray) -> np.ndarray:
    dev = next(model.parameters()).device
    return torch.sigmoid(model(torch.from_numpy(X).to(dev))).cpu().numpy()
