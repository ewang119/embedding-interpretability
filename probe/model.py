"""Probe models.

LinearProbe is the one in the proposal: one sigmoid neuron per concept, each
with d weights. As a single nn.Linear(d, D) trained with per-concept BCE, the
D neurons share no parameters, so this is the same as D separate logistic
regressions; it is just faster to train them together.

MLPProbe is the "deep network if we have time" option. Its weights cannot be
read as per-position scores, so positions for it come from ablation only
(see ablation.py), not from weights.py.
"""

from __future__ import annotations

import torch
from torch import nn


class LinearProbe(nn.Module):
    def __init__(self, d: int, n_concepts: int):
        super().__init__()
        self.linear = nn.Linear(d, n_concepts)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # returns logits (N, D)
        return self.linear(x)

    def weight_matrix(self):
        """(D, d) numpy array: row c is concept c's weight on every position."""
        return self.linear.weight.detach().cpu().numpy()

    def penalized_weights(self) -> torch.Tensor:
        return self.linear.weight


class MLPProbe(nn.Module):
    def __init__(self, d: int, n_concepts: int, hidden: int = 256, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d, hidden), nn.ReLU(), nn.Dropout(dropout), nn.Linear(hidden, n_concepts)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

    def weight_matrix(self):
        raise NotImplementedError("MLP weights are not per-position scores; use ablation instead")

    def penalized_weights(self) -> torch.Tensor:
        return self.net[0].weight


def build(kind: str, d: int, n_concepts: int, **kwargs) -> nn.Module:
    if kind == "linear":
        return LinearProbe(d, n_concepts)
    if kind == "mlp":
        return MLPProbe(d, n_concepts, **kwargs)
    raise ValueError(f"unknown probe kind {kind!r}")
