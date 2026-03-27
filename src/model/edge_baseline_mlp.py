"""Simple edge-level baseline model for anomaly prediction."""

from __future__ import annotations

import torch
from torch import nn


class EdgeBaselineMLP(nn.Module):
    """Small MLP baseline on aggregated edge features."""

    def __init__(self, input_dim: int = 4, hidden_dim: int = 32, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)
