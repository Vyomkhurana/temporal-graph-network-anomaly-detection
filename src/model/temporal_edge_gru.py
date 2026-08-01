"""Temporal edge sequence model for anomaly detection."""

from __future__ import annotations

import torch
from torch import nn


class TemporalEdgeGRU(nn.Module):
    """GRU-based temporal classifier over windowed edge features."""

    def __init__(self, input_dim: int, hidden_dim: int = 64, num_layers: int = 1, dropout: float = 0.1):
        super().__init__()
        self.gru = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Expect input shaped [batch, seq_len, features]."""
        out, _ = self.gru(x)
        last_state = out[:, -1, :]
        return self.head(last_state).squeeze(-1)
