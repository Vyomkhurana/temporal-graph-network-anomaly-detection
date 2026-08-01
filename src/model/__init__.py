"""
Temporal Graph Neural Network model architectures.
"""

from .edge_baseline_mlp import EdgeBaselineMLP
from .temporal_edge_gru import TemporalEdgeGRU

__all__ = ["EdgeBaselineMLP", "TemporalEdgeGRU"]
