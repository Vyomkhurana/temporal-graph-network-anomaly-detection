"""Evaluation helpers for threshold selection and metric reporting."""

from __future__ import annotations

from typing import Dict, Iterable, Tuple

import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score


def evaluate_at_threshold(y_true: np.ndarray, y_prob: np.ndarray, threshold: float) -> Dict[str, float]:
    """Compute classification metrics at a fixed probability threshold."""
    y_pred = (y_prob >= threshold).astype(int)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="binary", zero_division=0
    )

    metrics: Dict[str, float] = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
    }

    if len(np.unique(y_true)) > 1:
        metrics["auc"] = float(roc_auc_score(y_true, y_prob))
    else:
        metrics["auc"] = float("nan")

    return metrics


def select_best_threshold(y_true: np.ndarray, y_prob: np.ndarray, thresholds: Iterable[float] | None = None) -> Tuple[float, Dict[str, float]]:
    """Select the probability threshold that maximizes F1 on validation data."""
    if thresholds is None:
        thresholds = np.linspace(0.05, 0.95, 19)

    best_threshold = 0.5
    best_metrics: Dict[str, float] = {"accuracy": 0.0, "precision": 0.0, "recall": 0.0, "f1": 0.0}
    best_f1 = -1.0

    for threshold in thresholds:
        metrics = evaluate_at_threshold(y_true, y_prob, float(threshold))
        f1 = metrics["f1"]
        if f1 > best_f1:
            best_f1 = float(f1)
            best_threshold = float(threshold)
            best_metrics = metrics

    return best_threshold, best_metrics


def format_metrics(metrics: Dict[str, float]) -> str:
    """Format metrics dictionary into a readable multi-line string."""
    lines = []
    for key, value in metrics.items():
        if np.isnan(value):
            lines.append(f"- {key}: nan")
        else:
            lines.append(f"- {key}: {value:.6f}")
    return "\n".join(lines)
