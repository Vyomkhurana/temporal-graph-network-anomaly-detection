"""Train and evaluate a simple edge-level baseline model."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from src.model import EdgeBaselineMLP
from src.training.evaluation import format_metrics, select_best_threshold

FEATURE_COLUMNS = [
    "edge_event_count",
    "anomaly_event_count",
    "total_bytes_sent",
    "total_bytes_received",
]


def _load_split(path: str) -> pd.DataFrame:
    file_path = Path(path)
    if not file_path.exists():
        raise FileNotFoundError(f"Split file not found: {path}")
    return pd.read_csv(file_path, low_memory=False)


def _build_xy(df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
    missing = [c for c in FEATURE_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required feature columns: {', '.join(missing)}")

    x = df[FEATURE_COLUMNS].fillna(0.0).astype(float).to_numpy()
    # Binary target: any anomalous event in this edge-time bucket.
    y = (df["anomaly_event_count"].fillna(0.0).astype(float) > 0.0).astype(float).to_numpy()
    return x, y


def _standardize(train_x: np.ndarray, other_x: np.ndarray) -> np.ndarray:
    mean = train_x.mean(axis=0, keepdims=True)
    std = train_x.std(axis=0, keepdims=True)
    std[std == 0.0] = 1.0
    return (other_x - mean) / std


def _to_loader(x: np.ndarray, y: np.ndarray, batch_size: int, shuffle: bool) -> DataLoader:
    x_tensor = torch.tensor(x, dtype=torch.float32)
    y_tensor = torch.tensor(y, dtype=torch.float32)
    ds = TensorDataset(x_tensor, y_tensor)
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle)


def _predict_probs(model: nn.Module, loader: DataLoader, device: torch.device) -> Tuple[np.ndarray, np.ndarray]:
    model.eval()
    all_probs = []
    all_labels = []
    with torch.no_grad():
        for features, labels in loader:
            features = features.to(device)
            logits = model(features)
            probs = torch.sigmoid(logits).cpu().numpy()
            all_probs.append(probs)
            all_labels.append(labels.numpy())

    return np.concatenate(all_labels), np.concatenate(all_probs)


def _save_metrics(metrics: dict, output_path: str = "data/processed/metrics/baseline_metrics.json") -> Path:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    return path


def train_baseline(
    train_path: str,
    val_path: str,
    epochs: int = 5,
    batch_size: int = 256,
    lr: float = 1e-3,
    device: str = "cuda",
) -> dict:
    train_df = _load_split(train_path)
    val_df = _load_split(val_path)

    train_x, train_y = _build_xy(train_df)
    val_x, val_y = _build_xy(val_df)

    train_x_std = _standardize(train_x, train_x)
    val_x_std = _standardize(train_x, val_x)

    dev = torch.device(device if torch.cuda.is_available() and device.startswith("cuda") else "cpu")

    train_loader = _to_loader(train_x_std, train_y, batch_size=batch_size, shuffle=True)
    val_loader = _to_loader(val_x_std, val_y, batch_size=batch_size, shuffle=False)

    model = EdgeBaselineMLP(input_dim=len(FEATURE_COLUMNS)).to(dev)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.BCEWithLogitsLoss()

    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        for features, labels in train_loader:
            features = features.to(dev)
            labels = labels.to(dev)

            optimizer.zero_grad()
            logits = model(features)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            running_loss += float(loss.item())

        avg_loss = running_loss / max(1, len(train_loader))
        print(f"Epoch {epoch}/{epochs} - train_loss: {avg_loss:.6f}")

    y_true, y_prob = _predict_probs(model, val_loader, dev)
    best_threshold, val_metrics = select_best_threshold(y_true, y_prob)
    val_metrics["best_threshold"] = float(best_threshold)

    print("Validation metrics:")
    print(format_metrics(val_metrics))

    metrics_path = _save_metrics(val_metrics)
    print(f"Metrics saved to: {metrics_path}")

    return val_metrics


def main() -> int:
    parser = argparse.ArgumentParser(description="Train baseline edge anomaly model")
    parser.add_argument("--train", default="data/processed/splits/train_edges.csv")
    parser.add_argument("--val", default="data/processed/splits/val_edges.csv")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    train_baseline(
        train_path=args.train,
        val_path=args.val,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        device=args.device,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
