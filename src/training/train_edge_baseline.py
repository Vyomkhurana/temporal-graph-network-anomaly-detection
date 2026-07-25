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
from src.training.evaluation import evaluate_at_threshold, format_metrics, select_best_threshold

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


def _get_standardization_stats(train_x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    mean = train_x.mean(axis=0, keepdims=True)
    std = train_x.std(axis=0, keepdims=True)
    std[std == 0.0] = 1.0
    return mean, std


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


def _save_model_artifacts(
    model: nn.Module,
    feature_columns: list[str],
    threshold: float,
    mean: np.ndarray,
    std: np.ndarray,
    model_output_path: str,
    metadata_output_path: str,
) -> Tuple[Path, Path]:
    model_path = Path(model_output_path)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), model_path)

    metadata = {
        "feature_columns": feature_columns,
        "threshold": float(threshold),
        "standardization_mean": mean.flatten().tolist(),
        "standardization_std": std.flatten().tolist(),
        "model_type": "EdgeBaselineMLP",
    }

    meta_path = Path(metadata_output_path)
    meta_path.parent.mkdir(parents=True, exist_ok=True)
    meta_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    return model_path, meta_path


def train_baseline(
    train_path: str,
    val_path: str,
    test_path: str | None = None,
    epochs: int = 5,
    batch_size: int = 256,
    lr: float = 1e-3,
    device: str = "cuda",
    model_output_path: str = "models/edge_baseline.pt",
    metadata_output_path: str = "models/edge_baseline_metadata.json",
    metrics_output_path: str = "data/processed/metrics/baseline_metrics.json",
) -> dict:
    train_df = _load_split(train_path)
    val_df = _load_split(val_path)
    test_df = _load_split(test_path) if test_path else None

    train_x, train_y = _build_xy(train_df)
    val_x, val_y = _build_xy(val_df)
    test_x, test_y = _build_xy(test_df) if test_df is not None else (None, None)

    mean, std = _get_standardization_stats(train_x)
    train_x_std = (train_x - mean) / std
    val_x_std = (val_x - mean) / std
    test_x_std = ((test_x - mean) / std) if test_x is not None else None

    dev = torch.device(device if torch.cuda.is_available() and device.startswith("cuda") else "cpu")

    train_loader = _to_loader(train_x_std, train_y, batch_size=batch_size, shuffle=True)
    val_loader = _to_loader(val_x_std, val_y, batch_size=batch_size, shuffle=False)
    test_loader = _to_loader(test_x_std, test_y, batch_size=batch_size, shuffle=False) if test_x_std is not None else None

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

    test_metrics = None
    if test_loader is not None:
        test_y_true, test_y_prob = _predict_probs(model, test_loader, dev)
        test_metrics = evaluate_at_threshold(test_y_true, test_y_prob, best_threshold)

    print("Validation metrics:")
    print(format_metrics(val_metrics))

    if test_metrics is not None:
        print("Test metrics:")
        print(format_metrics(test_metrics))

    full_metrics = {"validation": val_metrics}
    if test_metrics is not None:
        full_metrics["test"] = test_metrics

    metrics_path = _save_metrics(full_metrics, output_path=metrics_output_path)
    print(f"Metrics saved to: {metrics_path}")

    model_path, meta_path = _save_model_artifacts(
        model=model,
        feature_columns=FEATURE_COLUMNS,
        threshold=best_threshold,
        mean=mean,
        std=std,
        model_output_path=model_output_path,
        metadata_output_path=metadata_output_path,
    )
    print(f"Model checkpoint saved to: {model_path}")
    print(f"Model metadata saved to: {meta_path}")

    return full_metrics


def main() -> int:
    parser = argparse.ArgumentParser(description="Train baseline edge anomaly model")
    parser.add_argument("--train", default="data/processed/splits/train_edges.csv")
    parser.add_argument("--val", default="data/processed/splits/val_edges.csv")
    parser.add_argument("--test", default="data/processed/splits/test_edges.csv")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--model-output", default="models/edge_baseline.pt")
    parser.add_argument("--metadata-output", default="models/edge_baseline_metadata.json")
    parser.add_argument("--metrics-output", default="data/processed/metrics/baseline_metrics.json")
    args = parser.parse_args()

    train_baseline(
        train_path=args.train,
        val_path=args.val,
        test_path=args.test,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        device=args.device,
        model_output_path=args.model_output,
        metadata_output_path=args.metadata_output,
        metrics_output_path=args.metrics_output,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
