"""Train a GRU-based temporal edge anomaly model."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from src.model import TemporalEdgeGRU
from src.training.evaluation import evaluate_at_threshold, format_metrics, select_best_threshold

FEATURE_COLUMNS = [
    "edge_event_count",
    "total_bytes_sent",
    "total_bytes_received",
]
SEQUENCE_LENGTH = 8


def _load_split(path: str) -> pd.DataFrame:
    file_path = Path(path)
    if not file_path.exists():
        raise FileNotFoundError(f"Split file not found: {path}")
    return pd.read_csv(file_path, low_memory=False)


def _standardize(train_x: np.ndarray, other_x: np.ndarray) -> np.ndarray:
    mean = train_x.mean(axis=(0, 1), keepdims=True)
    std = train_x.std(axis=(0, 1), keepdims=True)
    std[std == 0.0] = 1.0
    return (other_x - mean) / std


def _build_sequences(df: pd.DataFrame, seq_len: int = SEQUENCE_LENGTH) -> Tuple[np.ndarray, np.ndarray]:
    required = {"src_node_id", "t", "anomaly_event_count", *FEATURE_COLUMNS}
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {', '.join(sorted(missing))}")

    frame = df.sort_values(["src_node_id", "t"])
    sequences: List[np.ndarray] = []
    labels: List[int] = []

    for _, group in frame.groupby("src_node_id", sort=False):
        values = group[FEATURE_COLUMNS].fillna(0.0).astype(float).to_numpy()
        group_labels = (group["anomaly_event_count"].fillna(0.0).astype(float) > 0.0).astype(int).to_numpy()
        if len(values) < seq_len:
            continue
        for start in range(0, len(values) - seq_len + 1):
            window = values[start : start + seq_len]
            label = int(group_labels[start : start + seq_len].max())
            sequences.append(window)
            labels.append(label)

    if not sequences:
        raise ValueError("No sequences could be built from the provided split")

    return np.stack(sequences), np.asarray(labels, dtype=np.float32)


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


def _score_sequences(
    model: nn.Module,
    df: pd.DataFrame,
    seq_len: int,
    threshold: float,
    device: torch.device,
) -> pd.DataFrame:
    frame = df.sort_values(["src_node_id", "t"])
    records: List[dict] = []

    with torch.no_grad():
        for src_node_id, group in frame.groupby("src_node_id", sort=False):
            values = group[FEATURE_COLUMNS].fillna(0.0).astype(float).to_numpy()
            labels = (group["anomaly_event_count"].fillna(0.0).astype(float) > 0.0).astype(int).to_numpy()
            if len(values) < seq_len:
                continue

            for start in range(0, len(values) - seq_len + 1):
                window = torch.tensor(values[start : start + seq_len], dtype=torch.float32, device=device).unsqueeze(0)
                probability = float(torch.sigmoid(model(window)).item())
                end_row = group.iloc[start + seq_len - 1]
                records.append(
                    {
                        "src_node_id": int(src_node_id),
                        "dst_node_id": int(end_row["dst_node_id"]),
                        "t": int(end_row["t"]),
                        "anomaly_probability": probability,
                        "is_anomaly_pred": int(probability >= threshold),
                        "is_anomaly_actual": int(labels[start : start + seq_len].max()),
                        "sequence_length": seq_len,
                    }
                )

    return pd.DataFrame(records)


def _save_artifacts(
    model: nn.Module,
    threshold: float,
    seq_len: int,
    model_output_path: str,
    metadata_output_path: str,
    metrics_output_path: str,
    scored_output_path: str | None,
    metrics: dict,
) -> Tuple[Path, Path, Path]:
    model_path = Path(model_output_path)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), model_path)

    metadata = {
        "model_type": "TemporalEdgeGRU",
        "sequence_length": seq_len,
        "feature_columns": FEATURE_COLUMNS,
        "threshold": float(threshold),
    }
    meta_path = Path(metadata_output_path)
    meta_path.parent.mkdir(parents=True, exist_ok=True)
    meta_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    metrics_path = Path(metrics_output_path)
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    scored_path = Path(scored_output_path) if scored_output_path else None
    if scored_path is not None:
        scored_path.parent.mkdir(parents=True, exist_ok=True)

    return model_path, meta_path, metrics_path


def train_temporal_gru(
    train_path: str,
    val_path: str,
    test_path: str | None = None,
    epochs: int = 5,
    batch_size: int = 256,
    lr: float = 1e-3,
    device: str = "cuda",
    seq_len: int = SEQUENCE_LENGTH,
    model_output_path: str = "models/temporal_edge_gru.pt",
    metadata_output_path: str = "models/temporal_edge_gru_metadata.json",
    metrics_output_path: str = "data/processed/metrics/temporal_gru_metrics.json",
    scored_output_path: str = "data/processed/inference/temporal_gru_scored_sequences.csv",
) -> dict:
    train_df = _load_split(train_path)
    val_df = _load_split(val_path)
    test_df = _load_split(test_path) if test_path else None

    train_x, train_y = _build_sequences(train_df, seq_len=seq_len)
    val_x, val_y = _build_sequences(val_df, seq_len=seq_len)
    test_x, test_y = _build_sequences(test_df, seq_len=seq_len) if test_df is not None else (None, None)

    train_x_std = _standardize(train_x, train_x)
    val_x_std = _standardize(train_x, val_x)
    test_x_std = _standardize(train_x, test_x) if test_x is not None else None

    dev = torch.device(device if torch.cuda.is_available() and device.startswith("cuda") else "cpu")
    train_loader = _to_loader(train_x_std, train_y, batch_size=batch_size, shuffle=True)
    val_loader = _to_loader(val_x_std, val_y, batch_size=batch_size, shuffle=False)
    test_loader = _to_loader(test_x_std, test_y, batch_size=batch_size, shuffle=False) if test_x_std is not None else None

    model = TemporalEdgeGRU(input_dim=len(FEATURE_COLUMNS)).to(dev)
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

    scored_sequences = _score_sequences(model, test_df if test_df is not None else val_df, seq_len, best_threshold, dev)
    scored_path = Path(scored_output_path)
    scored_path.parent.mkdir(parents=True, exist_ok=True)
    scored_sequences.to_csv(scored_path, index=False)

    print("Validation metrics:")
    print(format_metrics(val_metrics))
    if test_metrics is not None:
        print("Test metrics:")
        print(format_metrics(test_metrics))

    full_metrics = {"validation": val_metrics}
    if test_metrics is not None:
        full_metrics["test"] = test_metrics

    model_path, meta_path, metrics_path = _save_artifacts(
        model=model,
        threshold=best_threshold,
        seq_len=seq_len,
        model_output_path=model_output_path,
        metadata_output_path=metadata_output_path,
        metrics_output_path=metrics_output_path,
        scored_output_path=scored_output_path,
        metrics=full_metrics,
    )
    print(f"Model checkpoint saved to: {model_path}")
    print(f"Model metadata saved to: {meta_path}")
    print(f"Metrics saved to: {metrics_path}")
    print(f"Temporal scored output saved to: {scored_path}")

    return full_metrics


def main() -> int:
    parser = argparse.ArgumentParser(description="Train temporal GRU edge anomaly model")
    parser.add_argument("--train", default="data/processed/splits/train_edges.csv")
    parser.add_argument("--val", default="data/processed/splits/val_edges.csv")
    parser.add_argument("--test", default="data/processed/splits/test_edges.csv")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--sequence-length", type=int, default=SEQUENCE_LENGTH)
    parser.add_argument("--model-output", default="models/temporal_edge_gru.pt")
    parser.add_argument("--metadata-output", default="models/temporal_edge_gru_metadata.json")
    parser.add_argument("--metrics-output", default="data/processed/metrics/temporal_gru_metrics.json")
    args = parser.parse_args()

    train_temporal_gru(
        train_path=args.train,
        val_path=args.val,
        test_path=args.test,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        device=args.device,
        seq_len=args.sequence_length,
        model_output_path=args.model_output,
        metadata_output_path=args.metadata_output,
        metrics_output_path=args.metrics_output,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
