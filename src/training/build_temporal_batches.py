"""Build chronological train/validation/test splits from PyG-ready edge table."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Tuple

import pandas as pd


REQUIRED_COLUMNS = {
    "t",
    "src_node_id",
    "dst_node_id",
    "edge_event_count",
    "anomaly_event_count",
    "total_bytes_sent",
    "total_bytes_received",
}


def _validate_columns(df: pd.DataFrame) -> None:
    missing = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    if missing:
        raise ValueError(f"Input table missing required columns: {', '.join(sorted(missing))}")


def _time_split_indices(unique_times: pd.Index, train_ratio: float, val_ratio: float) -> Tuple[int, int]:
    n = len(unique_times)
    train_end = max(1, int(n * train_ratio))
    val_end = max(train_end + 1, int(n * (train_ratio + val_ratio)))
    return train_end, min(val_end, n)


def build_temporal_splits(
    input_path: str = "data/processed/unsw_nb15_pyg_edges.csv",
    output_dir: str = "data/processed/splits",
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
) -> Tuple[Path, Path, Path]:
    """Split edges into train/val/test by ascending timestamp window."""
    source = Path(input_path)
    if not source.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    df = pd.read_csv(source, low_memory=False)
    _validate_columns(df)

    unique_times = pd.Index(sorted(df["t"].dropna().astype(int).unique()))
    if len(unique_times) < 3:
        raise ValueError("Need at least 3 unique time windows for train/val/test split")

    train_end, val_end = _time_split_indices(unique_times, train_ratio, val_ratio)

    train_times = set(unique_times[:train_end])
    val_times = set(unique_times[train_end:val_end])
    test_times = set(unique_times[val_end:])

    train_df = df[df["t"].isin(train_times)].copy()
    val_df = df[df["t"].isin(val_times)].copy()
    test_df = df[df["t"].isin(test_times)].copy()

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    train_path = out / "train_edges.csv"
    val_path = out / "val_edges.csv"
    test_path = out / "test_edges.csv"

    train_df.to_csv(train_path, index=False)
    val_df.to_csv(val_path, index=False)
    test_df.to_csv(test_path, index=False)

    return train_path, val_path, test_path


def main() -> int:
    parser = argparse.ArgumentParser(description="Build temporal train/val/test splits")
    parser.add_argument("--input", default="data/processed/unsw_nb15_pyg_edges.csv")
    parser.add_argument("--output-dir", default="data/processed/splits")
    parser.add_argument("--train-ratio", type=float, default=0.7)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    args = parser.parse_args()

    train_path, val_path, test_path = build_temporal_splits(
        input_path=args.input,
        output_dir=args.output_dir,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
    )

    print(f"Train split: {train_path}")
    print(f"Validation split: {val_path}")
    print(f"Test split: {test_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
