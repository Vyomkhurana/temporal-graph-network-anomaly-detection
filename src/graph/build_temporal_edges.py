"""Build temporal edge tables from normalized network events."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


REQUIRED_COLUMNS = {
    "timestamp",
    "src_ip",
    "dst_ip",
    "protocol",
    "event_type",
    "bytes_sent",
    "bytes_received",
    "label",
}


def _validate_columns(df: pd.DataFrame) -> None:
    missing = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    if missing:
        raise ValueError(f"Input events file is missing required columns: {', '.join(sorted(missing))}")


def build_temporal_edges(
    input_path: str = "data/processed/unsw_nb15_events.csv",
    output_path: str = "data/processed/unsw_nb15_temporal_edges.csv",
    window: str = "1H",
) -> Path:
    """Aggregate normalized events into time-windowed edge interactions."""
    in_path = Path(input_path)
    if not in_path.exists():
        raise FileNotFoundError(f"Normalized events file not found: {input_path}")

    df = pd.read_csv(in_path, low_memory=False)
    _validate_columns(df)

    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df = df.dropna(subset=["timestamp", "src_ip", "dst_ip", "protocol", "event_type"])

    # Normalize hour alias to avoid future pandas deprecation warnings.
    normalized_window = window.replace("H", "h")

    # Create a window key to model temporal graph evolution over fixed intervals.
    df["time_window_start"] = df["timestamp"].dt.floor(normalized_window)

    grouped = (
        df.groupby(["time_window_start", "src_ip", "dst_ip", "protocol", "event_type"], as_index=False)
        .agg(
            edge_event_count=("label", "size"),
            anomaly_event_count=("label", "sum"),
            total_bytes_sent=("bytes_sent", "sum"),
            total_bytes_received=("bytes_received", "sum"),
            first_seen=("timestamp", "min"),
            last_seen=("timestamp", "max"),
        )
        .sort_values(["time_window_start", "src_ip", "dst_ip", "protocol", "event_type"])
    )

    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    grouped.to_csv(out_path, index=False)
    return out_path


def main() -> int:
    parser = argparse.ArgumentParser(description="Build temporal edge table from normalized events")
    parser.add_argument("--input", default="data/processed/unsw_nb15_events.csv")
    parser.add_argument("--output", default="data/processed/unsw_nb15_temporal_edges.csv")
    parser.add_argument("--window", default="1H", help="Pandas offset alias, e.g. 5min, 15min, 1H")
    args = parser.parse_args()

    output = build_temporal_edges(args.input, args.output, args.window)
    print(f"Temporal edges written to: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
