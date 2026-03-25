"""Run lightweight sanity checks on model-ready temporal graph edges."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


REQUIRED_COLUMNS = {
    "t",
    "src_node_id",
    "dst_node_id",
    "edge_event_count",
    "anomaly_event_count",
}


def run_sanity_check(input_path: str = "data/processed/unsw_nb15_pyg_edges.csv") -> dict:
    """Compute basic quality metrics for the model-ready edge table."""
    path = Path(input_path)
    if not path.exists():
        raise FileNotFoundError(f"PyG edge table not found: {input_path}")

    df = pd.read_csv(path, low_memory=False)
    missing = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {', '.join(sorted(missing))}")

    nodes = pd.concat([df["src_node_id"], df["dst_node_id"]], ignore_index=True)
    total_edges = int(len(df))
    total_events = float(df["edge_event_count"].sum())
    total_anomaly_events = float(df["anomaly_event_count"].sum())
    anomaly_rate = (total_anomaly_events / total_events) if total_events > 0 else 0.0

    summary = {
        "rows": total_edges,
        "unique_nodes": int(nodes.nunique()),
        "time_windows": int(df["t"].nunique()),
        "total_events": int(total_events),
        "total_anomaly_events": int(total_anomaly_events),
        "anomaly_event_rate": round(anomaly_rate, 6),
    }
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description="Run graph sanity check")
    parser.add_argument("--input", default="data/processed/unsw_nb15_pyg_edges.csv")
    args = parser.parse_args()

    summary = run_sanity_check(args.input)
    print("Graph sanity summary:")
    for key, value in summary.items():
        print(f"- {key}: {value}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
