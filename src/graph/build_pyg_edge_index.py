"""Convert temporal edges and node mapping into PyG-ready edge index table."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


EDGE_REQUIRED = {
    "time_window_start",
    "src_ip",
    "dst_ip",
    "edge_event_count",
    "anomaly_event_count",
    "total_bytes_sent",
    "total_bytes_received",
}
MAPPING_REQUIRED = {"node_id", "node_value"}


def _validate_columns(df: pd.DataFrame, required: set[str], name: str) -> None:
    missing = [column for column in required if column not in df.columns]
    if missing:
        joined = ", ".join(sorted(missing))
        raise ValueError(f"{name} missing required columns: {joined}")


def build_pyg_edge_index(
    edges_path: str = "data/processed/unsw_nb15_temporal_edges.csv",
    node_mapping_path: str = "data/processed/unsw_nb15_node_mapping.csv",
    output_path: str = "data/processed/unsw_nb15_pyg_edges.csv",
) -> Path:
    """Create edge table with integer node indices for graph model ingestion."""
    edge_file = Path(edges_path)
    map_file = Path(node_mapping_path)
    if not edge_file.exists():
        raise FileNotFoundError(f"Temporal edge file not found: {edges_path}")
    if not map_file.exists():
        raise FileNotFoundError(f"Node mapping file not found: {node_mapping_path}")

    edges = pd.read_csv(edge_file, low_memory=False)
    mapping = pd.read_csv(map_file, low_memory=False)
    _validate_columns(edges, EDGE_REQUIRED, "Edge table")
    _validate_columns(mapping, MAPPING_REQUIRED, "Node mapping")

    node_to_id = dict(zip(mapping["node_value"].astype(str), mapping["node_id"]))

    out = edges.copy()
    out["src_node_id"] = out["src_ip"].astype(str).map(node_to_id)
    out["dst_node_id"] = out["dst_ip"].astype(str).map(node_to_id)
    out["time_window_start"] = pd.to_datetime(out["time_window_start"], errors="coerce")

    out = out.dropna(subset=["src_node_id", "dst_node_id", "time_window_start"])
    out["src_node_id"] = out["src_node_id"].astype(int)
    out["dst_node_id"] = out["dst_node_id"].astype(int)
    out["t"] = (out["time_window_start"].astype("int64") // 10**9).astype(int)

    model_ready = out[
        [
            "t",
            "src_node_id",
            "dst_node_id",
            "edge_event_count",
            "anomaly_event_count",
            "total_bytes_sent",
            "total_bytes_received",
        ]
    ].sort_values(["t", "src_node_id", "dst_node_id"])

    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    model_ready.to_csv(out_path, index=False)
    return out_path


def main() -> int:
    parser = argparse.ArgumentParser(description="Build PyG edge index table")
    parser.add_argument("--edges", default="data/processed/unsw_nb15_temporal_edges.csv")
    parser.add_argument("--mapping", default="data/processed/unsw_nb15_node_mapping.csv")
    parser.add_argument("--output", default="data/processed/unsw_nb15_pyg_edges.csv")
    args = parser.parse_args()

    output = build_pyg_edge_index(args.edges, args.mapping, args.output)
    print(f"PyG edge table written to: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
