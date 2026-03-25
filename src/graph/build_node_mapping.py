"""Build a deterministic node-to-id mapping from temporal edge data."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


REQUIRED_COLUMNS = {"src_ip", "dst_ip"}


def _validate_columns(df: pd.DataFrame) -> None:
    missing = [column for column in REQUIRED_COLUMNS if column not in df.columns]
    if missing:
        joined = ", ".join(sorted(missing))
        raise ValueError(f"Temporal edge file missing required columns: {joined}")


def build_node_mapping(
    input_path: str = "data/processed/unsw_nb15_temporal_edges.csv",
    output_path: str = "data/processed/unsw_nb15_node_mapping.csv",
) -> Path:
    """Create a stable mapping from IP/entity string to integer node_id."""
    in_path = Path(input_path)
    if not in_path.exists():
        raise FileNotFoundError(f"Temporal edge file not found: {input_path}")

    df = pd.read_csv(in_path, low_memory=False)
    _validate_columns(df)

    all_nodes = pd.concat([df["src_ip"], df["dst_ip"]], ignore_index=True).dropna().astype(str)
    unique_nodes = sorted(all_nodes.unique())

    mapping = pd.DataFrame(
        {
            "node_id": range(len(unique_nodes)),
            "node_value": unique_nodes,
        }
    )

    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    mapping.to_csv(out_path, index=False)
    return out_path


def main() -> int:
    parser = argparse.ArgumentParser(description="Build node mapping from temporal edge table")
    parser.add_argument("--input", default="data/processed/unsw_nb15_temporal_edges.csv")
    parser.add_argument("--output", default="data/processed/unsw_nb15_node_mapping.csv")
    args = parser.parse_args()

    output = build_node_mapping(args.input, args.output)
    print(f"Node mapping written to: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
