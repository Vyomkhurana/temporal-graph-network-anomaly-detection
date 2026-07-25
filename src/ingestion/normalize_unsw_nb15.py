"""Normalize UNSW-NB15 CSV partitions into a unified event table."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List

import pandas as pd

from src.ingestion.dataset_contract import load_dataset_contract


def _load_column_names(path: str) -> List[str]:
    file_path = Path(path)
    if not file_path.exists():
        raise FileNotFoundError(f"Column file not found: {path}")

    columns = [line.strip() for line in file_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    return columns


def _resolve_files(contract: Dict) -> List[str]:
    paths_cfg = contract.get("paths", {})
    return [path for path in [paths_cfg.get("train_file"), paths_cfg.get("validation_file"), paths_cfg.get("test_file")] if path]


def _normalize_frame(df: pd.DataFrame, source_file: str) -> pd.DataFrame:
    # Standardized event schema consumed by graph builder in next phase.
    out = pd.DataFrame()
    out["timestamp"] = pd.to_datetime(df["Stime"], unit="s", errors="coerce")
    out["src_ip"] = df["srcip"].astype(str)
    out["dst_ip"] = df["dstip"].astype(str)
    out["src_port"] = pd.to_numeric(df["sport"], errors="coerce")
    out["dst_port"] = pd.to_numeric(df["dsport"], errors="coerce")
    out["protocol"] = df["proto"].astype(str)
    out["event_type"] = df["state"].astype(str)
    out["bytes_sent"] = pd.to_numeric(df["sbytes"], errors="coerce")
    out["bytes_received"] = pd.to_numeric(df["dbytes"], errors="coerce")
    out["label"] = pd.to_numeric(df["label"], errors="coerce").fillna(0).astype(int)
    out["attack_cat"] = df["attack_cat"].astype(str)
    out["source_file"] = source_file
    return out


def normalize_unsw(contract_path: str, columns_path: str, output_path: str) -> Path:
    contract = load_dataset_contract(contract_path)
    files = _resolve_files(contract)
    columns = _load_column_names(columns_path)

    frames: List[pd.DataFrame] = []
    for file in files:
        df = pd.read_csv(file, header=None, names=columns, low_memory=False)
        frames.append(_normalize_frame(df, source_file=file))

    result = pd.concat(frames, ignore_index=True)
    result = result.dropna(subset=["timestamp", "src_ip", "dst_ip", "protocol"])

    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(out_path, index=False)
    return out_path


def main() -> int:
    parser = argparse.ArgumentParser(description="Normalize UNSW-NB15 raw files")
    parser.add_argument("--contract", default="configs/dataset_contract.yaml")
    parser.add_argument("--columns", default="configs/unsw_nb15_columns.txt")
    parser.add_argument("--output", default="data/processed/unsw_nb15_events.csv")
    args = parser.parse_args()

    output = normalize_unsw(args.contract, args.columns, args.output)
    print(f"Normalized events written to: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
