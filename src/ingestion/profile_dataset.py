"""Dataset profiling utilities for contract-driven ingestion."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Dict, List

from src.ingestion.dataset_contract import load_dataset_contract


def _resolve_input_files(contract: Dict[str, Any]) -> List[Path]:
    paths_cfg = contract.get("paths", {})
    candidates = [
        paths_cfg.get("train_file"),
        paths_cfg.get("validation_file"),
        paths_cfg.get("test_file"),
    ]
    files = [Path(path) for path in candidates if path]
    return files


def _read_first_row(csv_path: Path) -> List[str]:
    with csv_path.open("r", encoding="utf-8", errors="replace", newline="") as handle:
        reader = csv.reader(handle)
        first = next(reader)
    return first


def _count_rows(csv_path: Path) -> int:
    row_count = 0
    with csv_path.open("r", encoding="utf-8", errors="replace", newline="") as handle:
        reader = csv.reader(handle)
        for _ in reader:
            row_count += 1
    return row_count


def _infer_has_header(first_row: List[str]) -> bool:
    # Heuristic: if first row looks mostly numeric/IP-like tokens, treat as no header.
    alnum_count = 0
    for token in first_row:
        stripped = token.strip().lower().replace(".", "").replace("_", "")
        if stripped and stripped.isalpha():
            alnum_count += 1
    return alnum_count >= max(3, len(first_row) // 5)


def profile_files(contract_path: str) -> Dict[str, Any]:
    contract = load_dataset_contract(contract_path)
    metadata = contract.get("metadata", {})

    files = _resolve_input_files(contract)
    if not files:
        raise ValueError("No dataset files configured in contract paths")

    file_reports: List[Dict[str, Any]] = []
    for file_path in files:
        if not file_path.exists():
            file_reports.append(
                {
                    "file": str(file_path),
                    "exists": False,
                    "error": "File not found",
                }
            )
            continue

        first_row = _read_first_row(file_path)
        configured_has_header = metadata.get("has_header")
        inferred_has_header = _infer_has_header(first_row)
        has_header = configured_has_header if configured_has_header is not None else inferred_has_header

        total_rows = _count_rows(file_path)
        data_rows = total_rows - 1 if has_header and total_rows > 0 else total_rows

        file_reports.append(
            {
                "file": str(file_path),
                "exists": True,
                "total_rows": total_rows,
                "data_rows": data_rows,
                "column_count": len(first_row),
                "has_header": bool(has_header),
                "first_row_sample": first_row[:10],
            }
        )

    return {
        "contract_path": contract_path,
        "dataset_name": metadata.get("dataset_name"),
        "profile": file_reports,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Profile dataset files from contract")
    parser.add_argument("--contract", default="configs/dataset_contract.yaml", help="Path to dataset contract")
    parser.add_argument(
        "--output",
        default="data/processed/dataset_profile.json",
        help="Output JSON report path",
    )
    args = parser.parse_args()

    report = profile_files(args.contract)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(f"Dataset profile written to: {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
