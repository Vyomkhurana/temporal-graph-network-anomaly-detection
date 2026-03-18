"""Dataset contract loading and basic schema validation utilities."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import pandas as pd
import yaml


class DatasetContractError(Exception):
    """Raised when dataset contract or dataset schema is invalid."""


def load_dataset_contract(contract_path: str = "configs/dataset_contract.yaml") -> Dict:
    """Load dataset contract YAML and return it as a dictionary."""
    path = Path(contract_path)
    if not path.exists():
        raise DatasetContractError(f"Dataset contract not found: {contract_path}")

    with path.open("r", encoding="utf-8") as handle:
        contract = yaml.safe_load(handle) or {}

    return contract


def get_required_dataset_columns(contract: Dict) -> List[str]:
    """Extract required dataset column names from contract."""
    schema = contract.get("schema", {})
    required_mapping = schema.get("required_columns", {})

    required_columns = [value for value in required_mapping.values() if value]
    missing_placeholders = [key for key, value in required_mapping.items() if not value]

    if missing_placeholders:
        joined = ", ".join(missing_placeholders)
        raise DatasetContractError(
            "Dataset contract has unfilled required column mappings: " + joined
        )

    return required_columns


def validate_dataset_columns(df: pd.DataFrame, required_columns: List[str]) -> None:
    """Validate that required columns exist in the input dataframe."""
    missing = [column for column in required_columns if column not in df.columns]
    if missing:
        joined = ", ".join(missing)
        raise DatasetContractError(f"Dataset is missing required columns: {joined}")


def summarize_dataset(df: pd.DataFrame) -> Dict[str, int]:
    """Return lightweight dataset summary useful for ingestion logs."""
    return {
        "rows": int(df.shape[0]),
        "columns": int(df.shape[1]),
    }
