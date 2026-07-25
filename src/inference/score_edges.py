"""Score edge buckets with the trained baseline model."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
import torch

from src.model import EdgeBaselineMLP


def _load_metadata(path: str) -> dict:
    meta_path = Path(path)
    if not meta_path.exists():
        raise FileNotFoundError(f"Metadata file not found: {path}")
    return json.loads(meta_path.read_text(encoding="utf-8"))


def _load_model(model_path: str, input_dim: int, device: torch.device) -> EdgeBaselineMLP:
    path = Path(model_path)
    if not path.exists():
        raise FileNotFoundError(f"Model checkpoint not found: {model_path}")

    model = EdgeBaselineMLP(input_dim=input_dim)
    state = torch.load(path, map_location=device)
    model.load_state_dict(state)
    model.to(device)
    model.eval()
    return model


def score_edges(
    input_path: str,
    model_path: str = "models/edge_baseline.pt",
    metadata_path: str = "models/edge_baseline_metadata.json",
    output_path: str = "data/processed/inference/scored_edges.csv",
    device: str = "cuda",
) -> Path:
    data_path = Path(input_path)
    if not data_path.exists():
        raise FileNotFoundError(f"Input edge table not found: {input_path}")

    df = pd.read_csv(data_path, low_memory=False)
    metadata = _load_metadata(metadata_path)

    features = metadata["feature_columns"]
    threshold = float(metadata["threshold"])
    mean = torch.tensor(metadata["standardization_mean"], dtype=torch.float32)
    std = torch.tensor(metadata["standardization_std"], dtype=torch.float32)

    missing = [col for col in features if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required feature columns in input: {', '.join(missing)}")

    x = torch.tensor(df[features].fillna(0.0).astype(float).to_numpy(), dtype=torch.float32)
    x = (x - mean) / std

    runtime_device = torch.device(device if torch.cuda.is_available() and device.startswith("cuda") else "cpu")
    model = _load_model(model_path, input_dim=len(features), device=runtime_device)

    with torch.no_grad():
        probs = torch.sigmoid(model(x.to(runtime_device))).cpu().numpy()

    scored = df.copy()
    scored["anomaly_probability"] = probs
    scored["is_anomaly_pred"] = (scored["anomaly_probability"] >= threshold).astype(int)

    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    scored.to_csv(out_path, index=False)
    return out_path


def main() -> int:
    parser = argparse.ArgumentParser(description="Score edge buckets with trained baseline")
    parser.add_argument("--input", default="data/processed/splits/test_edges.csv")
    parser.add_argument("--model", default="models/edge_baseline.pt")
    parser.add_argument("--metadata", default="models/edge_baseline_metadata.json")
    parser.add_argument("--output", default="data/processed/inference/scored_edges.csv")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    output = score_edges(
        input_path=args.input,
        model_path=args.model,
        metadata_path=args.metadata,
        output_path=args.output,
        device=args.device,
    )

    print(f"Scored inference output written to: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
