"""Run full offline pipeline from raw UNSW files to scored inference output."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.graph.build_node_mapping import build_node_mapping
from src.graph.build_pyg_edge_index import build_pyg_edge_index
from src.graph.build_temporal_edges import build_temporal_edges
from src.graph.run_graph_sanity_check import run_sanity_check, save_summary
from src.inference.score_edges import score_edges
from src.ingestion.normalize_unsw_nb15 import normalize_unsw
from src.training.build_temporal_batches import build_temporal_splits
from src.training.train_edge_baseline import train_baseline


def run_full_pipeline(
    contract_path: str = "configs/dataset_contract.yaml",
    columns_path: str = "configs/unsw_nb15_columns.txt",
    device: str = "cuda",
    train_epochs: int = 5,
    batch_size: int = 256,
    learning_rate: float = 1e-3,
    time_window: str = "1h",
) -> dict:
    """Execute complete offline training and inference workflow."""
    events_path = normalize_unsw(contract_path, columns_path, "data/processed/unsw_nb15_events.csv")
    temporal_edges_path = build_temporal_edges(str(events_path), "data/processed/unsw_nb15_temporal_edges.csv", window=time_window)
    node_map_path = build_node_mapping(str(temporal_edges_path), "data/processed/unsw_nb15_node_mapping.csv")
    pyg_edges_path = build_pyg_edge_index(str(temporal_edges_path), str(node_map_path), "data/processed/unsw_nb15_pyg_edges.csv")

    sanity = run_sanity_check(str(pyg_edges_path))
    sanity_path = save_summary(sanity, "data/processed/graph_sanity_summary.json")

    train_path, val_path, test_path = build_temporal_splits(str(pyg_edges_path), "data/processed/splits")

    metrics = train_baseline(
        train_path=str(train_path),
        val_path=str(val_path),
        test_path=str(test_path),
        epochs=train_epochs,
        batch_size=batch_size,
        lr=learning_rate,
        device=device,
        model_output_path="models/edge_baseline.pt",
        metadata_output_path="models/edge_baseline_metadata.json",
        metrics_output_path="data/processed/metrics/baseline_metrics.json",
    )

    scored_path = score_edges(
        input_path=str(test_path),
        model_path="models/edge_baseline.pt",
        metadata_path="models/edge_baseline_metadata.json",
        output_path="data/processed/inference/scored_edges.csv",
        device=device,
    )

    report = {
        "events_path": str(events_path),
        "temporal_edges_path": str(temporal_edges_path),
        "node_mapping_path": str(node_map_path),
        "pyg_edges_path": str(pyg_edges_path),
        "graph_sanity_summary_path": str(sanity_path),
        "split_paths": {
            "train": str(train_path),
            "val": str(val_path),
            "test": str(test_path),
        },
        "metrics": metrics,
        "scored_inference_path": str(scored_path),
    }

    report_path = Path("data/processed/reports/pipeline_run_report.json")
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    report["report_path"] = str(report_path)
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description="Run full temporal graph anomaly pipeline")
    parser.add_argument("--contract", default="configs/dataset_contract.yaml")
    parser.add_argument("--columns", default="configs/unsw_nb15_columns.txt")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--window", default="1h")
    args = parser.parse_args()

    result = run_full_pipeline(
        contract_path=args.contract,
        columns_path=args.columns,
        device=args.device,
        train_epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        time_window=args.window,
    )

    print("Pipeline completed successfully")
    print(f"Run report: {result['report_path']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
