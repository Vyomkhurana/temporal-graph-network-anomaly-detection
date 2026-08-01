"""
Main entry point for the Temporal Graph Network Lateral Movement Detection system.

This script orchestrates the entire ML pipeline:
1. Data ingestion and preprocessing
2. Temporal graph construction
3. Model training or loading
4. Inference and anomaly detection
5. Results visualization

Author: ML Security Team
Date: March 2026
"""

import argparse
import subprocess
import sys
from pathlib import Path

from src.inference import score_edges
from src.pipeline import run_full_pipeline
from src.training.train_temporal_gru import train_temporal_gru
from src.utils.config import load_config
from src.utils.logger import setup_logger, get_logger
from src.utils.service_config import (
    ServiceConfigError,
    load_services_config,
    validate_enabled_services,
)


def _launch_dashboard() -> int:
    """Launch the Streamlit dashboard in a subprocess."""
    dashboard_path = Path("dashboard/app.py")
    if not dashboard_path.exists():
        print("Dashboard app file not found at dashboard/app.py")
        return 1

    command = [
        sys.executable,
        "-m",
        "streamlit",
        "run",
        str(dashboard_path),
    ]
    return subprocess.call(command)


def main():
    """
    Main execution function for the TGN lateral movement detection system.
    """
    parser = argparse.ArgumentParser(
        description="Temporal Graph Network - Network Lateral Movement Detection"
    )
    
    # Add command-line arguments
    parser.add_argument(
        "--mode",
        type=str,
        choices=["train", "train-temporal", "infer", "dashboard"],
        default="dashboard",
        help="Operation mode: train (baseline pipeline), train-temporal (GRU), infer (run inference), dashboard (launch UI)",
    )
    
    parser.add_argument(
        "--config",
        type=str,
        default="configs/default.yaml",
        help="Path to configuration file",
    )
    
    parser.add_argument(
        "--data",
        type=str,
        default="data/raw/unsw_nb15",
        help="Path to data directory",
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=5,
        help="Training epochs for the selected model",
    )

    parser.add_argument(
        "--window",
        type=str,
        default="1h",
        help="Temporal aggregation window (e.g., 15min, 1h)",
    )

    parser.add_argument(
        "--sequence-length",
        type=int,
        default=8,
        help="Sequence length for the temporal GRU model",
    )
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logger(log_level="INFO")
    logger = get_logger()
    
    # Display welcome message
    logger.info("=" * 70)
    logger.info("Temporal Graph Network - Lateral Movement Detection System")
    logger.info("=" * 70)
    logger.info(f"Mode: {args.mode}")
    logger.info(f"Config: {args.config}")
    logger.info(f"Data directory: {args.data}")
    logger.info("=" * 70)
    
    # Load configuration
    try:
        config = load_config(args.config)
        logger.info(f"Configuration loaded successfully from: {args.config}")
        logger.info(f"Device: {config.get('system.device', 'cpu')}")
    except FileNotFoundError as e:
        logger.warning(f"Config file not found: {args.config}")
        logger.warning("Proceeding with default settings")
        config = None

    # Load and validate optional service connections only if enabled
    try:
        services_config = load_services_config("configs/services.yaml")
        validate_enabled_services(services_config)
        logger.info("Service configuration loaded successfully")
    except ServiceConfigError as exc:
        logger.warning(f"Service configuration issue: {exc}")
        logger.warning("Continuing with local-only execution")
    except FileNotFoundError:
        logger.warning("Service config file missing; continuing with local-only execution")
    
    # Validate paths
    config_path = Path(args.config)
    data_path = Path(args.data)
    
    if not data_path.exists():
        logger.warning(f"Data directory '{data_path}' does not exist.")
        logger.warning("Please generate or provide network traffic data first.")
    
    # Execute based on mode
    if args.mode == "train":
        logger.info("Training mode selected...")
        try:
            report = run_full_pipeline(
                contract_path="configs/dataset_contract.yaml",
                columns_path="configs/unsw_nb15_columns.txt",
                device=config.get("system.device", "cpu") if config else "cpu",
                train_epochs=args.epochs,
                time_window=args.window,
            )
            logger.success("Training pipeline finished successfully")
            logger.info(f"Run report: {report['report_path']}")
            logger.info(f"Inference output: {report['scored_inference_path']}")
        except Exception as exc:
            logger.error(f"Training pipeline failed: {exc}")
            return 1

    elif args.mode == "train-temporal":
        logger.info("Temporal training mode selected...")
        try:
            train_temporal_gru(
                train_path="data/processed/splits/train_edges.csv",
                val_path="data/processed/splits/val_edges.csv",
                test_path="data/processed/splits/test_edges.csv",
                epochs=args.epochs,
                device=config.get("system.device", "cpu") if config else "cpu",
                seq_len=args.sequence_length,
            )
            logger.success("Temporal GRU training finished successfully")
            logger.info("Temporal model artifacts were written to models/ and data/processed/")
        except Exception as exc:
            logger.error(f"Temporal training failed: {exc}")
            return 1
        
    elif args.mode == "infer":
        logger.info("Inference mode selected...")
        try:
            output = score_edges(
                input_path="data/processed/splits/test_edges.csv",
                model_path="models/edge_baseline.pt",
                metadata_path="models/edge_baseline_metadata.json",
                output_path="data/processed/inference/scored_edges.csv",
                device=config.get("system.device", "cpu") if config else "cpu",
            )
            logger.success("Inference completed successfully")
            logger.info(f"Scored output: {output}")
        except Exception as exc:
            logger.error(f"Inference failed: {exc}")
            logger.info("Train first using: python main.py --mode train")
            return 1
        
    elif args.mode == "dashboard":
        logger.info("Dashboard mode selected...")
        logger.info("Launching Streamlit dashboard...")
        return _launch_dashboard()
    
    logger.info("")
    logger.success("Execution completed.")
    logger.info("Use --mode train to refresh model artifacts and --mode infer to score new edge batches.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
