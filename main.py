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
import sys
from pathlib import Path

from src.utils.config import load_config
from src.utils.logger import setup_logger, get_logger


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
        choices=["train", "infer", "dashboard"],
        default="dashboard",
        help="Operation mode: train (train model), infer (run inference), dashboard (launch UI)",
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
        default="data/synthetic",
        help="Path to data directory",
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
    
    # Validate paths
    config_path = Path(args.config)
    data_path = Path(args.data)
    
    if not data_path.exists():
        logger.warning(f"Data directory '{data_path}' does not exist.")
        logger.warning("Please generate or provide network traffic data first.")
    
    # Execute based on mode
    if args.mode == "train":
        logger.info("Training mode selected...")
        logger.warning("Training pipeline not yet implemented.")
        logger.info("Coming soon in the next development phase!")
        
    elif args.mode == "infer":
        logger.info("Inference mode selected...")
        logger.warning("Inference pipeline not yet implemented.")
        logger.info("Coming soon in the next development phase!")
        
    elif args.mode == "dashboard":
        logger.info("Dashboard mode selected...")
        logger.warning("Streamlit dashboard not yet implemented.")
        logger.info("Coming soon in the next development phase!")
        logger.info("")
        logger.info("To launch dashboard (when ready), run:")
        logger.info("  streamlit run dashboard/app.py")
    
    logger.info("")
    logger.success("Project initialization complete!")
    logger.info("Next steps: Generate synthetic data and build graph representation.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
