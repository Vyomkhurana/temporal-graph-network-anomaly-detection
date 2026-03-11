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
    
    # Display welcome message
    print("=" * 70)
    print("Temporal Graph Network - Lateral Movement Detection System")
    print("=" * 70)
    print(f"Mode: {args.mode}")
    print(f"Config: {args.config}")
    print(f"Data directory: {args.data}")
    print("=" * 70)
    print()
    
    # Validate paths
    config_path = Path(args.config)
    data_path = Path(args.data)
    
    if not data_path.exists():
        print(f"Warning: Data directory '{data_path}' does not exist.")
        print("Please generate or provide network traffic data first.")
    
    if not config_path.exists():
        print(f"Warning: Config file '{config_path}' does not exist.")
        print("Using default settings.")
    
    # Execute based on mode
    if args.mode == "train":
        print("Training mode selected...")
        print("Training pipeline not yet implemented.")
        print("Coming soon in the next development phase!")
        
    elif args.mode == "infer":
        print("Inference mode selected...")
        print("Inference pipeline not yet implemented.")
        print("Coming soon in the next development phase!")
        
    elif args.mode == "dashboard":
        print("Dashboard mode selected...")
        print("Streamlit dashboard not yet implemented.")
        print("Coming soon in the next development phase!")
        print()
        print("To launch dashboard (when ready), run:")
        print("  streamlit run dashboard/app.py")
    
    print()
    print("Project initialization complete!")
    print("Next steps: Generate synthetic data and build graph representation.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
