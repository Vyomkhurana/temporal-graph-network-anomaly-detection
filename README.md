# Real-Time Network Lateral Movement Detection using Temporal Graph Neural Networks

## Project Overview

This project implements a machine learning system that detects attacker lateral movement in enterprise networks using **Temporal Graph Neural Networks (TGN)**. The system models network activity as a dynamic temporal graph and identifies anomalous behaviors that indicate potential security threats.

## Problem Statement

Traditional SIEM (Security Information and Event Management) systems analyze isolated events and generate numerous false positives. Attackers typically move laterally between machines after initial compromise. Our detection system understands:

- **Relationships**: How machines, users, and IPs interact
- **Temporal patterns**: How these relationships evolve over time
- **Context**: Normal vs. anomalous behavior patterns

### Example Threat

A receptionist workstation initiating an SSH connection to a production database server would receive a very low probability score and trigger an alert.

## Technology Stack

- **Python 3.9+**: Core programming language
- **PyTorch**: Deep learning framework
- **PyTorch Geometric**: Graph neural network library
- **NetworkX**: Graph manipulation and analysis
- **Pandas**: Data processing
- **Streamlit**: Interactive dashboard
- **Docker**: Containerization (optional)
- **Kafka**: Real-time streaming (optional)
- **MLflow**: Experiment tracking (optional)

## Project Structure

```
├── src/                    # Source code
│   ├── ingestion/          # Data loading and preprocessing
│   ├── graph/              # Graph construction and management
│   ├── model/              # TGN model architecture
│   ├── training/           # Training pipeline
│   ├── inference/          # Prediction and scoring
│   └── streaming/          # Real-time processing
├── dashboard/              # Streamlit web interface
├── data/                   # Data storage
│   ├── raw/                # Original data files
│   ├── processed/          # Cleaned and transformed data
│   └── synthetic/          # Simulated network traffic
├── notebooks/              # Jupyter notebooks for experimentation
├── configs/                # Configuration files
├── README.md               # This file
├── requirements.txt        # Python dependencies
└── main.py                 # Entry point
```

## Graph Representation

### Nodes
- Machines (workstations, servers)
- IP addresses
- Users

### Edges
- Network connections
- Authentication events
- Protocol interactions

### Edge Features
- Timestamp
- Event type
- Metadata (bytes transferred, protocol, etc.)

## Setup Instructions

### Prerequisites
- Python 3.9 or higher
- Git
- (Optional) CUDA-capable GPU for faster training

### Installation

```bash
# Clone the repository
git clone <repository-url>
cd temporal-graph-network-anomaly-detection

# Create virtual environment
python -m venv venv

# Activate virtual environment
# On Windows:
venv\Scripts\activate
# On Linux/Mac:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

## Usage

### 1. Prepare Environment

```bash
python -m venv .venv
# On Windows
.venv\Scripts\activate
# On Linux/Mac
source .venv/bin/activate

pip install -r requirements.txt
```

### 2. Provide Dataset Files

Put these files into `data/raw/unsw_nb15/`:

- `UNSW-NB15_1.csv`
- `UNSW-NB15_2.csv`
- `UNSW-NB15_3.csv`

The default paths are already configured in `configs/dataset_contract.yaml`.

### 3. Run Full Offline Pipeline

```bash
python main.py --mode train --epochs 5 --window 1h
```

This command will:

1. Normalize raw UNSW data
2. Build temporal graph edges
3. Build node mapping
4. Build PyG-ready edge tables
5. Run graph sanity checks
6. Create temporal train/val/test splits
7. Train baseline anomaly model
8. Save model, metadata, and metrics
9. Score inference output for test split

### 4. Run Inference Only

```bash
python main.py --mode infer
```

### 5. Launch Dashboard

```bash
streamlit run dashboard/app.py
```

### 6. Useful Direct Commands

```bash
# Graph sanity check
python -m src.graph.run_graph_sanity_check --input data/processed/unsw_nb15_pyg_edges.csv

# Baseline training module
python -m src.training.train_edge_baseline --train data/processed/splits/train_edges.csv --val data/processed/splits/val_edges.csv --test data/processed/splits/test_edges.csv --epochs 5 --device cuda
```

## Implementation Status

- [x] Project initialization and module structure
- [x] Dataset contract and schema profiling
- [x] UNSW-NB15 normalization pipeline
- [x] Temporal graph edge construction
- [x] Node mapping and PyG-ready edge index generation
- [x] Graph sanity checker and JSON summary reporting
- [x] Chronological train/val/test split builder
- [x] Baseline edge anomaly model training
- [x] Threshold optimization and metric persistence
- [x] Model checkpoint and metadata persistence
- [x] Inference scoring pipeline
- [x] Streamlit operations dashboard
- [x] End-to-end orchestrated offline pipeline
- [ ] Temporal Graph Neural Network (TGN) model replacement for baseline
- [ ] Real-time streaming ingestion (Kafka)
- [ ] Containerized deployment and CI/CD

## Contributing

This is a learning and development project. Contributions and suggestions are welcome!

## License

MIT License

## Contact

For questions or discussions about this project, please open an issue.

---

**Note**: This project is for educational and research purposes. For production security systems, always consult with cybersecurity professionals.
