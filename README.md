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

*Coming soon - project is under active development*

## Roadmap

- [x] Initialize project structure
- [ ] Generate synthetic network traffic data
- [ ] Build temporal graph representation
- [ ] Implement TGN model architecture
- [ ] Create training pipeline
- [ ] Develop inference system
- [ ] Build Streamlit dashboard
- [ ] Add real-time streaming capabilities
- [ ] Containerize with Docker

## Contributing

This is a learning and development project. Contributions and suggestions are welcome!

## License

MIT License

## Contact

For questions or discussions about this project, please open an issue.

---

**Note**: This project is for educational and research purposes. For production security systems, always consult with cybersecurity professionals.
