# AI-HPC: N-Body GNN Simulation

**AI-Accelerated N-Body Gravitational Simulation using Graph Neural Networks**

An academic project that uses HPC simulation to generate N-body gravitational trajectories and trains a Graph Neural Network (GNN) to predict future particle states.

## Optimal Configuration

**Designed for 32GB RAM / 12GB VRAM (e.g., RTX 3060/4070)**

- **Particles:** 2000
- **Simulations:** 200
- **Steps:** 400
- **Batch Size:** 8 (optimized for VRAM)
- **Epochs:** 100

## Quick Start

### Full Pipeline (Recommended)

```bash
# Runs: Data Generation → Train GNN → Evaluate
python scripts/run_demo.py
```

### Manual Steps

```bash
# 1. Generate simulation data (high fidelity for 32GB RAM)
python scripts/generate_data.py --particles 2000 --simulations 200 --steps 400

# 2. Train GNN (tuned for 12GB VRAM)
python scripts/train_model.py --epochs 100 --batch-size 8

# 3. Evaluate against HPC ground truth
python scripts/evaluate.py
```

## Features

- **N-Body Simulation**: Direct O(N²) + Barnes-Hut O(N log N) physics engine
- **JIT-Compiled**: Numba-accelerated for near-C performance
- **GNN Prediction**: Physics-aware Graph Neural Network with message-passing
- **Physics-Informed Loss**: Enforces energy and momentum conservation
- **Evaluation**: RMSE, MAE, energy/momentum conservation metrics

## Project Structure

```
AI-HPC/
├── src/
│   ├── hpc/                # Physics simulation
│   │   ├── nbody.py        # N-body engine (Numba JIT)
│   │   ├── barnes_hut.py   # Barnes-Hut octree
│   │   └── checkpoint.py   # HDF5 data I/O
│   ├── ai/                 # Neural network
│   │   ├── model.py        # GNN architecture
│   │   ├── train.py        # Training pipeline
│   │   ├── predict.py      # Inference & rollout
│   │   └── config.py       # Configuration
│   └── utils/
│       ├── visualization.py
│       └── metrics.py
├── scripts/
│   ├── run_demo.py         # Full pipeline
│   ├── generate_data.py    # Data generation
│   ├── train_model.py      # Training
│   └── evaluate.py         # Evaluation
├── data/                   # Generated HPC data
├── models/                 # Saved models
└── results/                # Plots & metrics
```

## Requirements

```
pip install -r requirements.txt
```

- Python 3.8+
- PyTorch + PyTorch Geometric (GPU recommended)
- NumPy, Numba, SciPy
- h5py, matplotlib

## License

MIT
