# AI-HPC Hybrid N-Body Simulation

**AI-Accelerated N-Body Gravitational Simulation using HPC Checkpointing**

A hybrid high-performance computing and AI project that uses HPC to simulate N-body gravitational systems and trains neural networks to predict future states.

## Features

- **HPC Simulation**: Optimized N-body with Barnes-Hut O(N log N) algorithm
- **JIT Compilation**: Numba-accelerated computations for near-C performance
- **Parallel Processing**: Multiprocessing for CPU parallelization
- **AI Prediction**: Graph Neural Network (GNN) for physics-aware state prediction
- **Validation**: Comprehensive accuracy metrics and visualizations

## Quick Start

### 1. Automated Pipeline (Recommended)
This runs the entire workflow (Data Generation -> Training -> Evaluation) with hardware-optimized settings.

```bash
# For HPC Nodes (AMD + NVIDIA GPU)
# Optimized for 32 threads + 12GB VRAM + Batch Size 4 + Sparse GNN
python scripts/run_demo.py --profile hpc

# For Mac (M-Series)
# Optimized for Unified Memory + MPS + 4 cores
python scripts/run_demo.py --profile mac
```

### 2. Manual Execution
You can still run individual steps if needed:

```bash
# Generate Data (CPU Heavy)
python scripts/generate_data.py --particles 1000 --simulations 50

# Train GNN (GPU Heavy)
# Note: Use --profile hpc to load the safe VRAM settings automatically!
python scripts/train_model.py --profile hpc --epochs 100

# Evaluate
python scripts/evaluate.py --profile hpc
```

## Project Structure

```
AI-HPC/
├── src/
│   ├── hpc/                # HPC simulation
│   │   ├── nbody.py        # N-body physics engine
│   │   ├── barnes_hut.py   # Barnes-Hut tree algorithm
│   │   └── checkpoint.py   # State serialization
│   ├── ai/                 # Neural network
│   │   ├── model.py        # GNN architecture
│   │   ├── train.py        # Training pipeline
│   │   └── predict.py      # Inference
│   └── utils/              # Utilities
│       ├── visualization.py
│       └── metrics.py
├── scripts/                # Runnable scripts
├── data/                   # Generated HPC data
├── models/                 # Saved AI models
└── results/                # Plots and metrics
```

## Workflow

```
┌────────────────┐    ┌─────────────────┐    ┌────────────────┐
│  HPC Simulation│ -> │  Train AI Model │ -> │  AI Prediction │
│  (Barnes-Hut)  │    │  (GNN/LSTM)     │    │  (Fast!)       │
└────────────────┘    └─────────────────┘    └────────────────┘
```

## Requirements

- Python 3.8+
- NumPy, Numba (HPC)
- PyTorch, PyTorch Geometric (AI)
- Matplotlib (Visualization)

## Accuracy Metrics

The project computes:
- **RMSE**: Root Mean Square Error of position/velocity predictions
- **MAE**: Mean Absolute Error
- **Energy Conservation**: Validates physical plausibility
- **Trajectory Divergence**: Measures prediction quality over time

## License

MIT
