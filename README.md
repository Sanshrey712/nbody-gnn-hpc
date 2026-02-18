# AI-HPC: N-Body GNN Simulation

**AI-Accelerated N-Body Gravitational Simulation using Graph Neural Networks**

An academic project that uses HPC N-body simulation to generate gravitational trajectories and trains a physics-informed Graph Neural Network (GNN) to predict future particle states at orders of magnitude faster than traditional simulation.

## Configuration

**Designed for 32GB RAM / 12GB VRAM (e.g., RTX 3060)**

| Parameter | Value | Description |
|-----------|-------|-------------|
| Particles | 200 | Number of gravitational bodies |
| Simulations | 300 | Independent simulation trajectories |
| Steps | 400 | Timesteps per simulation |
| Batch Size | 24 | Training batch size (fits 12GB VRAM) |
| Hidden Dim | 256 | GNN hidden layer width |
| Layers | 6 | Message-passing depth |
| Epochs | 200 | Max training epochs (early stopping at 30) |

## Quick Start

### Full Pipeline (Recommended)

```bash
# Runs: Clean → Generate Data → Train GNN → Evaluate
python scripts/run_demo.py
```

### Reuse Existing Data

```bash
# Skip data generation, only retrain and evaluate
python scripts/run_demo.py --skip-datagen
```

### Manual Steps

```bash
# 1. Generate simulation data
python scripts/generate_data.py --particles 200 --simulations 300 --steps 400

# 2. Train GNN with physics-informed loss
python scripts/train_model.py --physics-loss --epochs 200

# 3. Evaluate against HPC ground truth
python scripts/evaluate.py --n-test-sims 10 --particles 200 --steps 400
```

## Features

- **N-Body Simulation**: Direct O(N²) + Barnes-Hut O(N log N) physics engine
- **JIT-Compiled**: Numba-accelerated force calculations for near-C performance
- **GNN Prediction**: Physics-aware Graph Neural Network with 6-layer message-passing
- **Physics-Informed Loss**: Enforces energy and momentum conservation during training
- **Input Normalization**: Automatic normalization of positions, velocities, and masses
- **Shared Masses**: All simulations use identical particle masses for consistent physics loss
- **Training Noise**: Gaussian input noise injection (σ=0.003) for robust multi-step rollout
- **Per-Epoch Logging**: Clear train/val/energy/momentum breakdown after each epoch
- **Evaluation**: RMSE, MAE, energy conservation, and trajectory comparison plots

## Project Structure

```
AI-HPC/
├── src/
│   ├── hpc/                 # HPC Physics Simulation
│   │   ├── nbody.py         # N-body engine (Numba JIT, leapfrog integrator)
│   │   ├── barnes_hut.py    # Barnes-Hut octree for O(N log N)
│   │   └── checkpoint.py    # HDF5 dataset creation & I/O
│   ├── ai/                  # Neural Network
│   │   ├── model.py         # GNN architecture (ParticleInteractionLayer)
│   │   ├── train.py         # Training pipeline, dataset, physics loss
│   │   ├── predict.py       # Inference, rollout, & HPC comparison
│   │   └── config.py        # TrainingConfig dataclass
│   └── utils/
│       ├── visualization.py # Plotting utilities
│       └── metrics.py       # Error metrics
├── scripts/
│   ├── run_demo.py          # Full pipeline orchestrator
│   ├── generate_data.py     # Data generation with shared masses
│   ├── train_model.py       # Training entry point
│   └── evaluate.py          # Evaluation & plotting
├── data/                    # Generated HPC data (HDF5)
├── models/                  # Saved model checkpoints
├── results/                 # Evaluation plots & metrics
└── METHODOLOGY.md           # Detailed technical documentation
```

## Requirements

```bash
pip install -r requirements.txt
```

- Python 3.8+
- PyTorch ≥ 2.0 + PyTorch Geometric ≥ 2.3 (GPU recommended)
- NumPy, Numba, SciPy, h5py
- matplotlib, seaborn, tqdm

## Results

- **Position Error**: ~0.6% relative to simulation scale (152 units error over ~25,000 units range)
- **Velocity Error**: ~6.8% relative to scale (20,000 units error over ~300,000 units range)
- **Physics**: Accurately models violent gravitational collapse and particle ejection.

See [RESULTS_ANALYSIS.md](RESULTS_ANALYSIS.md) for a detailed breakdown of why these results are strong despite large absolute error values.

## Methodology

See [METHODOLOGY.md](METHODOLOGY.md) for a comprehensive explanation of the physics simulation, data pipeline, model architecture, training procedure, and evaluation.

## License

MIT
