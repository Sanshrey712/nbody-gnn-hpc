# Methodology: AI-HPC N-Body GNN

This document provides a detailed, end-to-end explanation of the N-body GNN simulation pipeline — from the physics simulation that generates training data, through the neural network architecture and training procedure, to the evaluation against ground truth.

---

## Table of Contents

1. [Problem Statement](#1-problem-statement)
2. [Physics Simulation (HPC)](#2-physics-simulation-hpc)
3. [Data Generation Pipeline](#3-data-generation-pipeline)
4. [Dataset Construction](#4-dataset-construction)
5. [Model Architecture](#5-model-architecture)
6. [Normalization Strategy](#6-normalization-strategy)
7. [Training Procedure](#7-training-procedure)
8. [Prediction & Inference](#8-prediction--inference)
9. [Evaluation](#9-evaluation)
10. [Configuration Reference](#10-configuration-reference)

---

## 1. Problem Statement

### What We're Solving

Traditional N-body gravitational simulation uses direct pairwise force computation (O(N²) per timestep). For N=200 particles over 400 timesteps, this requires ~16 million force calculations per simulation. While optimizations like Barnes-Hut reduce this to O(N log N), it remains computationally expensive for real-time or large-scale applications.

### Our Approach

We train a **Graph Neural Network (GNN)** to learn the mapping from a particle system's current state (positions + velocities + masses) to its next state. Once trained, the GNN predicts the next timestep in a single forward pass — orders of magnitude faster than solving the differential equations.

### Key Insight

Gravitational N-body dynamics are inherently **graph-structured**: each particle interacts with every other particle through pairwise gravitational forces. A GNN naturally encodes this by treating particles as **nodes** and their interactions as **edges** with learned message-passing.

---

## 2. Physics Simulation (HPC)

### 2.1 Gravitational Force Calculation

The gravitational acceleration on particle *i* due to all other particles:

```
a_i = Σ_{j≠i} G * m_j * (r_j - r_i) / |r_j - r_i|³
```

Where:
- `G = 6.674 × 10⁻¹¹` m³ kg⁻¹ s⁻² (gravitational constant)
- `m_j` = mass of particle j
- `r_i`, `r_j` = position vectors

**Softening**: To prevent singularities when particles approach each other, we add a softening parameter `ε = 10⁻⁹`:

```
r_softened = √(|r_j - r_i|² + ε²)
```

### Implementation (`src/hpc/nbody.py`)

The force calculation is JIT-compiled with **Numba** using `@jit(nopython=True, parallel=True, fastmath=True)`. This compiles the Python code to optimized machine code with:
- **Parallel execution** across particles using `prange`
- **Fast math** optimizations for floating-point operations
- **No Python overhead** (nopython mode)

For N > 500 particles, the **Barnes-Hut algorithm** (`src/hpc/barnes_hut.py`) is used instead, which reduces complexity to O(N log N) by grouping distant particles into octree cells and approximating their collective gravitational effect.

### 2.2 Leapfrog Integration

We use the **leapfrog (Störmer-Verlet)** integration scheme, a symplectic integrator that preserves the Hamiltonian structure of gravitational dynamics. This means it conserves energy over long timescales better than simpler methods (e.g., Euler).

Each timestep proceeds as:

```
1. v(t + dt/2) = v(t) + (dt/2) * a(t)           # Half-step velocity
2. x(t + dt)   = x(t) + dt * v(t + dt/2)         # Full-step position  
3. a(t + dt)   = compute_accelerations(x(t + dt)) # New accelerations
4. v(t + dt)   = v(t + dt/2) + (dt/2) * a(t + dt) # Complete velocity
```

### 2.3 Initial Conditions

| Parameter | Value | Description |
|-----------|-------|-------------|
| Positions | Uniform in [-5, 5]³ | `box_size = 10` |
| Velocities | Uniform in [-0.5, 0.5]³ | `0.1 × box_size` |
| Masses | Uniform in [10¹⁰, 10¹²] | Solar-ish mass range |
| Timestep (dt) | 0.001 | Per leapfrog step |
| Softening (ε) | 10⁻⁹ | Prevents force singularities |

---

## 3. Data Generation Pipeline

### 3.1 Shared Masses

**Critical design choice**: All simulations share the **same set of particle masses**, pre-generated once with a fixed random seed (`seed=42`). This ensures:

- The HDF5 dataset stores a single `(N_particles,)` mass array that is exact for every training sample
- The physics-informed loss (which uses masses) is mathematically exact, not an approximation
- Evaluation simulations use matching masses (same seed and range)

```python
# In generate_data.py
rng = np.random.RandomState(42)
shared_masses = rng.uniform(1e10, 1e12, n_particles).astype(np.float32)
```

Each simulation receives these shared masses via override:
```python
sim.masses = shared_masses.copy()
sim.accelerations = sim._compute_accelerations()  # Recompute with correct masses
```

### 3.2 Simulation Execution

- **300 independent simulations** are run, each with different random initial positions/velocities (seeds `42+i` for simulation `i`), but identical particle masses
- Each simulation runs for **400 timesteps** using leapfrog integration
- States (positions, velocities, accelerations, masses, time) are saved at every step
- Simulations are parallelized across CPU cores using `multiprocessing.Pool`
- A **checkpoint system** (`src/hpc/checkpoint.py`) saves each simulation to disk, allowing resumption if interrupted

### 3.3 Train/Validation Split

After all simulations complete:
- **80%** of simulations → training dataset (`data/train_dataset.h5`)
- **20%** of simulations → validation dataset (`data/val_dataset.h5`)

---

## 4. Dataset Construction

### 4.1 Input-Target Pairs

The training data is structured as **single-step prediction tasks**:

```
Input:  State sequence of length L ending at time t    → shape (L, N_particles, 6)
Target: State at time t+1                               → shape (N_particles, 6)
```

Where each state is `[position_x, position_y, position_z, velocity_x, velocity_y, velocity_z]`.

The **sequence length** is 10 timesteps, and the dataset is created with **stride 1** (sliding window over each simulation's trajectory). For 300 simulations × 400 steps:

```
Samples per sim = (400 - 10) / 1 = 390 windows
Total train samples = 240 sims × 390 = 93,600  (≈93,840 with rounding)
Total val samples   =  60 sims × 390 = 23,400  (≈23,460 with rounding)
```

### 4.2 GNNDataset (`src/ai/train.py`)

At training time, each sample is constructed as a **PyTorch Geometric `Data` object**:

1. **Extract last state** from the input sequence (most recent timestep)
2. **Normalize** positions, velocities, and masses (see Section 6)
3. **Construct node features**: `[norm_pos(3), norm_vel(3), norm_mass(1)]` = 7 dimensions
4. **Build edges**: Precomputed k-nearest-neighbor graph (k=40) from average particle positions
5. **Normalize target** using the same statistics as input
6. **Return**: `Data(x=features, edge_index=edges, pos=normalized_positions, y=normalized_target)`

### 4.3 Edge Construction

Rather than constructing a new k-NN graph for every training sample (which would be expensive), we **precompute edges once** from the average positions of 10 random samples:

```python
avg_positions = mean of 10 random samples' positions
tree = cKDTree(avg_positions)
_, indices = tree.query(avg_positions, k=41)  # 40 neighbors + self
```

This produces `200 × 40 = 8,000` directed edges, reused for all training samples. This is valid because:
- In 200-particle systems, the spatial structure is relatively stable across timesteps
- The GNN learns to use the edge features (distance, direction) rather than relying on exact edge connectivity
- This reduces dataset loading time from minutes to milliseconds

### 4.4 HDF5 Storage

The dataset is stored in HDF5 format:

```
train_dataset.h5
├── inputs:  (93840, 10, 200, 6)  float32  — input sequences
├── targets: (93840, 200, 6)      float32  — target states
└── masses:  (200,)               float32  — shared particle masses
```

---

## 5. Model Architecture

### 5.1 Overview

The model (`src/ai/model.py`) is a **residual GNN** that predicts **state deltas** (Δpos, Δvel), not absolute next states. The output is:

```
predicted_next_state = current_state + learned_delta
```

This residual design has two benefits:
1. The model starts with an identity mapping (decoder initialized to zero), so initial predictions are reasonable
2. The model only needs to learn the small physics-driven changes between timesteps

### 5.2 Architecture Details

```
Input: x = (N_particles, 7)  [norm_pos, norm_vel, norm_mass]

┌─────────────────────────────┐
│ Node Encoder                │
│ Linear(7 → 256) → LN → SiLU → Dropout(0.1) → Linear(256 → 256)
└──────────────┬──────────────┘
               │
               ▼
┌─────────────────────────────┐
│ Message Passing × 6 layers  │  ← with residual connections + LayerNorm
│                             │
│  For each layer:            │
│    1. Compute edge features │  [distance, direction(3), 1/r²] = 5 dims
│    2. Edge MLP:             │  [h_i, h_j, edge_attr] → hidden → message
│    3. Aggregate messages    │  (sum aggregation)
│    4. Node MLP:             │  [h_i, aggregated] → hidden → h_i_new
│    5. Residual:             │  h = LayerNorm(h + h_new)
└──────────────┬──────────────┘
               │
               ▼
┌─────────────────────────────┐
│ Decoder                     │
│ Linear(256 → 256) → SiLU → Dropout(0.1) → Linear(256 → 128) → SiLU → Linear(128 → 6)
│ (output initialized to zero)│
└──────────────┬──────────────┘
               │
               ▼
Output: current_state[:, :6] + delta  →  (N_particles, 6)
```

**Total parameters**: 2,550,150

### 5.3 Edge Features

For each edge (i → j), the model computes physics-inspired features:

| Feature | Formula | Dimension | Physical meaning |
|---------|---------|-----------|-----------------|
| Distance | `\|r_j - r_i\|` | 1 | Separation magnitude |
| Direction | `(r_j - r_i) / \|r_j - r_i\|` | 3 | Unit vector pointing i→j |
| Inverse square | `1 / \|r_j - r_i\|²` | 1 | Gravitational force scaling |

These features are computed from **normalized** positions, giving the model scale-invariant geometric information.

### 5.4 Message Passing

Each `ParticleInteractionLayer` performs:

1. **Message computation**: For each edge (i, j), concatenate `[h_i, h_j, edge_features]` and pass through an MLP
2. **Aggregation**: Sum all incoming messages for each node (mimics gravitational superposition)
3. **Node update**: Concatenate `[h_i, aggregated_messages]` and pass through another MLP

The `sum` aggregation is physically motivated — gravitational acceleration is the sum of pairwise forces, so the network naturally learns a force-like computation.

---

## 6. Normalization Strategy

### 6.1 Why Normalize

Raw simulation data has disparate scales:
- Positions: ~[-5, 5] (std ≈ 130-230)
- Velocities: ~[-0.5, 0.5] initially, growing to std ≈ 530-950
- Masses: ~[10¹⁰, 10¹²]

Without normalization, the MSE loss would be dominated by the largest-magnitude features, and gradient updates would be unstable.

### 6.2 Statistics Computation

Normalization statistics are computed from 500 random training samples:

```python
# For each sample, use the last state of the input sequence
all_states = stack of (500, N_particles, 6) arrays
flat = reshape to (500 × N_particles, 6)

state_mean = flat.mean(axis=0)  → shape (6,)  [mean_px, mean_py, mean_pz, mean_vx, mean_vy, mean_vz]
state_std  = flat.std(axis=0)   → shape (6,)
state_std  = max(state_std, 1e-6)  # Prevent division by zero
```

### 6.3 Where Normalization Is Applied

| Location | What | Transform |
|----------|------|-----------|
| `GNNDataset.__getitem__` | Input positions | `(pos - mean[:3]) / std[:3]` |
| `GNNDataset.__getitem__` | Input velocities | `(vel - mean[3:6]) / std[3:6]` |
| `GNNDataset.__getitem__` | Target (next state) | Same as inputs |
| `GNNDataset.__getitem__` | Masses | `mass / mass.mean()` |
| `Predictor._create_graph` | Inference inputs | Same as dataset |
| `Predictor.predict_single` | Model output | **Denormalize**: `pred * std + mean` |

### 6.4 Consistency Guarantees

- **Validation dataset** uses the training dataset's normalization statistics (passed via `external_norm_stats` parameter), ensuring both datasets are in the same normalized space
- **Normalization stats are saved in the model checkpoint**, so the predictor at inference time uses the exact same normalization as training
- **Mass normalization** (divide by mean) is applied identically in both the dataset and the predictor

---

## 7. Training Procedure

### 7.1 Loss Function

The loss is a **physics-informed combination** of 4 terms:

```
L_total = L_position + L_velocity + 0.1 × L_energy + 0.1 × L_momentum
```

#### Position & Velocity Loss (weight: 1.0 each)
Standard MSE in normalized space:
```
L_pos = MSE(pred_pos_norm, target_pos_norm)
L_vel = MSE(pred_vel_norm, target_vel_norm)
```

#### Energy Conservation Loss (weight: 0.1)
Encourages the model to conserve total kinetic energy between consecutive timesteps:
```
KE_pred   = Σ_i  0.5 × norm_m_i × |v_pred_i|²     (summed per graph)
KE_target = Σ_i  0.5 × norm_m_i × |v_target_i|²    (summed per graph)
L_energy  = MSE(KE_pred, KE_target)
```

**Note**: Masses are normalized internally (`m / m.mean()`) to prevent the raw mass scale (~10¹¹) from exploding the energy term.

#### Momentum Conservation Loss (weight: 0.1)
Encourages conservation of total linear momentum:
```
p_pred   = Σ_i  norm_m_i × v_pred_i     (3D vector, summed per graph)
p_target = Σ_i  norm_m_i × v_target_i
L_momentum = MSE(p_pred, p_target)
```

### 7.2 Optimizer & Scheduler

| Component | Choice | Parameters |
|-----------|--------|------------|
| Optimizer | AdamW | lr=5×10⁻⁴, weight_decay=10⁻⁴ |
| Scheduler | CosineAnnealingWarmRestarts | T₀=20 epochs, T_mult=2, η_min=10⁻⁶ |
| Gradient clipping | max_norm = 1.0 | Prevents gradient explosions |

The cosine schedule with warm restarts periodically resets the learning rate, helping the model escape local minima. The restart intervals double each time: epoch 20, 60, 140, etc.

### 7.3 Training Noise Injection

To improve robustness during multi-step rollout (where prediction errors accumulate), Gaussian noise is added to inputs during training:

```python
noise = randn_like(batch.x[:, :6]) × 0.003
batch.x[:, :6] += noise
batch.pos = batch.x[:, :3]  # Update positions to match
```

The noise standard deviation (σ=0.003) is in normalized space, so it represents ~0.3% perturbation relative to the data's standard deviation. This simulates the kind of imperfect inputs the model will see during rollout.

### 7.4 Early Stopping

Training stops if the **validation loss** does not improve for 30 consecutive epochs. The model checkpoint with the lowest validation loss is always saved separately as `best_model.pt`.

### 7.5 Checkpointing

Model checkpoints include:
- Model weights (`model_state_dict`)
- Optimizer state
- Scheduler state
- Best validation loss
- Full training history
- **Normalization statistics** (critical for inference)

Checkpoints are saved every 10 epochs and at the end of training.

### 7.6 Understanding Training vs Validation Loss

A large gap between training loss (~7000+) and validation loss (~10) is **expected** and not a bug. This is caused by **dropout**:

- The model has **14 dropout layers** (encoder, 6×2 message-passing, decoder) at 10% each
- During training (`model.train()`), dropout is active, making predictions noisier
- The **energy loss** squares velocities and sums over 200 particles, amplifying dropout noise quadratically
- During validation (`model.eval()`), dropout is off, giving clean predictions with much lower energy terms
- The validation loss is the true metric for model quality

---

## 8. Prediction & Inference

### 8.1 Single-Step Prediction

The `Predictor` class (`src/ai/predict.py`) handles inference:

```
Input: raw positions, velocities, masses (physical units)
  ↓
Step 1: Normalize inputs using saved norm_stats
Step 2: Normalize masses (divide by mean)
Step 3: Build node features [norm_pos, norm_vel, norm_mass]
Step 4: Build k-NN graph (k=40)
Step 5: Forward pass through model
Step 6: Denormalize output: pred_physical = pred_norm × std + mean
  ↓
Output: predicted positions, velocities (physical units)
```

### 8.2 Multi-Step Rollout

For trajectory prediction, the model is applied iteratively:

```
state_0 (raw) → predict_single → state_1 (raw)
state_1 (raw) → predict_single → state_2 (raw)
...
state_{n-1} (raw) → predict_single → state_n (raw)
```

Each step independently:
1. Normalizes inputs using the training statistics
2. Runs the model
3. Denormalizes the output

This ensures no accumulation of normalization artifacts between steps.

### 8.3 Graph Construction at Inference

Unlike training (where edges are precomputed), the predictor constructs a **fresh k-NN graph** for each prediction step using the current normalized positions. This allows the graph topology to adapt as particles move during rollout.

---

## 9. Evaluation

### 9.1 Test Simulations

Evaluation (`scripts/evaluate.py`) creates **10 new HPC simulations** with:
- Same shared masses as training (seed=42, uniform [10¹⁰, 10¹²])
- Different initial positions/velocities (different seeds)
- Same physics parameters (box_size=10, dt=0.001)

This tests generalization: the model sees new initial conditions but the same particle mass distribution.

### 9.2 Comparison Protocol

For each test simulation:
1. Run HPC simulation for 400 steps (ground truth)
2. Give the GNN the state at step 5 (after the input sequence length)
3. Roll out the GNN for the remaining steps
4. Compare GNN trajectory vs HPC trajectory

### 9.3 Metrics

| Metric | Definition | What it measures |
|--------|------------|-----------------|
| Position RMSE | `√(mean((ai_pos - hpc_pos)²))` | Trajectory accuracy |
| Velocity RMSE | `√(mean((ai_vel - hpc_vel)²))` | Dynamics accuracy |
| Energy conservation | Change in total KE over trajectory | Physics consistency |
| Momentum conservation | Change in total momentum over trajectory | Physics consistency |
| Error growth rate | RMSE vs timestep | Rollout stability |

### 9.4 Output

Evaluation produces:
- **Plots**: Trajectory comparisons, error growth curves, energy/momentum evolution
- **JSON metrics**: Numerical summary of all test simulations
- All saved to `results/` directory

---

## 10. Configuration Reference

All configurable parameters are defined in `src/ai/config.py`:

```python
@dataclass
class TrainingConfig:
    # Training
    batch_size: int = 24        # Graphs per training batch
    learning_rate: float = 5e-4 # Initial learning rate
    epochs: int = 200           # Maximum training epochs
    early_stopping: int = 30    # Patience (epochs without improvement)
    
    # Model architecture
    hidden_dim: int = 256       # Width of all hidden layers
    n_layers: int = 6           # Number of message-passing layers
    k_neighbors: int = 40       # k-NN graph connectivity
    dropout: float = 0.1        # Dropout rate (14 layers total)
    
    # Regularization
    weight_decay: float = 1e-4  # AdamW L2 regularization
    noise_std: float = 0.003    # Input noise injection (normalized space)
    
    # Data generation
    particles: int = 200        # N-body particles
    simulations: int = 300      # Independent simulation trajectories
    steps: int = 400            # Timesteps per simulation
    dt: float = 0.01            # Simulation timestep
    
    # Experiment
    test_size: float = 0.2      # Validation split fraction
    n_test_sims: int = 10       # Test simulations for evaluation
    workers: int = 4            # Dataloader workers
    sequence_length: int = 10   # Input sequence length
```

---

## Data Flow Diagram

```
┌──────────────┐     ┌────────────────┐     ┌──────────────┐
│ NBodySimulator│ ──→ │ generate_data.py│ ──→ │ HDF5 Dataset │
│ (Numba JIT)  │     │ (shared masses)│     │ (93K samples)│
└──────────────┘     └────────────────┘     └──────┬───────┘
                                                   │
                                                   ▼
┌──────────────┐     ┌────────────────┐     ┌──────────────┐
│  best_model  │ ←── │   Trainer      │ ←── │  GNNDataset  │
│  checkpoint  │     │ (physics loss) │     │ (normalize)  │
└──────┬───────┘     └────────────────┘     └──────────────┘
       │
       ▼
┌──────────────┐     ┌────────────────┐     ┌──────────────┐
│  Predictor   │ ──→ │  evaluate.py   │ ──→ │   Results    │
│ (denormalize)│     │ (AI vs HPC)    │     │ (plots/JSON) │
└──────────────┘     └────────────────┘     └──────────────┘
```
