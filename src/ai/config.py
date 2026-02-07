from dataclasses import dataclass
from typing import Optional
import torch

@dataclass
class TrainingConfig:
    """Base configuration for training."""
    batch_size: int = 64
    workers: int = 4
    learning_rate: float = 1e-3
    hidden_dim: int = 128
    n_layers: int = 3
    sequence_length: int = 10
    early_stopping: int = 20
    epochs: int = 100
    k_neighbors: int = 20  # For GNN: Number of neighbors for graph construction
    device: str = 'auto'
    
    # Data Generation & Evaluation
    particles: int = 100
    simulations: int = 100
    steps: int = 500
    n_test_sims: int = 10

    @classmethod
    def get_device(cls, configured_device: str) -> str:
        if configured_device != 'auto':
            return configured_device
        
        if torch.cuda.is_available():
            return 'cuda'
        elif torch.backends.mps.is_available():
            return 'mps'
        return 'cpu'

@dataclass
class HPCConfig(TrainingConfig):
    """
    Configuration for HPC Node.
    Specs: 32 GB RAM, AMD Ryzen 9 5900XT (32 threads), NVIDIA RTX 3060 (12GB VRAM)
    """
    batch_size: int = 4    # Optimized for 12GB VRAM (Graph Memory is expensive)
    workers: int = 24      # Aggressive loading for 32-thread CPU
    hidden_dim: int = 256  # Safe capacity for 12GB VRAM
    n_layers: int = 4      # Balanced depth
    k_neighbors: int = 30  # Richer graph connectivity for HPC
    device: str = 'cuda'
    
    # Data Generation
    particles: int = 5000  # Large scale simulation
    simulations: int = 600 # Very diverse dataset
    steps: int = 600       # Long trajectories
    n_test_sims: int = 100 # Extensive evaluation

@dataclass
class MacAirConfig(TrainingConfig):
    """
    Configuration for Mac M4 Air.
    Specs: 16GB Unified Memory
    """
    batch_size: int = 32   # Conservative for 16GB shared memory
    workers: int = 4       # Efficient cores
    # Data Generation
    particles: int = 1000  # Moderate scale for Mac
    simulations: int = 200 # Good dataset size
    steps: int = 400       # Standard trajectories
    n_test_sims: int = 20  # Standard evaluation

def get_config(profile_name: str) -> TrainingConfig:
    configs = {
        'hpc': HPCConfig(),
        'mac': MacAirConfig(),
        'default': TrainingConfig()
    }
    
    # Auto-detect if 'auto' is selected or nothing provided
    if profile_name == 'auto':
        if torch.cuda.is_available():
            print("Auto-detected CUDA capability -> Using HPC profile")
            return HPCConfig()
        elif torch.backends.mps.is_available():
            print("Auto-detected MPS capability -> Using Mac profile")
            return MacAirConfig()
        else:
            print("No accelerator detected -> Using Default profile")
            return TrainingConfig()
            
    return configs.get(profile_name, TrainingConfig())
