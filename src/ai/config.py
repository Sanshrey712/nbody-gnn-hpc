from dataclasses import dataclass
import torch

@dataclass
class TrainingConfig:
    """Training configuration for N-body GNN.
    
    Optimized for: 32 GB RAM, 12 GB VRAM.
    """
    
    # Training
    batch_size: int = 24
    learning_rate: float = 5e-4
    epochs: int = 200
    early_stopping: int = 30
    
    # Model
    hidden_dim: int = 256
    n_layers: int = 6
    k_neighbors: int = 40
    dropout: float = 0.1
    
    # Regularization
    weight_decay: float = 1e-4
    noise_std: float = 0.003  # Input noise injection during training
    
    # Data Generation
    particles: int = 200
    simulations: int = 300
    steps: int = 400
    dt: float = 0.01
    
    # Experiment
    test_size: float = 0.2
    n_test_sims: int = 10
    workers: int = 4
    sequence_length: int = 10

    @staticmethod
    def get_device() -> str:
        if torch.cuda.is_available():
            return 'cuda'
        elif torch.backends.mps.is_available():
            return 'mps'
        return 'cpu'
