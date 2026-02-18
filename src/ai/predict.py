"""
Prediction/Inference Module for N-Body GNN Model.

Provides:
- Single-step prediction
- Multi-step rollout
- Comparison with HPC simulation
"""

import torch
import torch.nn as nn
import numpy as np
from torch_geometric.data import Data
from typing import Optional, Tuple, Dict
from pathlib import Path

from .model import NBodyGNN


class Predictor:
    """Inference engine for trained N-body GNN model."""
    
    def __init__(self,
                 model: nn.Module,
                 model_path: Optional[str] = None,
                 device: str = None,
                 k_neighbors: Optional[int] = None):
        if device is None:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        self.device = device
        self.model = model.to(device)
        self.model.eval()
        self.k_neighbors = k_neighbors
        self.norm_stats = None
        
        if model_path:
            self.load_model(model_path)
    
    def load_model(self, model_path: str) -> None:
        """Load model weights and normalization stats from checkpoint."""
        checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)
        if 'model_state_dict' in checkpoint:
            self.model.load_state_dict(checkpoint['model_state_dict'])
            # Load normalization stats if available
            if 'norm_stats' in checkpoint and checkpoint['norm_stats'] is not None:
                self.norm_stats = checkpoint['norm_stats']
                print(f"Loaded normalization stats")
        else:
            self.model.load_state_dict(checkpoint)
        self.model.eval()
        print(f"Loaded model from {model_path}")
    
    def _create_graph(self, positions, velocities, masses):
        """Create a PyG Data object from numpy arrays (applies normalization)."""
        n = len(masses)
        
        # Normalize if stats are available
        if self.norm_stats is not None:
            mean = self.norm_stats['state_mean']
            std = self.norm_stats['state_std']
            norm_pos = (positions - mean[:3]) / std[:3]
            norm_vel = (velocities - mean[3:6]) / std[3:6]
        else:
            norm_pos = positions
            norm_vel = velocities
        
        # Normalize masses
        norm_mass = (masses / masses.mean()).reshape(-1, 1)
        
        # Node features: normalized pos + vel + mass
        x = np.concatenate([norm_pos, norm_vel, norm_mass], axis=1)
        x = torch.tensor(x, dtype=torch.float32, device=self.device)
        
        # Build edges (k-NN or fully connected) — use normalized positions
        if self.k_neighbors is not None and self.k_neighbors < n - 1:
            from scipy.spatial import cKDTree
            tree = cKDTree(norm_pos)
            _, indices = tree.query(norm_pos, k=self.k_neighbors + 1)
            row = np.repeat(np.arange(n), self.k_neighbors)
            col = indices[:, 1:].flatten()
        else:
            row = np.repeat(np.arange(n), n)
            col = np.tile(np.arange(n), n)
            mask = row != col
            row, col = row[mask], col[mask]
        
        edge_index = torch.tensor(np.stack([row, col]), dtype=torch.long, device=self.device)
        pos = torch.tensor(norm_pos, dtype=torch.float32, device=self.device)
        
        return Data(x=x, edge_index=edge_index, pos=pos)
    
    @torch.no_grad()
    def predict_single(self,
                       positions: np.ndarray,
                       velocities: np.ndarray,
                       masses: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Predict next state from current state.
        
        Input/output are in physical (unnormalized) units.
        Internally normalizes for the model, then denormalizes the output.
        """
        data = self._create_graph(positions, velocities, masses)
        pred_norm = self.model(data).cpu().numpy()
        
        # Denormalize output back to physical units
        if self.norm_stats is not None:
            mean = self.norm_stats['state_mean']
            std = self.norm_stats['state_std']
            pred_pos = pred_norm[:, :3] * std[:3] + mean[:3]
            pred_vel = pred_norm[:, 3:6] * std[3:6] + mean[3:6]
        else:
            pred_pos = pred_norm[:, :3]
            pred_vel = pred_norm[:, 3:6]
        
        return pred_pos, pred_vel
    
    @torch.no_grad()
    def predict_rollout(self,
                        initial_positions: np.ndarray,
                        initial_velocities: np.ndarray,
                        masses: np.ndarray,
                        n_steps: int) -> Dict[str, np.ndarray]:
        """
        Multi-step prediction rollout.
        
        Returns:
            Dictionary with 'positions' and 'velocities' trajectory arrays
        """
        n_particles = len(masses)
        
        positions_traj = np.zeros((n_steps + 1, n_particles, 3))
        velocities_traj = np.zeros((n_steps + 1, n_particles, 3))
        
        positions_traj[0] = initial_positions
        velocities_traj[0] = initial_velocities
        
        pos = initial_positions.copy()
        vel = initial_velocities.copy()
        
        for step in range(n_steps):
            pred_pos, pred_vel = self.predict_single(pos, vel, masses)
            positions_traj[step + 1] = pred_pos
            velocities_traj[step + 1] = pred_vel
            pos = pred_pos
            vel = pred_vel
        
        return {
            'positions': positions_traj,
            'velocities': velocities_traj,
            'n_steps': n_steps,
            'n_particles': n_particles
        }


def compare_with_hpc(predictor: Predictor,
                     hpc_trajectory: Dict,
                     start_step: int = 0,
                     n_prediction_steps: int = 100) -> Dict:
    """Compare AI predictions with HPC ground truth."""
    positions = hpc_trajectory['positions']
    velocities = hpc_trajectory['velocities']
    masses = hpc_trajectory['masses']
    
    init_pos = positions[start_step]
    init_vel = velocities[start_step]
    
    ai_result = predictor.predict_rollout(
        init_pos, init_vel, masses, n_prediction_steps
    )
    
    end_step = min(start_step + n_prediction_steps + 1, len(positions))
    hpc_pos = positions[start_step:end_step]
    hpc_vel = velocities[start_step:end_step]
    
    ai_pos = ai_result['positions'][:len(hpc_pos)]
    ai_vel = ai_result['velocities'][:len(hpc_vel)]
    
    pos_error = np.sqrt(np.mean((ai_pos - hpc_pos) ** 2, axis=(1, 2)))
    vel_error = np.sqrt(np.mean((ai_vel - hpc_vel) ** 2, axis=(1, 2)))
    
    return {
        'ai_positions': ai_pos,
        'ai_velocities': ai_vel,
        'hpc_positions': hpc_pos,
        'hpc_velocities': hpc_vel,
        'position_rmse': pos_error,
        'velocity_rmse': vel_error,
        'mean_position_rmse': float(np.mean(pos_error)),
        'mean_velocity_rmse': float(np.mean(vel_error)),
        'final_position_rmse': float(pos_error[-1]),
        'final_velocity_rmse': float(vel_error[-1])
    }
