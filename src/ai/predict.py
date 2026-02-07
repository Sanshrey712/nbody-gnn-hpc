"""
Prediction/Inference Module for N-Body AI Models.

Provides:
- Single-step prediction
- Multi-step rollout
- Comparison with HPC simulation
"""

import torch
import torch.nn as nn
import numpy as np
from torch_geometric.data import Data
from typing import Optional, Tuple, List, Dict
from pathlib import Path
import h5py

from .model import NBodyGNN, NBodyLSTM, HybridGNNLSTM, create_model


class Predictor:
    """
    Inference engine for trained N-body models.
    """
    
    def __init__(self,
                 model: nn.Module,
                 model_path: Optional[str] = None,
                 device: str = None):
        """
        Initialize predictor.
        
        Args:
            model: The neural network model
            model_path: Path to saved model weights
            device: Inference device
        """
        if device is None:
            device = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
        
        self.device = device
        self.model = model.to(device)
        self.model.eval()
        
        if model_path:
            self.load_model(model_path)
        
        self.is_gnn = isinstance(model, NBodyGNN)
        self.is_lstm = isinstance(model, NBodyLSTM)
        self.is_hybrid = isinstance(model, HybridGNNLSTM)
    
    def load_model(self, model_path: str) -> None:
        """Load model weights from checkpoint."""
        checkpoint = torch.load(model_path, map_location=self.device)
        
        if 'model_state_dict' in checkpoint:
            self.model.load_state_dict(checkpoint['model_state_dict'])
        else:
            self.model.load_state_dict(checkpoint)
        
        self.model.eval()
        print(f"Loaded model from {model_path}")
    
    @torch.no_grad()
    def predict_single(self,
                       positions: np.ndarray,
                       velocities: np.ndarray,
                       masses: np.ndarray,
                       history: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Predict next state from current state.
        
        Args:
            positions: Current positions (N, 3)
            velocities: Current velocities (N, 3)
            masses: Particle masses (N,)
            history: For LSTM, history of states (seq_len, N, 6)
            
        Returns:
            Tuple of (predicted_positions, predicted_velocities)
        """
        n_particles = len(masses)
        
        if self.is_gnn:
            # Create graph data
            data = NBodyGNN.create_graph_data(
                positions, velocities, masses, device=self.device
            )
            
            # Predict
            pred = self.model(data)
            pred = pred.cpu().numpy()
            
        elif self.is_lstm:
            if history is None:
                # Create single-step "sequence"
                state = np.concatenate([positions, velocities], axis=-1)
                history = np.expand_dims(state, axis=0)  # (1, N, 6)
            
            # Convert to tensor
            x = torch.tensor(history, dtype=torch.float32).unsqueeze(0)  # (1, seq, N, 6)
            x = x.to(self.device)
            
            pred = self.model(x)
            pred = pred.squeeze(0).cpu().numpy()
            
        else:
            raise NotImplementedError("Model type not supported for single prediction")
        
        # Split prediction into position and velocity
        pred_pos = pred[:, :3]
        pred_vel = pred[:, 3:6]
        
        return pred_pos, pred_vel
    
    @torch.no_grad()
    def predict_rollout(self,
                        initial_positions: np.ndarray,
                        initial_velocities: np.ndarray,
                        masses: np.ndarray,
                        n_steps: int,
                        history: Optional[np.ndarray] = None) -> Dict[str, np.ndarray]:
        """
        Perform multi-step prediction rollout.
        
        Args:
            initial_positions: Starting positions (N, 3)
            initial_velocities: Starting velocities (N, 3)
            masses: Particle masses (N,)
            n_steps: Number of steps to predict
            history: For LSTM, initial history (seq_len, N, 6)
            
        Returns:
            Dictionary with trajectory arrays
        """
        n_particles = len(masses)
        
        # Initialize trajectory storage
        positions_traj = np.zeros((n_steps + 1, n_particles, 3))
        velocities_traj = np.zeros((n_steps + 1, n_particles, 3))
        
        positions_traj[0] = initial_positions
        velocities_traj[0] = initial_velocities
        
        # Current state
        pos = initial_positions.copy()
        vel = initial_velocities.copy()
        
        # History for LSTM
        if self.is_lstm and history is not None:
            current_history = history.copy()
        else:
            current_history = None
        
        # Rollout
        for step in range(n_steps):
            pred_pos, pred_vel = self.predict_single(
                pos, vel, masses, current_history
            )
            
            # Store prediction
            positions_traj[step + 1] = pred_pos
            velocities_traj[step + 1] = pred_vel
            
            # Update state
            pos = pred_pos
            vel = pred_vel
            
            # Update history for LSTM
            if self.is_lstm and current_history is not None:
                new_state = np.concatenate([pos, vel], axis=-1)
                current_history = np.concatenate([
                    current_history[1:], 
                    new_state[np.newaxis, ...]
                ], axis=0)
        
        return {
            'positions': positions_traj,
            'velocities': velocities_traj,
            'n_steps': n_steps,
            'n_particles': n_particles
        }
    
    @torch.no_grad()
    def batch_predict(self,
                      batch_positions: np.ndarray,
                      batch_velocities: np.ndarray,
                      masses: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Predict for a batch of states.
        
        Args:
            batch_positions: Batch of positions (B, N, 3)
            batch_velocities: Batch of velocities (B, N, 3)
            masses: Particle masses (N,)
            
        Returns:
            Tuple of (predicted_positions, predicted_velocities) both (B, N, 3)
        """
        batch_size = len(batch_positions)
        n_particles = len(masses)
        
        all_pred_pos = []
        all_pred_vel = []
        
        for i in range(batch_size):
            pred_pos, pred_vel = self.predict_single(
                batch_positions[i],
                batch_velocities[i],
                masses
            )
            all_pred_pos.append(pred_pos)
            all_pred_vel.append(pred_vel)
        
        return np.stack(all_pred_pos), np.stack(all_pred_vel)


def compare_with_hpc(predictor: Predictor,
                     hpc_trajectory: Dict,
                     start_step: int = 0,
                     n_prediction_steps: int = 100) -> Dict:
    """
    Compare AI predictions with HPC ground truth.
    
    Args:
        predictor: Trained predictor
        hpc_trajectory: HPC trajectory from checkpoint
        start_step: Step to start prediction from
        n_prediction_steps: Number of steps to predict
        
    Returns:
        Dictionary with comparison metrics
    """
    positions = hpc_trajectory['positions']
    velocities = hpc_trajectory['velocities']
    masses = hpc_trajectory['masses']
    
    # Get initial state
    init_pos = positions[start_step]
    init_vel = velocities[start_step]
    
    # AI rollout
    ai_result = predictor.predict_rollout(
        init_pos, init_vel, masses, n_prediction_steps
    )
    
    # Get corresponding HPC ground truth
    end_step = min(start_step + n_prediction_steps + 1, len(positions))
    hpc_pos = positions[start_step:end_step]
    hpc_vel = velocities[start_step:end_step]
    
    # Truncate AI predictions to match
    ai_pos = ai_result['positions'][:len(hpc_pos)]
    ai_vel = ai_result['velocities'][:len(hpc_vel)]
    
    # Compute errors
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


def load_predictor(model_type: str,
                   model_path: str,
                   model_config: Dict,
                   device: str = None) -> Predictor:
    """
    Load a trained predictor.
    
    Args:
        model_type: 'gnn', 'lstm', or 'hybrid'
        model_path: Path to saved model
        model_config: Model configuration dictionary
        device: Inference device
        
    Returns:
        Initialized Predictor
    """
    model = create_model(model_type, **model_config)
    return Predictor(model, model_path, device)
