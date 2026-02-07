"""
Training Pipeline for N-Body AI Models.

Features:
- Support for GNN, LSTM, and hybrid models
- Physics-informed loss functions
- Learning rate scheduling
- Early stopping
- Checkpoint saving
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torch_geometric.data import Data, Batch
import numpy as np
import h5py
import os
from pathlib import Path
from typing import Dict, Optional, Tuple, List
from tqdm import tqdm
import json
from datetime import datetime

from .model import NBodyGNN, NBodyLSTM, HybridGNNLSTM, create_model


class NBodyDataset(Dataset):
    """
    PyTorch Dataset for N-body training data.
    
    Loads data created by the checkpoint module.
    """
    
    def __init__(self, 
                 data_path: str,
                 sequence_length: int = 10,
                 model_type: str = 'lstm',
                 k_neighbors: Optional[int] = None):
        """
        Initialize dataset.
        
        Args:
            data_path: Path to HDF5 dataset
            sequence_length: Number of input timesteps
            model_type: 'gnn', 'lstm', or 'hybrid'
            k_neighbors: For GNN, number of neighbors per particle
        """
        self.data_path = data_path
        self.sequence_length = sequence_length
        self.model_type = model_type
        self.k_neighbors = k_neighbors
        
        # Load data
        with h5py.File(data_path, 'r') as f:
            self.inputs = f['inputs'][:]  # (n_samples, seq_len, n_particles, 6)
            self.targets = f['targets'][:]  # (n_samples, n_particles, 6)
            self.n_samples = f.attrs['n_samples']
        
        print(f"Loaded dataset: {self.n_samples} samples")
    
    def __len__(self):
        return self.n_samples
    
    def __getitem__(self, idx):
        input_seq = torch.tensor(self.inputs[idx], dtype=torch.float32)
        target = torch.tensor(self.targets[idx], dtype=torch.float32)
        
        return input_seq, target


class GNNDataset(Dataset):
    """
    Dataset that creates graph data for GNN models.
    """
    
    def __init__(self,
                 data_path: str,
                 sequence_length: int = 10,
                 k_neighbors: Optional[int] = None,
                 include_mass: bool = True):
        """
        Initialize GNN dataset.
        """
        self.data_path = data_path
        self.sequence_length = sequence_length
        self.k_neighbors = k_neighbors
        self.include_mass = include_mass
        
        with h5py.File(data_path, 'r') as f:
            self.inputs = f['inputs'][:]
            self.targets = f['targets'][:]
            self.n_samples = f.attrs['n_samples']
            
            # Try to load masses if available
            if 'masses' in f:
                self.masses = f['masses'][:]
            else:
                # Default unit masses
                n_particles = self.inputs.shape[2]
                self.masses = np.ones(n_particles)
        
        self.n_particles = self.inputs.shape[2]
        
        # Precompute edge indices for fully connected graph
        if k_neighbors is None or k_neighbors >= self.n_particles - 1:
            row = np.repeat(np.arange(self.n_particles), self.n_particles)
            col = np.tile(np.arange(self.n_particles), self.n_particles)
            mask = row != col
            self.edge_index = torch.tensor(
                np.stack([row[mask], col[mask]]), 
                dtype=torch.long
            )
        else:
            self.edge_index = None  # Will compute per sample based on positions
    
    def __len__(self):
        return self.n_samples
    
    def __getitem__(self, idx):
        # Get the last state from input sequence
        last_state = self.inputs[idx, -1]  # (n_particles, 6)
        positions = last_state[:, :3]
        velocities = last_state[:, 3:6]
        
        # Create node features
        if self.include_mass:
            x = np.concatenate([positions, velocities, self.masses.reshape(-1, 1)], axis=1)
        else:
            x = np.concatenate([positions, velocities], axis=1)
        
        x = torch.tensor(x, dtype=torch.float32)
        pos = torch.tensor(positions, dtype=torch.float32)
        target = torch.tensor(self.targets[idx], dtype=torch.float32)
        
        # Create graph
        if self.edge_index is None:
            # Compute k-nearest neighbors
            from scipy.spatial import distance_matrix
            dist = distance_matrix(positions, positions)
            row, col = [], []
            for i in range(self.n_particles):
                neighbors = np.argsort(dist[i])[1:self.k_neighbors+1]
                row.extend([i] * self.k_neighbors)
                col.extend(neighbors)
            edge_index = torch.tensor(np.stack([row, col]), dtype=torch.long)
        else:
            edge_index = self.edge_index
        
        return Data(x=x, edge_index=edge_index, pos=pos, y=target)


def collate_graphs(batch):
    """Custom collate function for batching graphs."""
    return Batch.from_data_list(batch)


class PhysicsInformedLoss(nn.Module):
    """
    Physics-informed loss function for N-body prediction.
    
    Combines:
    - Position error
    - Velocity error
    - Energy conservation error
    - Momentum conservation error
    """
    
    def __init__(self,
                 position_weight: float = 1.0,
                 velocity_weight: float = 1.0,
                 energy_weight: float = 0.1,
                 momentum_weight: float = 0.1):
        super().__init__()
        
        self.position_weight = position_weight
        self.velocity_weight = velocity_weight
        self.energy_weight = energy_weight
        self.momentum_weight = momentum_weight
        self.mse = nn.MSELoss()
    
    def forward(self, 
                pred: torch.Tensor, 
                target: torch.Tensor,
                masses: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, Dict]:
        """
        Compute physics-informed loss.
        
        Args:
            pred: Predicted state (batch, n_particles, 6)
            target: Target state (batch, n_particles, 6)
            masses: Particle masses for conservation laws
            
        Returns:
            Total loss, dict of component losses
        """
        # Split position and velocity
        pred_pos = pred[..., :3]
        pred_vel = pred[..., 3:6]
        target_pos = target[..., :3]
        target_vel = target[..., 3:6]
        
        # Position and velocity MSE
        pos_loss = self.mse(pred_pos, target_pos)
        vel_loss = self.mse(pred_vel, target_vel)
        
        # Conservation losses (optional)
        energy_loss = torch.tensor(0.0, device=pred.device)
        momentum_loss = torch.tensor(0.0, device=pred.device)
        
        if masses is not None and self.momentum_weight > 0:
            # Momentum conservation: sum(m * v) should be preserved
            pred_momentum = (masses.unsqueeze(-1) * pred_vel).sum(dim=1)
            target_momentum = (masses.unsqueeze(-1) * target_vel).sum(dim=1)
            momentum_loss = self.mse(pred_momentum, target_momentum)
        
        if masses is not None and self.energy_weight > 0:
            # Kinetic energy: 0.5 * sum(m * |v|^2)
            pred_ke = 0.5 * (masses * (pred_vel ** 2).sum(dim=-1)).sum(dim=-1)
            target_ke = 0.5 * (masses * (target_vel ** 2).sum(dim=-1)).sum(dim=-1)
            energy_loss = self.mse(pred_ke, target_ke)
        
        # Total loss
        total_loss = (self.position_weight * pos_loss + 
                      self.velocity_weight * vel_loss +
                      self.energy_weight * energy_loss +
                      self.momentum_weight * momentum_loss)
        
        losses = {
            'total': total_loss.item(),
            'position': pos_loss.item(),
            'velocity': vel_loss.item(),
            'energy': energy_loss.item(),
            'momentum': momentum_loss.item()
        }
        
        return total_loss, losses


class Trainer:
    """
    Training manager for N-body AI models.
    """
    
    def __init__(self,
                 model: nn.Module,
                 train_dataset: Dataset,
                 val_dataset: Optional[Dataset] = None,
                 model_dir: str = "./models",
                 device: str = None,
                 learning_rate: float = 1e-3,
                 batch_size: int = 32,
                 use_physics_loss: bool = True,
                 num_workers: int = 4):
        """
        Initialize trainer.
        """
        self.model = model
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.model_dir = Path(model_dir)
        self.model_dir.mkdir(parents=True, exist_ok=True)
        
        # Device
        if device is None:
            device = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
        self.device = device
        self.model.to(self.device)
        
        # Training settings
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.num_workers = num_workers
        
        # Create data loaders
        is_gnn = isinstance(train_dataset, GNNDataset)
        collate_fn = collate_graphs if is_gnn else None
        
        self.train_loader = DataLoader(
            train_dataset, 
            batch_size=batch_size, 
            shuffle=True,
            collate_fn=collate_fn,
            num_workers=num_workers,
            pin_memory=True if device != 'cpu' else False
        )
        
        if val_dataset:
            self.val_loader = DataLoader(
                val_dataset,
                batch_size=batch_size,
                shuffle=False,
                collate_fn=collate_fn,
                num_workers=num_workers,
                pin_memory=True if device != 'cpu' else False
            )
        else:
            self.val_loader = None
        
        # Loss and optimizer
        if use_physics_loss:
            self.criterion = PhysicsInformedLoss()
        else:
            self.criterion = nn.MSELoss()
        
        self.optimizer = optim.AdamW(
            model.parameters(), 
            lr=learning_rate,
            weight_decay=1e-5
        )
        
        # Learning rate scheduler
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, 
            mode='min', 
            factor=0.5, 
            patience=10
        )
        
        # Training history
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'learning_rate': []
        }
        
        self.best_val_loss = float('inf')
        self.is_gnn = is_gnn
    
    def train_epoch(self) -> float:
        """Train for one epoch."""
        self.model.train()
        total_loss = 0.0
        n_batches = 0
        
        for batch in self.train_loader:
            self.optimizer.zero_grad()
            
            if self.is_gnn:
                # GNN batch
                batch = batch.to(self.device)
                pred = self.model(batch)
                target = batch.y
            else:
                # LSTM batch
                inputs, target = batch
                inputs = inputs.to(self.device)
                target = target.to(self.device)
                pred = self.model(inputs)
            
            # Compute loss
            if isinstance(self.criterion, PhysicsInformedLoss):
                loss, _ = self.criterion(pred, target)
            else:
                loss = self.criterion(pred, target)
            
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            self.optimizer.step()
            
            total_loss += loss.item()
            n_batches += 1
        
        return total_loss / n_batches
    
    @torch.no_grad()
    def validate(self) -> float:
        """Validate the model."""
        if self.val_loader is None:
            return float('nan')
        
        self.model.eval()
        total_loss = 0.0
        n_batches = 0
        
        for batch in self.val_loader:
            if self.is_gnn:
                batch = batch.to(self.device)
                pred = self.model(batch)
                target = batch.y
            else:
                inputs, target = batch
                inputs = inputs.to(self.device)
                target = target.to(self.device)
                pred = self.model(inputs)
            
            if isinstance(self.criterion, PhysicsInformedLoss):
                loss, _ = self.criterion(pred, target)
            else:
                loss = self.criterion(pred, target)
            
            total_loss += loss.item()
            n_batches += 1
        
        return total_loss / n_batches
    
    def train(self,
              n_epochs: int = 100,
              early_stopping_patience: int = 30,
              save_every: int = 10,
              verbose: bool = True) -> Dict:
        """
        Train the model.
        
        Args:
            n_epochs: Number of training epochs
            early_stopping_patience: Epochs without improvement before stopping
            save_every: Save checkpoint every N epochs
            verbose: Print progress
            
        Returns:
            Training history
        """
        print(f"Training on {self.device}")
        print(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        
        patience_counter = 0
        
        pbar = tqdm(range(n_epochs), desc="Training")
        for epoch in pbar:
            # Train
            train_loss = self.train_epoch()
            self.history['train_loss'].append(train_loss)
            
            # Validate
            val_loss = self.validate()
            self.history['val_loss'].append(val_loss)
            
            # Learning rate
            current_lr = self.optimizer.param_groups[0]['lr']
            self.history['learning_rate'].append(current_lr)
            
            # Update scheduler
            self.scheduler.step(val_loss if not np.isnan(val_loss) else train_loss)
            
            # Update progress bar
            pbar.set_postfix({
                'train': f'{train_loss:.6f}',
                'val': f'{val_loss:.6f}',
                'lr': f'{current_lr:.2e}'
            })
            
            # Check for improvement
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.save_model('best_model.pt')
                patience_counter = 0
            else:
                patience_counter += 1
            
            # Early stopping
            if patience_counter >= early_stopping_patience:
                print(f"\nEarly stopping at epoch {epoch + 1}")
                break
            
            # Periodic save
            if (epoch + 1) % save_every == 0:
                self.save_model(f'checkpoint_epoch_{epoch + 1}.pt')
        
        # Save final model
        self.save_model('final_model.pt')
        
        # Save history
        self._save_history()
        
        return self.history
    
    def save_model(self, filename: str) -> str:
        """Save model checkpoint."""
        filepath = self.model_dir / filename
        
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_val_loss': self.best_val_loss,
            'history': self.history
        }
        
        torch.save(checkpoint, filepath)
        return str(filepath)
    
    def load_model(self, filename: str) -> None:
        """Load model checkpoint."""
        filepath = self.model_dir / filename
        checkpoint = torch.load(filepath, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.best_val_loss = checkpoint['best_val_loss']
        self.history = checkpoint['history']
    
    def _save_history(self) -> None:
        """Save training history to JSON."""
        history_path = self.model_dir / 'training_history.json'
        with open(history_path, 'w') as f:
            json.dump(self.history, f, indent=2)
