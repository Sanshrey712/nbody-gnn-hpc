"""
Training Pipeline for N-Body GNN Model.

Features:
- GNN graph dataset with precomputed k-NN edges
- Physics-informed loss (position + velocity + energy + momentum conservation)
- Input normalization with statistics stored for inference
- Training noise injection for robust rollout
- Cosine annealing LR schedule
- Checkpoint saving with normalization stats
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

from .model import NBodyGNN


class GNNDataset(Dataset):
    """
    Dataset that creates graph data for GNN training.
    Precomputes k-NN edges from average positions for fast training.
    
    Computes and exposes normalization statistics (mean/std of features).
    """
    
    def __init__(self,
                 data_path: str,
                 sequence_length: int = 5,
                 k_neighbors: Optional[int] = None,
                 include_mass: bool = True,
                 external_norm_stats: Optional[Dict[str, np.ndarray]] = None):
        self.data_path = data_path
        self.sequence_length = sequence_length
        self.k_neighbors = k_neighbors
        self.include_mass = include_mass
        
        # Lazy loading setup
        self.file = None
        self.inputs = None
        self.targets = None
        
        # Read metadata and precompute edges
        with h5py.File(data_path, 'r') as f:
            self.n_samples = f.attrs['n_samples']
            self.n_particles = f['inputs'].shape[2]
            
            # Load masses (small array, safe to keep in memory)
            if 'masses' in f:
                self.masses = f['masses'][:]
            else:
                self.masses = np.ones(self.n_particles)
            
            # Use external normalization stats if provided (e.g. val uses train's stats)
            if external_norm_stats is not None:
                self.state_mean = external_norm_stats['state_mean']
                self.state_std = external_norm_stats['state_std']
                print(f"  Using external normalization stats")
            else:
                # Compute normalization statistics from a subset of data
                n_stat_samples = min(500, self.n_samples)
                stat_indices = np.random.choice(self.n_samples, n_stat_samples, replace=False)
                
                # Collect stats from inputs (last state) and targets
                all_states = []
                for idx in stat_indices:
                    last_state = f['inputs'][idx, -1]  # (n_particles, 6) — pos+vel
                    all_states.append(last_state)
                all_states = np.stack(all_states)  # (n_stat, n_particles, 6)
                
                # Compute per-feature mean and std
                flat = all_states.reshape(-1, 6)
                self.state_mean = flat.mean(axis=0).astype(np.float32)
                self.state_std = flat.std(axis=0).astype(np.float32)
                self.state_std = np.maximum(self.state_std, 1e-6)  # Avoid division by zero
            
            print(f"  Normalization stats — mean: {self.state_mean}, std: {self.state_std}")
            
            # Precompute edge indices
            if k_neighbors is None or k_neighbors >= self.n_particles - 1:
                # Fully connected graph
                row = np.repeat(np.arange(self.n_particles), self.n_particles)
                col = np.tile(np.arange(self.n_particles), self.n_particles)
                mask = row != col
                self.edge_index = torch.tensor(
                    np.stack([row[mask], col[mask]]), 
                    dtype=torch.long
                )
                print(f"Using fully connected graph ({self.edge_index.shape[1]} edges)")
            else:
                # Precompute k-NN from average positions
                print(f"Precomputing {k_neighbors}-NN edges...")
                
                n_samples_to_avg = min(10, self.n_samples)
                sample_indices = np.random.choice(self.n_samples, n_samples_to_avg, replace=False)
                
                avg_positions = np.zeros((self.n_particles, 3), dtype=np.float32)
                for idx in sample_indices:
                    last_state = f['inputs'][idx, -1]  # (n_particles, 6)
                    avg_positions += last_state[:, :3]
                avg_positions /= n_samples_to_avg
                
                from scipy.spatial import cKDTree
                tree = cKDTree(avg_positions)
                _, indices = tree.query(avg_positions, k=k_neighbors + 1)
                
                row = np.repeat(np.arange(self.n_particles), k_neighbors)
                col = indices[:, 1:].flatten()
                self.edge_index = torch.tensor(np.stack([row, col]), dtype=torch.long)
                
                print(f"  Created {self.edge_index.shape[1]} edges (precomputed, reused for all samples)")
        
        print(f"Dataset: {self.n_samples} samples, {self.n_particles} particles")

    def _ensure_file_open(self):
        """Ensure HDF5 file is open in current process (for multiprocessing)."""
        if self.file is None:
            self.file = h5py.File(self.data_path, 'r')
            self.inputs = self.file['inputs']
            self.targets = self.file['targets']
    
    def __del__(self):
        if hasattr(self, 'file') and self.file is not None:
            self.file.close()
    
    def __len__(self):
        return self.n_samples
    
    def __getitem__(self, idx):
        self._ensure_file_open()
        # Get the last state from input sequence
        last_state = self.inputs[idx, -1]  # (n_particles, 6)
        positions = last_state[:, :3]
        velocities = last_state[:, 3:6]
        
        # Normalize positions and velocities
        norm_pos = (positions - self.state_mean[:3]) / self.state_std[:3]
        norm_vel = (velocities - self.state_mean[3:6]) / self.state_std[3:6]
        
        # Create node features: norm_pos + norm_vel + normalized_mass
        if self.include_mass:
            norm_mass = (self.masses / self.masses.mean()).reshape(-1, 1).astype(np.float32)
            x = np.concatenate([norm_pos, norm_vel, norm_mass], axis=1)
        else:
            x = np.concatenate([norm_pos, norm_vel], axis=1)
        
        x = torch.tensor(x, dtype=torch.float32)
        pos = torch.tensor(norm_pos, dtype=torch.float32)  # Use normalized pos for edges too
        
        # Normalize target the same way
        raw_target = self.targets[idx]  # (n_particles, 6)
        norm_target = np.empty_like(raw_target, dtype=np.float32)
        norm_target[:, :3] = (raw_target[:, :3] - self.state_mean[:3]) / self.state_std[:3]
        norm_target[:, 3:6] = (raw_target[:, 3:6] - self.state_mean[3:6]) / self.state_std[3:6]
        target = torch.tensor(norm_target, dtype=torch.float32)
        
        return Data(x=x, edge_index=self.edge_index, pos=pos, y=target)
    
    def get_normalization_stats(self) -> Dict[str, np.ndarray]:
        """Return normalization statistics for use during inference."""
        return {
            'state_mean': self.state_mean,
            'state_std': self.state_std,
        }
    
    def get_masses_tensor(self) -> torch.Tensor:
        """Return masses as a tensor for physics loss."""
        return torch.tensor(self.masses, dtype=torch.float32)


def collate_graphs(batch):
    """Custom collate function for batching graphs."""
    return Batch.from_data_list(batch)


class PhysicsInformedLoss(nn.Module):
    """
    Physics-informed loss combining position/velocity MSE
    with energy and momentum conservation terms.
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
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor,
                masses: Optional[torch.Tensor] = None,
                batch_index: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, Dict]:
        """
        Compute physics-informed loss.
        
        Args:
            pred: Predicted states (N_total, 6) — pos(3) + vel(3)
            target: Target states (N_total, 6)
            masses: Particle masses (N_total,) — MUST be provided for physics loss
            batch_index: Batch assignment per node (from PyG Batch)
        """
        # Split position and velocity
        pred_pos = pred[..., :3]
        pred_vel = pred[..., 3:6]
        target_pos = target[..., :3]
        target_vel = target[..., 3:6]
        
        pos_loss = self.mse(pred_pos, target_pos)
        vel_loss = self.mse(pred_vel, target_vel)
        
        energy_loss = torch.tensor(0.0, device=pred.device)
        momentum_loss = torch.tensor(0.0, device=pred.device)
        
        
        if masses is not None and batch_index is not None:
            # Normalize masses to prevent loss explosion (raw masses are ~1e10-1e12)
            mass_scale = masses.mean()
            if mass_scale > 0:
                norm_masses = masses / mass_scale
            else:
                norm_masses = masses
            
            n_graphs = batch_index.max().item() + 1
            
            # Compute per-graph momentum conservation (using normalized masses)
            if self.momentum_weight > 0:
                pred_momentum = norm_masses.unsqueeze(-1) * pred_vel     # (N_total, 3)
                target_momentum = norm_masses.unsqueeze(-1) * target_vel
                
                # Sum momentum per graph in the batch
                pred_total_mom = torch.zeros(n_graphs, 3, device=pred.device)
                target_total_mom = torch.zeros(n_graphs, 3, device=pred.device)
                pred_total_mom.scatter_add_(0, batch_index.unsqueeze(-1).expand_as(pred_momentum), pred_momentum)
                target_total_mom.scatter_add_(0, batch_index.unsqueeze(-1).expand_as(target_momentum), target_momentum)
                
                momentum_loss = self.mse(pred_total_mom, target_total_mom)
            
            # Compute per-graph kinetic energy conservation (using normalized masses)
            if self.energy_weight > 0:
                pred_ke = 0.5 * norm_masses * (pred_vel ** 2).sum(dim=-1)    # (N_total,)
                target_ke = 0.5 * norm_masses * (target_vel ** 2).sum(dim=-1)
                
                # Sum KE per graph
                pred_total_ke = torch.zeros(n_graphs, device=pred.device)
                target_total_ke = torch.zeros(n_graphs, device=pred.device)
                pred_total_ke.scatter_add_(0, batch_index, pred_ke)
                target_total_ke.scatter_add_(0, batch_index, target_ke)
                
                energy_loss = self.mse(pred_total_ke, target_total_ke)
        
        total_loss = (self.position_weight * pos_loss + 
                      self.velocity_weight * vel_loss +
                      self.energy_weight * energy_loss +
                      self.momentum_weight * momentum_loss)
        
        losses = {
            'total': total_loss.item(),
            'position': pos_loss.item(),
            'velocity': vel_loss.item(),
            'energy': energy_loss.item() if isinstance(energy_loss, torch.Tensor) else energy_loss,
            'momentum': momentum_loss.item() if isinstance(momentum_loss, torch.Tensor) else momentum_loss
        }
        
        return total_loss, losses


class Trainer:
    """Training manager for N-body GNN model."""
    
    def __init__(self,
                 model: nn.Module,
                 train_dataset: Dataset,
                 val_dataset: Optional[Dataset] = None,
                 model_dir: str = "./models",
                 device: str = None,
                 learning_rate: float = 5e-4,
                 batch_size: int = 24,
                 use_physics_loss: bool = True,
                 num_workers: int = 2,
                 weight_decay: float = 1e-4,
                 noise_std: float = 0.003,
                 n_epochs: int = 200):
        self.model = model
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.model_dir = Path(model_dir)
        self.model_dir.mkdir(parents=True, exist_ok=True)
        
        # Device
        if device is None:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.device = device
        self.model.to(self.device)
        
        # Training settings
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.num_workers = num_workers
        self.noise_std = noise_std
        
        # Get masses from dataset for physics loss
        base_dataset = train_dataset
        if hasattr(train_dataset, 'dataset'):
            base_dataset = train_dataset.dataset  # Handle Subset wrapper
        
        if hasattr(base_dataset, 'get_masses_tensor'):
            self.masses = base_dataset.get_masses_tensor().to(self.device)
        else:
            self.masses = None
        
        # Get normalization stats
        if hasattr(base_dataset, 'get_normalization_stats'):
            self.norm_stats = base_dataset.get_normalization_stats()
        else:
            self.norm_stats = None
        
        # Data loaders
        self.train_loader = DataLoader(
            train_dataset, 
            batch_size=batch_size, 
            shuffle=True,
            collate_fn=collate_graphs,
            num_workers=num_workers,
            pin_memory=True if device != 'cpu' else False
        )
        self.current_epoch = 0
        
        if val_dataset:
            self.val_loader = DataLoader(
                val_dataset,
                batch_size=batch_size,
                shuffle=False,
                collate_fn=collate_graphs,
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
            weight_decay=weight_decay
        )
        
        # Cosine annealing with warm restarts
        self.scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
            self.optimizer, T_0=20, T_mult=2, eta_min=1e-6
        )
        
        # History
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'learning_rate': [],
            'energy_loss': [],
            'momentum_loss': [],
        }
        self.best_val_loss = float('inf')
    
    def _expand_masses_for_batch(self, batch) -> Optional[torch.Tensor]:
        """Expand per-simulation masses to match batched graph nodes."""
        if self.masses is None:
            return None
        
        # batch.batch gives the graph index for each node
        # All graphs in a batch have the same particle set, so masses are the same
        n_particles = len(self.masses)
        n_graphs = batch.batch.max().item() + 1
        
        # Tile masses for each graph in the batch
        expanded = self.masses.repeat(n_graphs)
        return expanded
    
    def train_epoch(self) -> float:
        """Train for one epoch."""
        self.model.train()
        total_loss = 0.0
        n_batches = 0
        
        pbar = tqdm(self.train_loader, desc=f"Epoch {self.current_epoch}", leave=False)
        for batch in pbar:
            self.optimizer.zero_grad()
            
            batch = batch.to(self.device)
            
            # Add training noise to inputs for rollout robustness
            if self.noise_std > 0:
                noise = torch.randn_like(batch.x[:, :6]) * self.noise_std
                batch.x = batch.x.clone()
                batch.x[:, :6] = batch.x[:, :6] + noise
                # Also update pos to match noised positions
                if hasattr(batch, 'pos') and batch.pos is not None:
                    batch.pos = batch.x[:, :3].clone()
            
            pred = self.model(batch)
            target = batch.y
            
            if isinstance(self.criterion, PhysicsInformedLoss):
                masses = self._expand_masses_for_batch(batch)
                loss, loss_details = self.criterion(
                    pred, target, masses=masses, batch_index=batch.batch
                )
            else:
                loss = self.criterion(pred, target)
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            
            total_loss += loss.item()
            n_batches += 1
        
        return total_loss / n_batches
    
    @torch.no_grad()
    def validate(self) -> Tuple[float, Dict]:
        """Validate the model. Returns (val_loss, loss_details)."""
        if self.val_loader is None:
            return float('nan'), {}
        
        self.model.eval()
        total_loss = 0.0
        total_details = {}
        n_batches = 0
        
        for batch in self.val_loader:
            batch = batch.to(self.device)
            pred = self.model(batch)
            target = batch.y
            
            if isinstance(self.criterion, PhysicsInformedLoss):
                masses = self._expand_masses_for_batch(batch)
                loss, details = self.criterion(
                    pred, target, masses=masses, batch_index=batch.batch
                )
                for k, v in details.items():
                    total_details[k] = total_details.get(k, 0) + v
            else:
                loss = self.criterion(pred, target)
            
            total_loss += loss.item()
            n_batches += 1
        
        avg_details = {k: v / n_batches for k, v in total_details.items()}
        return total_loss / n_batches, avg_details
    
    def train(self,
              n_epochs: int = 50,
              early_stopping_patience: int = 30,
              save_every: int = 10,
              verbose: bool = True) -> Dict:
        """Train the model with early stopping."""
        print(f"Training on {self.device}")
        print(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        if self.masses is not None:
            print(f"Physics loss: ENABLED (masses loaded for {len(self.masses)} particles)")
        else:
            print(f"Physics loss: DISABLED (no masses)")
        print(f"Input noise std: {self.noise_std}")
        
        patience_counter = 0
        
        pbar = tqdm(range(n_epochs), desc="Training")
        for epoch in pbar:
            self.current_epoch = epoch + 1
            
            train_loss = self.train_epoch()
            self.history['train_loss'].append(train_loss)
            
            val_loss, val_details = self.validate()
            self.history['val_loss'].append(val_loss)
            
            current_lr = self.optimizer.param_groups[0]['lr']
            self.history['learning_rate'].append(current_lr)
            
            # Track physics loss components
            self.history['energy_loss'].append(val_details.get('energy', 0))
            self.history['momentum_loss'].append(val_details.get('momentum', 0))
            
            # Step the cosine scheduler
            self.scheduler.step()
            
            pbar.set_postfix({
                'train': f'{train_loss:.4f}',
                'val': f'{val_loss:.4f}',
                'lr': f'{current_lr:.2e}',
                'E': f'{val_details.get("energy", 0):.4f}',
                'M': f'{val_details.get("momentum", 0):.4f}'
            })
            
            # Print clear epoch summary (visible even when tqdm scrolls)
            best_marker = " ★ BEST" if val_loss < self.best_val_loss else ""
            print(f"\n  Epoch {self.current_epoch:3d} | train: {train_loss:.4f} | val: {val_loss:.4f} | "
                  f"E: {val_details.get('energy', 0):.4f} | M: {val_details.get('momentum', 0):.4f} | "
                  f"lr: {current_lr:.2e}{best_marker}")
            
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.save_model('best_model.pt')
                patience_counter = 0
            else:
                patience_counter += 1
            
            if patience_counter >= early_stopping_patience:
                print(f"\nEarly stopping at epoch {epoch + 1}")
                break
            
            if (epoch + 1) % save_every == 0:
                self.save_model(f'checkpoint_epoch_{epoch + 1}.pt')
        
        self.save_model('final_model.pt')
        self._save_history()
        return self.history
    
    def save_model(self, filename: str) -> str:
        """Save model checkpoint with normalization stats."""
        filepath = self.model_dir / filename
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_val_loss': self.best_val_loss,
            'history': self.history,
            'norm_stats': self.norm_stats,
        }
        torch.save(checkpoint, filepath)
        return str(filepath)
    
    def load_model(self, filename: str) -> None:
        """Load model checkpoint."""
        filepath = self.model_dir / filename
        checkpoint = torch.load(filepath, map_location=self.device, weights_only=False)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.best_val_loss = checkpoint['best_val_loss']
        self.history = checkpoint['history']
        if 'norm_stats' in checkpoint:
            self.norm_stats = checkpoint['norm_stats']
    
    def _save_history(self) -> None:
        """Save training history to JSON."""
        history_path = self.model_dir / 'training_history.json'
        with open(history_path, 'w') as f:
            json.dump(self.history, f, indent=2)
