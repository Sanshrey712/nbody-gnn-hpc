"""
Graph Neural Network for N-Body State Prediction.

A physics-aware GNN that treats particles as graph nodes and models
pairwise gravitational interactions through message-passing.

Key design choices:
- Delta prediction: predicts state *changes* (Δpos, Δvel) not absolute states
- Dropout regularization to prevent overfitting
- Residual connections with LayerNorm for stable deep networks
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing
from torch_geometric.data import Data
from typing import Optional
import numpy as np


class ParticleInteractionLayer(MessagePassing):
    """
    Message-passing layer for particle interactions.
    
    Learns gravitational-like interactions between particles
    based on their positions and velocities.
    """
    
    def __init__(self, 
                 node_features: int,
                 edge_features: int,
                 hidden_dim: int = 128,
                 dropout: float = 0.1):
        super().__init__(aggr='add')
        
        # Edge network: encodes pairwise interactions
        self.edge_mlp = nn.Sequential(
            nn.Linear(2 * node_features + edge_features, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
        )
        
        # Node update network
        self.node_mlp = nn.Sequential(
            nn.Linear(node_features + hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, node_features)
        )
    
    def forward(self, x, edge_index, edge_attr):
        return self.propagate(edge_index, x=x, edge_attr=edge_attr)
    
    def message(self, x_i, x_j, edge_attr):
        """Compute messages between particles."""
        edge_input = torch.cat([x_i, x_j, edge_attr], dim=-1)
        return self.edge_mlp(edge_input)
    
    def update(self, aggr_out, x):
        """Update node features."""
        return self.node_mlp(torch.cat([x, aggr_out], dim=-1))


class NBodyGNN(nn.Module):
    """
    Graph Neural Network for N-body state prediction.
    
    Uses a physics-informed architecture that:
    - Treats particles as graph nodes
    - Models pairwise interactions as edges
    - Predicts state DELTAS (Δpos, Δvel) added to current state
    - Uses dropout for regularization
    """
    
    def __init__(self,
                 node_input_dim: int = 7,   # pos(3) + vel(3) + mass(1)
                 hidden_dim: int = 128,
                 n_layers: int = 3,
                 output_dim: int = 6,       # delta_position(3) + delta_velocity(3)
                 dropout: float = 0.1):
        super().__init__()
        
        self.output_dim = output_dim
        edge_dim = 5  # distance(1) + direction(3) + inv_dist_sq(1)
        
        # Input encoding
        self.node_encoder = nn.Sequential(
            nn.Linear(node_input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # Message passing layers
        self.layers = nn.ModuleList([
            ParticleInteractionLayer(hidden_dim, edge_dim, hidden_dim, dropout=dropout)
            for _ in range(n_layers)
        ])
        
        # Layer norms for residual connections
        self.layer_norms = nn.ModuleList([
            nn.LayerNorm(hidden_dim) for _ in range(n_layers)
        ])
        
        # Output decoder — predicts DELTAS
        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.SiLU(),
            nn.Linear(hidden_dim // 2, output_dim)
        )
        
        # Initialize decoder output near zero (small deltas initially)
        nn.init.zeros_(self.decoder[-1].weight)
        nn.init.zeros_(self.decoder[-1].bias)
    
    def compute_edge_features(self, pos: torch.Tensor, 
                               edge_index: torch.Tensor) -> torch.Tensor:
        """Compute physics-informed edge features (distance, direction, 1/r²)."""
        row, col = edge_index
        diff = pos[col] - pos[row]
        dist = torch.norm(diff, dim=-1, keepdim=True) + 1e-8
        direction = diff / dist
        inv_dist_sq = 1.0 / (dist ** 2 + 1e-6)  # ~gravitational scaling
        return torch.cat([dist, direction, inv_dist_sq], dim=-1)
    
    def forward(self, data: Data) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            data: PyG Data with x (node features), edge_index, pos
                
        Returns:
            Predicted NEXT STATE (current_state + delta) — shape (N, output_dim)
        """
        x = data.x
        edge_index = data.edge_index
        
        # Extract current pos and vel for delta prediction
        current_pos = x[:, :3]
        current_vel = x[:, 3:6]
        current_state = torch.cat([current_pos, current_vel], dim=-1)
        
        # Compute edge features from positions
        if hasattr(data, 'pos') and data.pos is not None:
            edge_attr = self.compute_edge_features(data.pos, edge_index)
        else:
            edge_attr = self.compute_edge_features(current_pos, edge_index)
        
        # Encode nodes
        h = self.node_encoder(x)
        
        # Message passing with residual connections
        for layer, norm in zip(self.layers, self.layer_norms):
            h_new = layer(h, edge_index, edge_attr)
            h = norm(h + h_new)
        
        # Decode to DELTA
        delta = self.decoder(h)
        
        # Return current_state + delta (residual prediction)
        return current_state + delta
