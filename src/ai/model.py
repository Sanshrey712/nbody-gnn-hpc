"""
Neural Network Models for N-Body State Prediction.

This module implements:
- Graph Neural Network (GNN) for physics-aware particle interaction learning
- LSTM for temporal sequence prediction
- Hybrid GNN-LSTM for best accuracy
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing, GCNConv, GATConv
from torch_geometric.data import Data, Batch
from typing import Tuple, Optional
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
                 hidden_dim: int = 128):
        super().__init__(aggr='add')
        
        # Edge network: encodes pairwise interactions
        self.edge_mlp = nn.Sequential(
            nn.Linear(2 * node_features + edge_features, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # Node update network
        self.node_mlp = nn.Sequential(
            nn.Linear(node_features + hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, node_features)
        )
    
    def forward(self, x, edge_index, edge_attr):
        """
        Args:
            x: Node features (N, node_features)
            edge_index: Edge indices (2, E)
            edge_attr: Edge features (E, edge_features)
        """
        return self.propagate(edge_index, x=x, edge_attr=edge_attr)
    
    def message(self, x_i, x_j, edge_attr):
        """Compute messages between particles."""
        # Concatenate sender, receiver, and edge features
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
    - Learns to predict accelerations/velocities
    """
    
    def __init__(self,
                 node_input_dim: int = 9,  # pos(3) + vel(3) + mass(1) + acc(3) or similar
                 hidden_dim: int = 128,
                 n_layers: int = 4,
                 output_dim: int = 6,  # position(3) + velocity(3) prediction
                 use_edge_features: bool = True):
        """
        Initialize the GNN.
        
        Args:
            node_input_dim: Input dimension per particle
            hidden_dim: Hidden layer dimension
            n_layers: Number of message-passing layers
            output_dim: Output dimension (typically 6 for pos+vel)
            use_edge_features: Whether to use distance/direction as edge features
        """
        super().__init__()
        
        self.use_edge_features = use_edge_features
        edge_dim = 4 if use_edge_features else 1  # distance + direction (3) + 1
        
        # Input encoding
        self.node_encoder = nn.Sequential(
            nn.Linear(node_input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # Message passing layers
        self.layers = nn.ModuleList([
            ParticleInteractionLayer(hidden_dim, edge_dim, hidden_dim)
            for _ in range(n_layers)
        ])
        
        # Layer norms for residual connections
        self.layer_norms = nn.ModuleList([
            nn.LayerNorm(hidden_dim) for _ in range(n_layers)
        ])
        
        # Output decoder
        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.SiLU(),
            nn.Linear(hidden_dim // 2, output_dim)
        )
    
    def compute_edge_features(self, pos: torch.Tensor, 
                               edge_index: torch.Tensor) -> torch.Tensor:
        """Compute physics-informed edge features (distance, direction)."""
        row, col = edge_index
        diff = pos[col] - pos[row]  # Direction vector
        dist = torch.norm(diff, dim=-1, keepdim=True) + 1e-8
        direction = diff / dist
        
        return torch.cat([dist, direction], dim=-1)
    
    def forward(self, data: Data) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            data: PyTorch Geometric Data object with:
                - x: Node features (N, node_input_dim)
                - edge_index: Edge connectivity (2, E)
                - pos: Particle positions (N, 3) for edge feature computation
                
        Returns:
            Predicted state changes (N, output_dim)
        """
        x = data.x
        edge_index = data.edge_index
        
        # Compute edge features
        if self.use_edge_features and hasattr(data, 'pos'):
            edge_attr = self.compute_edge_features(data.pos, edge_index)
        else:
            # Default edge features (just ones)
            edge_attr = torch.ones(edge_index.shape[1], 1, device=x.device)
        
        # Encode nodes
        h = self.node_encoder(x)
        
        # Message passing with residual connections
        for layer, norm in zip(self.layers, self.layer_norms):
            h_new = layer(h, edge_index, edge_attr)
            h = norm(h + h_new)  # Residual connection
        
        # Decode to output
        out = self.decoder(h)
        
        return out
    
    @staticmethod
    def create_graph_data(positions: np.ndarray,
                          velocities: np.ndarray,
                          masses: np.ndarray,
                          k_neighbors: int = None,
                          device: str = 'cpu') -> Data:
        """
        Create PyTorch Geometric Data from numpy arrays.
        
        Args:
            positions: (N, 3) positions
            velocities: (N, 3) velocities
            masses: (N,) masses
            k_neighbors: Number of nearest neighbors for edges (None = fully connected)
            device: Target device
            
        Returns:
            PyTorch Geometric Data object
        """
        n = len(masses)
        
        # Node features: position + velocity + mass
        x = np.concatenate([
            positions, 
            velocities, 
            masses.reshape(-1, 1)
        ], axis=1)
        x = torch.tensor(x, dtype=torch.float32, device=device)
        
        # Create edges (fully connected or k-nearest neighbors)
        if k_neighbors is None or k_neighbors >= n - 1:
            # Fully connected
            row = np.repeat(np.arange(n), n)
            col = np.tile(np.arange(n), n)
            # Remove self-loops
            mask = row != col
            row = row[mask]
            col = col[mask]
        else:
            # K-nearest neighbors
            from scipy.spatial import distance_matrix
            dist = distance_matrix(positions, positions)
            row, col = [], []
            for i in range(n):
                neighbors = np.argsort(dist[i])[1:k_neighbors+1]  # Exclude self
                row.extend([i] * k_neighbors)
                col.extend(neighbors)
            row = np.array(row)
            col = np.array(col)
        
        edge_index = torch.tensor(np.stack([row, col]), dtype=torch.long, device=device)
        pos = torch.tensor(positions, dtype=torch.float32, device=device)
        
        return Data(x=x, edge_index=edge_index, pos=pos)


class NBodyLSTM(nn.Module):
    """
    LSTM-based model for temporal N-body prediction.
    
    Takes a sequence of states and predicts the next state.
    Good for capturing temporal dynamics.
    """
    
    def __init__(self,
                 input_dim: int = 6,  # pos(3) + vel(3) per particle
                 hidden_dim: int = 256,
                 n_layers: int = 2,
                 output_dim: int = 6,
                 dropout: float = 0.1):
        """
        Initialize LSTM model.
        
        Args:
            input_dim: Features per particle (typically 6: pos + vel)
            hidden_dim: LSTM hidden dimension
            n_layers: Number of LSTM layers
            output_dim: Output features per particle
            dropout: Dropout rate
        """
        super().__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        
        # Particle encoder
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # LSTM for temporal dynamics
        self.lstm = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=n_layers,
            batch_first=True,
            dropout=dropout if n_layers > 1 else 0
        )
        
        # Output decoder
        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input sequence (batch, seq_len, n_particles, input_dim)
            
        Returns:
            Predicted next state (batch, n_particles, output_dim)
        """
        batch_size, seq_len, n_particles, feat_dim = x.shape
        
        # Encode each particle independently
        x = x.view(batch_size * n_particles, seq_len, feat_dim)
        x = self.encoder(x)  # (B*N, seq, hidden)
        
        # LSTM
        lstm_out, _ = self.lstm(x)  # (B*N, seq, hidden)
        
        # Take last timestep
        last_hidden = lstm_out[:, -1, :]  # (B*N, hidden)
        
        # Decode
        out = self.decoder(last_hidden)  # (B*N, output)
        
        # Reshape back
        out = out.view(batch_size, n_particles, -1)
        
        return out


class HybridGNNLSTM(nn.Module):
    """
    Hybrid model combining GNN and LSTM.
    
    - GNN captures spatial particle interactions
    - LSTM captures temporal dynamics
    
    Best for accurate long-term predictions.
    """
    
    def __init__(self,
                 n_particles: int,
                 node_input_dim: int = 7,
                 hidden_dim: int = 128,
                 gnn_layers: int = 3,
                 lstm_layers: int = 2,
                 output_dim: int = 6,
                 dropout: float = 0.1):
        """
        Initialize hybrid model.
        """
        super().__init__()
        
        self.n_particles = n_particles
        self.hidden_dim = hidden_dim
        
        # GNN for spatial interactions
        self.gnn = NBodyGNN(
            node_input_dim=node_input_dim,
            hidden_dim=hidden_dim,
            n_layers=gnn_layers,
            output_dim=hidden_dim  # Output embeddings, not predictions
        )
        
        # LSTM for temporal dynamics
        self.lstm = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=lstm_layers,
            batch_first=True,
            dropout=dropout if lstm_layers > 1 else 0
        )
        
        # Output decoder
        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, output_dim)
        )
    
    def forward(self, 
                graph_sequence: list,
                return_embeddings: bool = False) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            graph_sequence: List of Data objects (sequence of graphs)
            return_embeddings: Whether to return embeddings instead of predictions
            
        Returns:
            Predicted next state (n_particles, output_dim)
        """
        # Process each graph with GNN
        gnn_embeddings = []
        for graph in graph_sequence:
            emb = self.gnn(graph)  # (n_particles, hidden_dim)
            gnn_embeddings.append(emb)
        
        # Stack into sequence (1, seq_len, n_particles, hidden)
        embeddings = torch.stack(gnn_embeddings, dim=0)
        embeddings = embeddings.permute(1, 0, 2)  # (n_particles, seq_len, hidden)
        
        # LSTM
        lstm_out, _ = self.lstm(embeddings)
        last_hidden = lstm_out[:, -1, :]  # (n_particles, hidden)
        
        if return_embeddings:
            return last_hidden
        
        # Combine GNN and LSTM features
        combined = torch.cat([gnn_embeddings[-1], last_hidden], dim=-1)
        
        # Decode
        out = self.decoder(combined)
        
        return out


def create_model(model_type: str = 'gnn', **kwargs) -> nn.Module:
    """
    Factory function to create models.
    
    Args:
        model_type: 'gnn', 'lstm', or 'hybrid'
        **kwargs: Model-specific arguments
        
    Returns:
        Initialized model
    """
    models = {
        'gnn': NBodyGNN,
        'lstm': NBodyLSTM,
        'hybrid': HybridGNNLSTM
    }
    
    if model_type not in models:
        raise ValueError(f"Unknown model type: {model_type}")
    
    return models[model_type](**kwargs)
