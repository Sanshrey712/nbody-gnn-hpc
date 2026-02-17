#!/usr/bin/env python3
"""
Train GNN Model for N-Body Prediction.

Usage:
    python scripts/train_model.py --epochs 200
"""

import argparse
import sys
from pathlib import Path
import torch
import json

sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from ai.model import NBodyGNN
from ai.train import Trainer, GNNDataset
from ai.config import TrainingConfig


def main():
    parser = argparse.ArgumentParser(description='Train N-Body GNN Model')
    parser.add_argument('--epochs', '-e', type=int, default=None)
    parser.add_argument('--batch-size', '-b', type=int, default=None)
    parser.add_argument('--learning-rate', '-lr', type=float, default=None)
    parser.add_argument('--hidden-dim', type=int, default=None)
    parser.add_argument('--n-layers', type=int, default=None)
    parser.add_argument('--data-dir', '-d', type=str, default='./data')
    parser.add_argument('--model-dir', '-o', type=str, default='./models')
    parser.add_argument('--early-stopping', type=int, default=None)
    parser.add_argument('--physics-loss', action='store_true', default=True)
    parser.add_argument('--workers', '-w', type=int, default=None)
    parser.add_argument('--max-samples', type=int, default=None,
                       help='Limit training samples (default: use all)')
    parser.add_argument('--dropout', type=float, default=None)
    parser.add_argument('--noise-std', type=float, default=None)
    parser.add_argument('--weight-decay', type=float, default=None)
    
    args = parser.parse_args()

    config = TrainingConfig()
    
    # Override with CLI args
    if args.batch_size is not None: config.batch_size = args.batch_size
    if args.epochs is not None: config.epochs = args.epochs
    if args.learning_rate is not None: config.learning_rate = args.learning_rate
    if args.hidden_dim is not None: config.hidden_dim = args.hidden_dim
    if args.n_layers is not None: config.n_layers = args.n_layers
    if args.early_stopping is not None: config.early_stopping = args.early_stopping
    if args.workers is not None: config.workers = args.workers
    if args.dropout is not None: config.dropout = args.dropout
    if args.noise_std is not None: config.noise_std = args.noise_std
    if args.weight_decay is not None: config.weight_decay = args.weight_decay
    
    device = config.get_device()
    data_dir = Path(args.data_dir)
    model_dir = Path(args.model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)
    
    train_path = data_dir / 'train_dataset.h5'
    val_path = data_dir / 'val_dataset.h5'
    
    if not train_path.exists():
        print(f"Error: Training data not found at {train_path}")
        print("Run generate_data.py first!")
        sys.exit(1)
    
    print("=" * 60)
    print("N-BODY GNN TRAINING")
    print("=" * 60)
    print(f"  Device:          {device}")
    print(f"  Epochs:          {config.epochs}")
    print(f"  Batch Size:      {config.batch_size}")
    print(f"  Learning Rate:   {config.learning_rate}")
    print(f"  Hidden Dim:      {config.hidden_dim}")
    print(f"  Layers:          {config.n_layers}")
    print(f"  k-Neighbors:     {config.k_neighbors}")
    print(f"  Dropout:         {config.dropout}")
    print(f"  Weight Decay:    {config.weight_decay}")
    print(f"  Noise Std:       {config.noise_std}")
    print(f"  Physics Loss:    {args.physics_loss}")
    print("=" * 60)
    
    # Create datasets
    print("\nLoading datasets...")
    train_dataset = GNNDataset(
        str(train_path),
        sequence_length=config.sequence_length,
        k_neighbors=config.k_neighbors
    )
    
    # Val dataset uses training normalization stats for consistency
    train_norm_stats = train_dataset.get_normalization_stats()
    val_dataset = GNNDataset(
        str(val_path),
        sequence_length=config.sequence_length,
        k_neighbors=config.k_neighbors,
        external_norm_stats=train_norm_stats
    ) if val_path.exists() else None
    
    # Limit dataset size if requested
    if args.max_samples and len(train_dataset) > args.max_samples:
        print(f"Subsampling: {len(train_dataset)} -> {args.max_samples}")
        train_dataset = torch.utils.data.Subset(train_dataset, range(args.max_samples))
    
    model_config = {
        'node_input_dim': 7,   # pos(3) + vel(3) + mass(1)
        'hidden_dim': config.hidden_dim,
        'n_layers': config.n_layers,
        'output_dim': 6,       # pos(3) + vel(3)
        'dropout': config.dropout,
    }
    
    print(f"\n  Train samples: {len(train_dataset)}")
    if val_dataset:
        print(f"  Val samples:   {len(val_dataset)}")
    
    # Create model
    model = NBodyGNN(**model_config)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Parameters:    {n_params:,}")
    
    # Save config
    saved_config = {
        'model_type': 'gnn',
        'model_config': model_config,
        'training_config': vars(config),
    }
    with open(model_dir / 'config.json', 'w') as f:
        json.dump(saved_config, f, indent=2, default=str)
    
    # Train
    trainer = Trainer(
        model=model,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        model_dir=str(model_dir),
        learning_rate=config.learning_rate,
        batch_size=config.batch_size,
        use_physics_loss=args.physics_loss,
        num_workers=config.workers,
        device=device,
        weight_decay=config.weight_decay,
        noise_std=config.noise_std,
        n_epochs=config.epochs,
    )
    
    print("\nStarting training...")
    history = trainer.train(
        n_epochs=config.epochs,
        early_stopping_patience=config.early_stopping,
        save_every=10
    )
    
    print("\n" + "=" * 60)
    print("TRAINING COMPLETE")
    print("=" * 60)
    print(f"  Best Val Loss:    {trainer.best_val_loss:.6f}")
    print(f"  Final Train Loss: {history['train_loss'][-1]:.6f}")
    print(f"  Model saved to:   {model_dir}")
    print("=" * 60)
    
    # Plot training history
    try:
        from utils.visualization import Visualizer
        viz = Visualizer(str(model_dir / 'plots'))
        viz.plot_training_history(history, save_name='training_history.png', show=False)
        print(f"  Training plot:    {model_dir / 'plots' / 'training_history.png'}")
    except Exception as e:
        print(f"  (Could not create plot: {e})")


if __name__ == '__main__':
    main()
