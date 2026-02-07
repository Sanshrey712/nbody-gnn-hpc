#!/usr/bin/env python3
"""
Train AI Model for N-Body Prediction.

Usage:
    python scripts/train_model.py --epochs 100 --model gnn
"""

import argparse
import sys
from pathlib import Path
import json

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from ai.model import create_model, NBodyGNN, NBodyLSTM
from ai.train import Trainer, NBodyDataset, GNNDataset
from ai.config import get_config, TrainingConfig


def main():
    parser = argparse.ArgumentParser(description='Train N-Body AI Model')
    parser.add_argument('--profile', type=str, default='auto',
                       choices=['auto', 'hpc', 'mac', 'default'],
                       help='Hardware profile to use')
    parser.add_argument('--model', '-m', type=str, default='gnn',
                       choices=['gnn', 'lstm', 'hybrid'],
                       help='Model type')
    parser.add_argument('--epochs', '-e', type=int, default=None,
                       help='Training epochs')
    parser.add_argument('--batch-size', '-b', type=int, default=None,
                       help='Batch size')
    parser.add_argument('--learning-rate', '-lr', type=float, default=None,
                       help='Learning rate')
    parser.add_argument('--hidden-dim', type=int, default=None,
                       help='Hidden dimension')
    parser.add_argument('--n-layers', type=int, default=None,
                       help='Number of layers')
    parser.add_argument('--data-dir', '-d', type=str, default='./data',
                       help='Data directory')
    parser.add_argument('--model-dir', '-o', type=str, default='./models',
                       help='Model output directory')
    parser.add_argument('--sequence-length', type=int, default=None,
                       help='Sequence length for LSTM')
    parser.add_argument('--early-stopping', type=int, default=None,
                       help='Early stopping patience')
    parser.add_argument('--physics-loss', action='store_true',
                       help='Use physics-informed loss')
    parser.add_argument('--workers', '-w', type=int, default=None,
                       help='Number of data loading workers')
    
    args = parser.parse_args()

    # Load configuration
    config = get_config(args.profile)
    print(f"\nUsing Hardware Profile: {config.__class__.__name__}")
    
    # Override config with command line arguments if provided
    if args.batch_size is not None: config.batch_size = args.batch_size
    if args.epochs is not None: config.epochs = args.epochs
    if args.learning_rate is not None: config.learning_rate = args.learning_rate
    if args.hidden_dim is not None: config.hidden_dim = args.hidden_dim
    if args.n_layers is not None: config.n_layers = args.n_layers
    if args.sequence_length is not None: config.sequence_length = args.sequence_length
    if args.early_stopping is not None: config.early_stopping = args.early_stopping
    if args.workers is not None: config.workers = args.workers
    
    # Determine device
    device = TrainingConfig.get_device(config.device)
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
    print("N-BODY AI MODEL TRAINING")
    print("=" * 60)
    print(f"  Model Type:      {args.model}")
    print(f"  Profile:         {args.profile}")
    print(f"  Device:          {device}")
    print(f"  Epochs:          {config.epochs}")
    print(f"  Batch Size:      {config.batch_size}")
    print(f"  Learning Rate:   {config.learning_rate}")
    print(f"  Hidden Dim:      {config.hidden_dim}")
    print(f"  Layers:          {config.n_layers}")
    print(f"  Physics Loss:    {args.physics_loss}")
    print("=" * 60)
    
    # Create datasets
    print("\nLoading datasets...")
    
    if args.model == 'gnn':
        train_dataset = GNNDataset(
            str(train_path),
            sequence_length=config.sequence_length,
            k_neighbors=config.k_neighbors
        )
        val_dataset = GNNDataset(
            str(val_path),
            sequence_length=config.sequence_length,
            k_neighbors=config.k_neighbors
        ) if val_path.exists() else None
        
        # GNN config
        model_config = {
            'node_input_dim': 7,  # pos(3) + vel(3) + mass(1)
            'hidden_dim': config.hidden_dim,
            'n_layers': config.n_layers,
            'output_dim': 6  # pos(3) + vel(3)
        }
        
    else:  # lstm or hybrid
        train_dataset = NBodyDataset(
            str(train_path),
            sequence_length=config.sequence_length,
            model_type=args.model
        )
        val_dataset = NBodyDataset(
            str(val_path),
            sequence_length=config.sequence_length,
            model_type=args.model
        ) if val_path.exists() else None
        
        # LSTM config
        model_config = {
            'input_dim': 6,
            'hidden_dim': config.hidden_dim,
            'n_layers': config.n_layers,
            'output_dim': 6
        }
    
    print(f"  Train samples: {len(train_dataset)}")
    if val_dataset:
        print(f"  Val samples:   {len(val_dataset)}")
    
    # Create model
    print("\nCreating model...")
    model = create_model(args.model, **model_config)
    
    # Count parameters
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Model parameters: {n_params:,}")
    
    # Save config
    saved_config = {
        'model_type': args.model,
        'profile': args.profile,
        'model_config': model_config,
        'training_config': vars(config),
        'args': vars(args)
    }
    
    with open(model_dir / 'config.json', 'w') as f:
        json.dump(saved_config, f, indent=2, default=str)
    
    # Create trainer
    trainer = Trainer(
        model=model,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        model_dir=str(model_dir),
        learning_rate=config.learning_rate,
        batch_size=config.batch_size,
        use_physics_loss=args.physics_loss,
        num_workers=config.workers,
        device=device
    )
    
    # Train
    print("\nStarting training...")
    history = trainer.train(
        n_epochs=config.epochs,
        early_stopping_patience=config.early_stopping,
        save_every=10
    )
    
    # Summary
    print("\n" + "=" * 60)
    print("TRAINING COMPLETE")
    print("=" * 60)
    print(f"  Best Val Loss:   {trainer.best_val_loss:.6f}")
    print(f"  Final Train Loss: {history['train_loss'][-1]:.6f}")
    print(f"  Model saved to:  {model_dir}")
    print("=" * 60)
    
    # Plot training history
    try:
        from utils.visualization import Visualizer
        viz = Visualizer(str(model_dir / 'plots'))
        viz.plot_training_history(history, save_name='training_history.png', show=False)
        print(f"  Training plot:   {model_dir / 'plots' / 'training_history.png'}")
    except Exception as e:
        print(f"  (Could not create plot: {e})")


if __name__ == '__main__':
    main()
