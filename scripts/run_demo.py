#!/usr/bin/env python3
"""
Full Demo Pipeline: Clean → Generate Data → Train GNN → Evaluate

Runs the complete workflow end-to-end.

Usage:
    python scripts/run_demo.py
    python scripts/run_demo.py --skip-training     # Skip training, only evaluate
    python scripts/run_demo.py --skip-datagen       # Reuse existing data
"""

import argparse
import sys
import os
import shutil
from pathlib import Path
import subprocess
import time

sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from ai.config import TrainingConfig


def run_command(cmd: list, description: str):
    """Run a command and print output."""
    print(f"\n{'='*60}")
    print(f"STEP: {description}")
    print(f"{'='*60}")
    print(f"Running: {' '.join(cmd)}\n")
    
    start = time.time()
    result = subprocess.run(cmd, cwd=Path(__file__).parent.parent)
    elapsed = time.time() - start
    
    if result.returncode != 0:
        print(f"\nError: {description} failed with code {result.returncode}")
        sys.exit(1)
    
    print(f"\nCompleted in {elapsed:.1f}s ({elapsed/60:.1f} min)")
    return result


def clean_previous_results(project_root: Path, keep_data: bool = False):
    """Delete all previous data, models, and results."""
    print(f"\n{'='*60}")
    print("STEP: Cleaning Previous Results")
    print(f"{'='*60}")
    
    dirs_to_clean = [
        ('results', 'Results'),
    ]
    
    if not keep_data:
        dirs_to_clean.insert(0, ('data/checkpoints', 'Checkpoints'))
    
    files_to_clean = [
        'models/best_model.pt',
        'models/final_model.pt',
        'models/training_history.json',
        'models/config.json',
    ]
    
    if not keep_data:
        files_to_clean += [
            'data/train_dataset.h5',
            'data/val_dataset.h5',
        ]
    
    # Also clean any checkpoint_epoch_*.pt files
    models_dir = project_root / 'models'
    if models_dir.exists():
        for f in models_dir.glob('checkpoint_epoch_*.pt'):
            files_to_clean.append(str(f.relative_to(project_root)))
    
    for dir_path, label in dirs_to_clean:
        full_path = project_root / dir_path
        if full_path.exists():
            shutil.rmtree(full_path)
            print(f"  Deleted {label}: {dir_path}")
    
    for file_path in files_to_clean:
        full_path = project_root / file_path
        if full_path.exists():
            full_path.unlink()
            print(f"  Deleted: {file_path}")
    
    # Recreate directories
    (project_root / 'data').mkdir(exist_ok=True)
    (project_root / 'models').mkdir(exist_ok=True)
    (project_root / 'results').mkdir(exist_ok=True)
    
    print("  Clean complete!\n")


def main():
    parser = argparse.ArgumentParser(description='Run full AI-HPC demo pipeline')
    parser.add_argument('--skip-training', action='store_true',
                       help='Skip training (use existing model)')
    parser.add_argument('--skip-datagen', action='store_true',
                       help='Skip data generation (reuse existing data)')
    parser.add_argument('--no-clean', action='store_true',
                       help='Do not delete previous results')
    parser.add_argument('--max-samples', type=int, default=None,
                       help='Limit training samples (default: use all)')
    
    args = parser.parse_args()
    
    config = TrainingConfig()
    project_root = Path(__file__).parent.parent
    
    print("\n" + "=" * 60)
    print("AI-HPC N-BODY GNN PIPELINE")
    print("=" * 60)
    print(f"  Particles:    {config.particles}")
    print(f"  Simulations:  {config.simulations}")
    print(f"  Steps:        {config.steps}")
    print(f"  Hidden Dim:   {config.hidden_dim}")
    print(f"  Layers:       {config.n_layers}")
    print(f"  Dropout:      {config.dropout}")
    print(f"  k-Neighbors:  {config.k_neighbors}")
    print(f"  Batch Size:   {config.batch_size}")
    print(f"  Noise Std:    {config.noise_std}")
    print(f"  Weight Decay: {config.weight_decay}")
    print(f"  Device:       {config.get_device()}")
    print("=" * 60)
    
    scripts_dir = Path(__file__).parent
    python = sys.executable
    
    total_start = time.time()
    
    # Step 0: Clean previous results
    if not args.no_clean:
        clean_previous_results(project_root, keep_data=args.skip_datagen)
    
    # Step 1: Generate Data
    if args.skip_datagen:
        training_data = project_root / 'data' / 'train_dataset.h5'
        if not training_data.exists():
            print("Error: --skip-datagen specified but no training data found!")
            sys.exit(1)
        print("\n[Skipping data generation - using existing data]")
    else:
        run_command([
            python, str(scripts_dir / 'generate_data.py'),
            '--particles', str(config.particles),
            '--simulations', str(config.simulations),
            '--steps', str(config.steps),
            '--sequence-length', str(config.sequence_length),
            '--workers', str(config.workers)
        ], "Generating Training Data")
    
    # Step 2: Train Model
    if not args.skip_training:
        cmd = [
            python, str(scripts_dir / 'train_model.py'),
            '--physics-loss',
            '--epochs', str(config.epochs),
        ]
        if args.max_samples:
            cmd += ['--max-samples', str(args.max_samples)]
        run_command(cmd, "Training GNN Model")
    else:
        print("\n[Skipping training - using existing model]")
    
    # Step 3: Evaluate
    run_command([
        python, str(scripts_dir / 'evaluate.py'),
        '--n-test-sims', str(config.n_test_sims),
        '--particles', str(config.particles),
        '--steps', str(config.steps)
    ], "Evaluating AI vs HPC")
    
    # Step 4: Export to CSV (if script exists)
    export_script = scripts_dir / 'export_csv.py'
    if export_script.exists():
        run_command([python, str(export_script)], "Exporting Results to CSV")
    
    total_elapsed = time.time() - total_start
    
    print("\n" + "=" * 60)
    print("PIPELINE COMPLETE!")
    print("=" * 60)
    print(f"\nTotal time: {total_elapsed:.1f}s ({total_elapsed/60:.1f} min)")
    print("\nOutputs:")
    print("  - data/       (training & validation datasets)")
    print("  - models/     (trained model & training history)")
    print("  - results/    (evaluation metrics & plots)")
    print("=" * 60)


if __name__ == '__main__':
    main()
