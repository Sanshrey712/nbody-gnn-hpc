#!/usr/bin/env python3
"""
Full Demo Pipeline: Generate Data → Train → Evaluate

This script runs the complete AI-HPC workflow end-to-end using the 
configured hardware profile.

Usage:
    python scripts/run_demo.py --profile hpc
    python scripts/run_demo.py --profile mac
"""

import argparse
import sys
import os
from pathlib import Path
import subprocess
import time

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from ai.config import get_config

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
    
    print(f"\nCompleted in {elapsed:.1f}s")
    return result


def main():
    parser = argparse.ArgumentParser(description='Run full AI-HPC demo')
    parser.add_argument('--profile', type=str, default='auto',
                       choices=['auto', 'hpc', 'mac', 'default'],
                       help='Hardware profile to use')
    parser.add_argument('--skip-training', action='store_true',
                       help='Skip training (use existing model)')
    parser.add_argument('--model', type=str, default='gnn',
                       choices=['gnn', 'lstm', 'hybrid'],
                       help='Model type to use')
    
    args = parser.parse_args()
    
    # Load Configuration
    config = get_config(args.profile)
    print("\n" + "=" * 60)
    print("AI-HPC AUTOMATED PIPELINE")
    print("=" * 60)
    print(f"Profile:      {config.__class__.__name__}")
    print(f"Model:        {args.model}")
    print(f"\nConfiguration:")
    print(f"  Particles:    {config.particles}")
    print(f"  Simulations:  {config.simulations}")
    print(f"  Steps:        {config.steps}")
    print(f"  Device:       {config.device}")
    print(f"  Workers:      {config.workers}")
    print("=" * 60)
    
    scripts_dir = Path(__file__).parent
    python = sys.executable
    
    total_start = time.time()
    
    # Step 1: Generate Data
    # generate_data.py handles its own parallelization based on workers
    run_command([
        python, str(scripts_dir / 'generate_data.py'),
        '--particles', str(config.particles),
        '--simulations', str(config.simulations),
        '--steps', str(config.steps),
        '--workers', str(config.workers),
        '--sequence-length', str(config.sequence_length)
    ], "Generating Training Data")
    
    # Step 2: Train Model
    if not args.skip_training:
        run_command([
            python, str(scripts_dir / 'train_model.py'),
            '--profile', args.profile,  # Pass profile down
            '--model', args.model,
            '--physics-loss'
        ], f"Training {args.model.upper()} Model")
    else:
        print("\n[Skipping training - using existing model]")
    
    # Step 3: Evaluate
    run_command([
        python, str(scripts_dir / 'evaluate.py'),
        '--n-test-sims', str(config.n_test_sims),
        '--particles', str(config.particles), # Test on same scale
        '--steps', str(config.steps)
    ], "Evaluating AI vs HPC")
    
    # Step 4: Export to CSV
    run_command([
        python, str(scripts_dir / 'export_csv.py')
    ], "Exporting Results to CSV")
    
    # Summary
    total_elapsed = time.time() - total_start
    
    print("\n" + "=" * 60)
    print("PIPELINE COMPLETE!")
    print("=" * 60)
    print(f"\nTotal time: {total_elapsed:.1f}s ({total_elapsed/60:.1f} min)")
    print("\nOutputs available in:")
    print("  - models/")
    print("  - results/")
    print("=" * 60)


if __name__ == '__main__':
    main()
