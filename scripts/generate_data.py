#!/usr/bin/env python3
"""
Generate HPC Training Data for AI Model.

Runs N-body simulations and saves trajectories for training.

Usage:
    python scripts/generate_data.py --particles 1000 --simulations 50 --steps 200
"""

import argparse
import sys
import os

# Limit threads per process to prevent oversubscription when running many workers
# This is CRITICAL for performance when using Numba parallel=True in a multiprocessing context
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['NUMBA_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'

import numpy as np
from pathlib import Path
from tqdm import tqdm
import multiprocessing as mp

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from hpc.nbody import NBodySimulator
from hpc.checkpoint import CheckpointManager, create_training_dataset


def generate_single_simulation(args):
    """Generate a single simulation."""
    sim_id, n_particles, n_steps, save_interval, box_size, seed = args
    
    # Create simulator with unique seed
    sim = NBodySimulator(
        n_particles=n_particles,
        box_size=box_size,
        dt=0.001,
        seed=seed,
        use_barnes_hut=(n_particles > 500)  # Use Barnes-Hut for large sims
    )
    
    # Run simulation
    states = sim.run(n_steps, save_interval=save_interval, verbose=False)
    
    return {
        'positions': np.stack([s['positions'] for s in states]),
        'velocities': np.stack([s['velocities'] for s in states]),
        'accelerations': np.stack([s['accelerations'] for s in states]),
        'masses': states[0]['masses'],
        'times': np.array([s['time'] for s in states]),
        'n_steps': len(states)
    }


def main():
    parser = argparse.ArgumentParser(description='Generate HPC training data')
    parser.add_argument('--particles', '-n', type=int, default=100,
                       help='Number of particles per simulation')
    parser.add_argument('--simulations', '-s', type=int, default=20,
                       help='Number of simulations to run')
    parser.add_argument('--steps', type=int, default=200,
                       help='Timesteps per simulation')
    parser.add_argument('--save-interval', type=int, default=1,
                       help='Save state every N steps')
    parser.add_argument('--box-size', type=float, default=10.0,
                       help='Simulation box size')
    parser.add_argument('--workers', '-w', type=int, default=None,
                       help='Number of parallel workers')
    parser.add_argument('--output-dir', '-o', type=str, default='./data',
                       help='Output directory')
    parser.add_argument('--sequence-length', type=int, default=10,
                       help='Sequence length for training samples')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed base')
    parser.add_argument('--batch-size', type=int, default=10,
                       help='Process simulations in batches to save memory')
    
    args = parser.parse_args()
    
    # Setup
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    checkpoint_dir = output_dir / 'checkpoints'
    checkpoint_dir.mkdir(exist_ok=True)
    
    n_workers = args.workers or min(mp.cpu_count(), 4)  # Limit workers to save memory
    
    print("=" * 60)
    print("HPC DATA GENERATION")
    print("=" * 60)
    print(f"  Particles:     {args.particles}")
    print(f"  Simulations:   {args.simulations}")
    print(f"  Steps:         {args.steps}")
    print(f"  Workers:       {n_workers}")
    print(f"  Output Dir:    {output_dir}")
    print(f"  Batch Size:    {args.batch_size}")
    print("=" * 60)
    
    manager = CheckpointManager(str(checkpoint_dir))
    
    # Process simulations in batches to save memory
    all_trajectories_for_dataset = []
    n_batches = (args.simulations + args.batch_size - 1) // args.batch_size
    
    print(f"\nProcessing {args.simulations} simulations in {n_batches} batches...")
    
    for batch_idx in range(n_batches):
        start_idx = batch_idx * args.batch_size
        end_idx = min(start_idx + args.batch_size, args.simulations)
        batch_size = end_idx - start_idx
        
        print(f"\n--- Batch {batch_idx + 1}/{n_batches} (sims {start_idx}-{end_idx-1}) ---")
        
        # Prepare simulation arguments for this batch
        sim_args = [
            (i, args.particles, args.steps, args.save_interval, 
             args.box_size, args.seed + i)
            for i in range(start_idx, end_idx)
        ]
        
        # Run simulations in parallel
        if n_workers > 1 and batch_size > 1:
            with mp.Pool(n_workers) as pool:
                batch_trajectories = list(tqdm(
                    pool.imap(generate_single_simulation, sim_args),
                    total=batch_size,
                    desc="Simulations"
                ))
        else:
            batch_trajectories = []
            for sim_arg in tqdm(sim_args, desc="Simulations"):
                batch_trajectories.append(generate_single_simulation(sim_arg))
        
        # Save trajectories to disk immediately
        for i, traj in enumerate(tqdm(batch_trajectories, desc="Saving", leave=False)):
            sim_idx = start_idx + i
            manager.save_trajectory(
                states=[{
                    'positions': traj['positions'][t],
                    'velocities': traj['velocities'][t],
                    'accelerations': traj['accelerations'][t],
                    'masses': traj['masses'],
                    'time': float(traj['times'][t]),
                    'step': t
                } for t in range(traj['n_steps'])],
                name=f"sim_{sim_idx:04d}",
                metadata={
                    'n_particles': args.particles,
                    'n_steps': args.steps,
                    'box_size': args.box_size,
                    'seed': args.seed + sim_idx
                }
            )
        
        # Keep lightweight trajectory data for dataset creation
        for traj in batch_trajectories:
            all_trajectories_for_dataset.append({
                'positions': traj['positions'],
                'velocities': traj['velocities'],
                'n_steps': traj['n_steps']
            })
            # Free the heavy arrays we don't need
            del traj['accelerations']
            del traj['masses']
            del traj['times']
        
        # Clear batch to free memory
        del batch_trajectories
        import gc
        gc.collect()
    
    print(f"\nGenerated {len(all_trajectories_for_dataset)} trajectories")
    
    # Create training dataset
    print("\nCreating training dataset...")
    dataset_path = output_dir / 'training_dataset.h5'
    
    create_training_dataset(
        all_trajectories_for_dataset,
        str(dataset_path),
        sequence_length=args.sequence_length,
        stride=1
    )
    
    # Split into train/val
    print("\nSplitting into train/val...")
    n_train = int(0.8 * len(all_trajectories_for_dataset))
    
    create_training_dataset(
        all_trajectories_for_dataset[:n_train],
        str(output_dir / 'train_dataset.h5'),
        sequence_length=args.sequence_length,
        stride=1
    )
    
    create_training_dataset(
        all_trajectories_for_dataset[n_train:],
        str(output_dir / 'val_dataset.h5'),
        sequence_length=args.sequence_length,
        stride=1
    )
    
    print("\n" + "=" * 60)
    print("DATA GENERATION COMPLETE")
    print("=" * 60)
    print(f"  Trajectories saved to: {checkpoint_dir}")
    print(f"  Training dataset:      {output_dir / 'train_dataset.h5'}")
    print(f"  Validation dataset:    {output_dir / 'val_dataset.h5'}")
    print("=" * 60)


if __name__ == '__main__':
    main()
