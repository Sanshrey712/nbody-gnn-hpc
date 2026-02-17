#!/usr/bin/env python3
"""
Generate Training Data for N-Body GNN.

Runs N-body simulations and saves trajectories as HDF5 datasets.

Usage:
    python scripts/generate_data.py --particles 500 --simulations 50 --steps 200
"""

import argparse
import sys
import os

# Limit threads per process to prevent oversubscription with Numba parallel
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['NUMBA_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'

import numpy as np
from pathlib import Path
from tqdm import tqdm
import multiprocessing as mp

sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from hpc.nbody import NBodySimulator
from hpc.checkpoint import CheckpointManager, create_training_dataset


def generate_single_simulation(args):
    """Generate a single simulation."""
    sim_id, n_particles, n_steps, save_interval, box_size, seed, shared_masses = args
    
    sim = NBodySimulator(
        n_particles=n_particles,
        box_size=box_size,
        dt=0.001,
        seed=seed,
        use_barnes_hut=(n_particles > 500)
    )
    
    # Override with shared masses so all sims use the same particle masses
    if shared_masses is not None:
        sim.masses = shared_masses.copy()
        sim.accelerations = sim._compute_accelerations()  # Recompute with correct masses
    
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
    parser = argparse.ArgumentParser(description='Generate N-body training data')
    parser.add_argument('--particles', '-n', type=int, default=500,
                       help='Number of particles per simulation')
    parser.add_argument('--simulations', '-s', type=int, default=50,
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
    parser.add_argument('--sequence-length', type=int, default=5,
                       help='Sequence length for training samples')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed base')
    parser.add_argument('--batch-size', type=int, default=10,
                       help='Process simulations in batches to save memory')
    
    args = parser.parse_args()
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    checkpoint_dir = output_dir / 'checkpoints'
    checkpoint_dir.mkdir(exist_ok=True)
    
    n_workers = args.workers or min(mp.cpu_count(), 4)
    
    print("=" * 60)
    print("N-BODY DATA GENERATION")
    print("=" * 60)
    print(f"  Particles:     {args.particles}")
    print(f"  Simulations:   {args.simulations}")
    print(f"  Steps:         {args.steps}")
    print(f"  Workers:       {n_workers}")
    print(f"  Output Dir:    {output_dir}")
    print("=" * 60)
    
    manager = CheckpointManager(str(checkpoint_dir))
    
    # Pre-generate shared masses so ALL simulations use the same particle masses.
    # This ensures the physics loss (which uses a single mass array) is exact.
    rng = np.random.RandomState(args.seed)
    shared_masses = rng.uniform(1e10, 1e12, args.particles).astype(np.float32)
    print(f"  Shared masses: range [{shared_masses.min():.2e}, {shared_masses.max():.2e}]")
    
    all_trajectories = []
    n_batches = (args.simulations + args.batch_size - 1) // args.batch_size
    
    print(f"\nProcessing {args.simulations} simulations in {n_batches} batches...")
    
    for batch_idx in range(n_batches):
        start_idx = batch_idx * args.batch_size
        end_idx = min(start_idx + args.batch_size, args.simulations)
        batch_size = end_idx - start_idx
        
        print(f"\n--- Batch {batch_idx + 1}/{n_batches} (sims {start_idx}-{end_idx-1}) ---")
        
        sim_args = []
        skipped = 0
        
        for i in range(start_idx, end_idx):
            if manager.trajectory_exists(f"sim_{i:04d}"):
                skipped += 1
                continue
            sim_args.append((
                i, args.particles, args.steps, args.save_interval, 
                args.box_size, args.seed + i, shared_masses
            ))
        
        if not sim_args:
            print(f"  Already complete (skipped {skipped})")
            continue
        
        print(f"  Running {len(sim_args)} sims (skipped {skipped})...")
        
        if n_workers > 1 and len(sim_args) > 1:
            with mp.Pool(n_workers) as pool:
                batch_trajs = list(tqdm(
                    pool.imap(generate_single_simulation, sim_args),
                    total=len(sim_args), desc="Simulations"
                ))
        else:
            batch_trajs = [generate_single_simulation(a) for a in tqdm(sim_args, desc="Simulations")]
        
        # Save checkpoints
        for i, traj in enumerate(batch_trajs):
            sim_idx = sim_args[i][0]
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
                metadata={'n_particles': args.particles, 'seed': args.seed + sim_idx}
            )
        
        # Keep lightweight data for dataset creation
        for traj in batch_trajs:
            all_trajectories.append({
                'positions': traj['positions'],
                'velocities': traj['velocities'],
                'masses': traj['masses'],
                'n_steps': traj['n_steps']
            })
        
        del batch_trajs
        import gc; gc.collect()
    
    print(f"\nGenerated {len(all_trajectories)} trajectories")
    
    # Create training dataset (80/20 split)
    print("\nCreating training datasets...")
    
    n_train = int(0.8 * len(all_trajectories))
    
    # Use masses from first trajectory (all sims have the same particle count)
    masses = all_trajectories[0].get('masses', None)
    
    create_training_dataset(
        all_trajectories[:n_train],
        str(output_dir / 'train_dataset.h5'),
        sequence_length=args.sequence_length, stride=1,
        masses=masses
    )
    
    create_training_dataset(
        all_trajectories[n_train:],
        str(output_dir / 'val_dataset.h5'),
        sequence_length=args.sequence_length, stride=1,
        masses=masses
    )
    
    print("\n" + "=" * 60)
    print("DATA GENERATION COMPLETE")
    print("=" * 60)
    print(f"  Train dataset: {output_dir / 'train_dataset.h5'}")
    print(f"  Val dataset:   {output_dir / 'val_dataset.h5'}")
    print("=" * 60)


if __name__ == '__main__':
    main()
