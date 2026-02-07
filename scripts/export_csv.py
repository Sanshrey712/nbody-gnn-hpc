#!/usr/bin/env python3
"""
Export Data to CSV for Readability.

Exports:
- Evaluation metrics to CSV
- Sample trajectory to CSV
- Training summary to CSV

Usage:
    python scripts/export_csv.py
"""

import argparse
import sys
import csv
import json
from pathlib import Path
import numpy as np

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))


def export_metrics_csv(results_path: Path, output_dir: Path):
    """Export evaluation metrics to CSV."""
    
    results_file = results_path / 'evaluation_results.json'
    if not results_file.exists():
        print(f"  ⚠ No evaluation results found at {results_file}")
        return
    
    with open(results_file) as f:
        results = json.load(f)
    
    # Summary metrics CSV
    summary_csv = output_dir / 'metrics_summary.csv'
    with open(summary_csv, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Metric', 'Value', 'Std Dev'])
        
        avg = results.get('average_metrics', {})
        metrics = ['position_rmse', 'position_mae', 'velocity_rmse', 'velocity_mae']
        
        for m in metrics:
            if m in avg:
                std = avg.get(f'{m}_std', 0)
                writer.writerow([m, f"{avg[m]:.6e}", f"{std:.6e}"])
    
    print(f"  ✓ Metrics summary: {summary_csv}")
    
    # Per-simulation metrics CSV
    per_sim_csv = output_dir / 'metrics_per_simulation.csv'
    per_sim = results.get('per_simulation_metrics', [])
    
    if per_sim:
        with open(per_sim_csv, 'w', newline='') as f:
            writer = csv.writer(f)
            
            # Header
            keys = [k for k in per_sim[0].keys() if not k.startswith('trajectory_distances')]
            writer.writerow(['simulation_id'] + keys)
            
            # Data
            for i, sim in enumerate(per_sim):
                row = [i + 1]
                for k in keys:
                    val = sim.get(k, '')
                    if isinstance(val, float):
                        row.append(f"{val:.6e}")
                    else:
                        row.append(val)
                writer.writerow(row)
        
        print(f"  ✓ Per-simulation metrics: {per_sim_csv}")


def export_trajectory_csv(data_dir: Path, output_dir: Path, sim_id: int = 0):
    """Export a sample trajectory to CSV."""
    
    try:
        import h5py
    except ImportError:
        print("  ⚠ h5py not installed, skipping trajectory export")
        return
    
    # Find a trajectory file
    checkpoints = data_dir / 'checkpoints'
    if not checkpoints.exists():
        print(f"  ⚠ No checkpoints found at {checkpoints}")
        return
    
    traj_files = list(checkpoints.glob('*_trajectory.h5'))
    if not traj_files:
        print("  ⚠ No trajectory files found")
        return
    
    traj_file = traj_files[min(sim_id, len(traj_files) - 1)]
    
    with h5py.File(traj_file, 'r') as f:
        positions = f['positions'][:]
        velocities = f['velocities'][:]
        masses = f['masses'][:]
        times = f['times'][:]
    
    n_steps, n_particles, _ = positions.shape
    
    # Export first 5 particles for readability (full data would be huge)
    sample_particles = min(5, n_particles)
    sample_steps = min(50, n_steps)  # First 50 timesteps
    
    traj_csv = output_dir / 'sample_trajectory.csv'
    with open(traj_csv, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            'timestep', 'time', 'particle_id', 'mass',
            'pos_x', 'pos_y', 'pos_z',
            'vel_x', 'vel_y', 'vel_z'
        ])
        
        for t in range(sample_steps):
            for p in range(sample_particles):
                writer.writerow([
                    t, f"{times[t]:.6f}", p, f"{masses[p]:.6e}",
                    f"{positions[t, p, 0]:.6e}", f"{positions[t, p, 1]:.6e}", f"{positions[t, p, 2]:.6e}",
                    f"{velocities[t, p, 0]:.6e}", f"{velocities[t, p, 1]:.6e}", f"{velocities[t, p, 2]:.6e}"
                ])
    
    print(f"  ✓ Sample trajectory ({sample_steps} steps, {sample_particles} particles): {traj_csv}")
    
    # Also export simulation info
    info_csv = output_dir / 'simulation_info.csv'
    with open(info_csv, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Parameter', 'Value'])
        writer.writerow(['Total Particles', n_particles])
        writer.writerow(['Total Timesteps', n_steps])
        writer.writerow(['Time Range', f"{times[0]:.4f} to {times[-1]:.4f}"])
        writer.writerow(['Source File', traj_file.name])
    
    print(f"  ✓ Simulation info: {info_csv}")


def export_training_csv(model_dir: Path, output_dir: Path):
    """Export training history to CSV."""
    
    history_file = model_dir / 'training_history.json'
    if not history_file.exists():
        print(f"  ⚠ No training history found at {history_file}")
        return
    
    with open(history_file) as f:
        history = json.load(f)
    
    training_csv = output_dir / 'training_history.csv'
    with open(training_csv, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['epoch', 'train_loss', 'val_loss', 'learning_rate'])
        
        n_epochs = len(history.get('train_loss', []))
        for i in range(n_epochs):
            writer.writerow([
                i + 1,
                f"{history['train_loss'][i]:.6e}",
                f"{history['val_loss'][i]:.6e}" if history.get('val_loss') else '',
                f"{history['learning_rate'][i]:.6e}" if history.get('learning_rate') else ''
            ])
    
    print(f"  ✓ Training history: {training_csv}")


def main():
    parser = argparse.ArgumentParser(description='Export data to CSV')
    parser.add_argument('--data-dir', '-d', type=str, default='./data',
                       help='Data directory')
    parser.add_argument('--model-dir', '-m', type=str, default='./models',
                       help='Model directory')
    parser.add_argument('--results-dir', '-r', type=str, default='./results',
                       help='Results directory')
    parser.add_argument('--output-dir', '-o', type=str, default='./results/csv',
                       help='CSV output directory')
    
    args = parser.parse_args()
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 50)
    print("EXPORTING DATA TO CSV")
    print("=" * 50)
    print(f"Output directory: {output_dir}\n")
    
    # Export metrics
    print("Exporting evaluation metrics...")
    export_metrics_csv(Path(args.results_dir), output_dir)
    
    # Export sample trajectory
    print("\nExporting sample trajectory...")
    export_trajectory_csv(Path(args.data_dir), output_dir)
    
    # Export training history
    print("\nExporting training history...")
    export_training_csv(Path(args.model_dir), output_dir)
    
    print("\n" + "=" * 50)
    print("CSV EXPORT COMPLETE")
    print("=" * 50)
    print(f"\nAll CSV files saved to: {output_dir}/")
    print("\nFiles created:")
    for f in output_dir.glob('*.csv'):
        print(f"  - {f.name}")


if __name__ == '__main__':
    main()
