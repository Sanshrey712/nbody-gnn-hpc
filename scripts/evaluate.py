#!/usr/bin/env python3
"""
Evaluate GNN Model Against HPC Ground Truth.

Usage:
    python scripts/evaluate.py
"""

import argparse
import sys
from pathlib import Path
import json
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from hpc.nbody import NBodySimulator
from ai.model import NBodyGNN
from ai.predict import Predictor
from utils.metrics import compute_all_metrics, format_metrics_report
from utils.visualization import Visualizer


def main():
    parser = argparse.ArgumentParser(description='Evaluate GNN Model')
    parser.add_argument('--model-path', '-m', type=str, default='./models/best_model.pt')
    parser.add_argument('--config-path', '-c', type=str, default='./models/config.json')
    parser.add_argument('--output-dir', '-o', type=str, default='./results')
    parser.add_argument('--n-test-sims', type=int, default=10)
    parser.add_argument('--particles', '-n', type=int, default=200)
    parser.add_argument('--steps', type=int, default=400)
    parser.add_argument('--seed', type=int, default=9999)
    
    args = parser.parse_args()
    
    model_path = Path(args.model_path)
    config_path = Path(args.config_path)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if not model_path.exists():
        print(f"Error: Model not found at {model_path}")
        sys.exit(1)
    
    print("=" * 60)
    print("GNN MODEL EVALUATION")
    print("=" * 60)
    
    # Load config
    if config_path.exists():
        with open(config_path) as f:
            config = json.load(f)
        model_config = config['model_config']
        k_neighbors = config.get('training_config', {}).get('k_neighbors', 40)
    else:
        model_config = {
            'node_input_dim': 7,
            'hidden_dim': 256,
            'n_layers': 6,
            'output_dim': 6,
            'dropout': 0.1,
        }
        k_neighbors = 40
    
    # Create predictor
    print("\nLoading model...")
    model = NBodyGNN(**model_config)
    predictor = Predictor(model, str(model_path), k_neighbors=k_neighbors)
    
    # Run test simulations
    print(f"\nRunning {args.n_test_sims} test simulations ({args.particles} particles, {args.steps} steps)...")
    test_results = []
    visualizer = Visualizer(str(output_dir / 'plots'))
    
    # Use shared masses matching training (same seed=42 and range as generate_data.py)
    rng = np.random.RandomState(42)
    shared_masses = rng.uniform(1e10, 1e12, args.particles).astype(np.float32)
    
    seq_len = 5  # Start prediction after this many steps
    
    for i in range(args.n_test_sims):
        print(f"\n  Test {i+1}/{args.n_test_sims}")
        
        # Run HPC ground truth
        sim = NBodySimulator(
            n_particles=args.particles,
            box_size=10.0, dt=0.001,
            seed=args.seed + i
        )
        # Override with shared masses (same as training)
        sim.masses = shared_masses.copy()
        sim.accelerations = sim._compute_accelerations()
        hpc_states = sim.run(args.steps, save_interval=1, verbose=False)
        
        hpc_trajectory = {
            'positions': np.stack([s['positions'] for s in hpc_states]),
            'velocities': np.stack([s['velocities'] for s in hpc_states]),
            'masses': hpc_states[0]['masses']
        }
        
        # AI prediction starting from step seq_len
        init_pos = hpc_trajectory['positions'][seq_len]
        init_vel = hpc_trajectory['velocities'][seq_len]
        masses = hpc_trajectory['masses']
        
        prediction_steps = args.steps - seq_len - 1
        
        ai_result = predictor.predict_rollout(
            init_pos, init_vel, masses, n_steps=prediction_steps
        )
        
        # Compare with ground truth
        hpc_pos = hpc_trajectory['positions'][seq_len:seq_len+prediction_steps+1]
        hpc_vel = hpc_trajectory['velocities'][seq_len:seq_len+prediction_steps+1]
        
        metrics = compute_all_metrics(
            ai_result['positions'][:len(hpc_pos)],
            ai_result['velocities'][:len(hpc_vel)],
            hpc_pos, hpc_vel, masses
        )
        
        test_results.append(metrics)
        print(f"    Position RMSE: {metrics['position_rmse']:.6e}")
        print(f"    Velocity RMSE: {metrics['velocity_rmse']:.6e}")
        
        # Visualize first test
        if i == 0:
            visualizer.plot_comparison(
                hpc_pos, ai_result['positions'][:len(hpc_pos)],
                title="Test 1: HPC vs AI",
                save_name='comparison_test_1.png', show=False
            )
            
            pos_rmse = np.sqrt(np.mean(
                (ai_result['positions'][:len(hpc_pos)] - hpc_pos) ** 2, axis=(1, 2)
            ))
            vel_rmse = np.sqrt(np.mean(
                (ai_result['velocities'][:len(hpc_vel)] - hpc_vel) ** 2, axis=(1, 2)
            ))
            visualizer.plot_error_over_time(
                pos_rmse, vel_rmse,
                title="Test 1: Error Over Time",
                save_name='error_over_time_test_1.png', show=False
            )
            
            # Energy Conservation Plot
            try:
                from utils.metrics import compute_energy_error
                pred_energy, _ = compute_energy_error(
                    ai_result['positions'], ai_result['velocities'], masses
                )
                target_energy, _ = compute_energy_error(
                    hpc_pos, hpc_vel, masses
                )
                visualizer.plot_energy_conservation(
                    target_energy, pred_energy,
                    title="Test 1: Energy Conservation",
                    save_name='energy_conservation_test_1.png', show=False
                )
            except Exception as e:
                print(f"    (Could not plot energy: {e})")
    
    # Aggregate results
    print("\n" + "=" * 60)
    print("EVALUATION RESULTS")
    print("=" * 60)
    
    avg_metrics = {}
    for key in test_results[0].keys():
        if isinstance(test_results[0][key], (int, float)) and not np.isnan(test_results[0][key]):
            values = [r[key] for r in test_results if not np.isnan(r.get(key, float('nan')))]
            if values:
                avg_metrics[key] = float(np.mean(values))
                avg_metrics[f'{key}_std'] = float(np.std(values))
    
    print(f"\nAveraged over {args.n_test_sims} test simulations:")
    print("-" * 40)
    print(f"  Position RMSE:  {avg_metrics.get('position_rmse', 'N/A'):.6e} ± {avg_metrics.get('position_rmse_std', 0):.6e}")
    print(f"  Position MAE:   {avg_metrics.get('position_mae', 'N/A'):.6e} ± {avg_metrics.get('position_mae_std', 0):.6e}")
    print(f"  Velocity RMSE:  {avg_metrics.get('velocity_rmse', 'N/A'):.6e} ± {avg_metrics.get('velocity_rmse_std', 0):.6e}")
    print(f"  Velocity MAE:   {avg_metrics.get('velocity_mae', 'N/A'):.6e} ± {avg_metrics.get('velocity_mae_std', 0):.6e}")
    print("-" * 40)
    
    # Save results
    results = {
        'model_path': str(model_path),
        'model_type': 'gnn',
        'n_test_simulations': args.n_test_sims,
        'n_particles': args.particles,
        'n_steps': args.steps,
        'average_metrics': avg_metrics,
        'per_simulation_metrics': test_results
    }
    
    results_path = output_dir / 'evaluation_results.json'
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    print(f"\n  Results: {results_path}")
    print(f"  Plots:   {output_dir / 'plots'}")
    print("=" * 60)


if __name__ == '__main__':
    main()
