#!/usr/bin/env python3
"""
Evaluate AI Model Against HPC Ground Truth.

Usage:
    python scripts/evaluate.py --model-path models/best_model.pt
"""

import argparse
import sys
from pathlib import Path
import json
import numpy as np

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from hpc.nbody import NBodySimulator
from hpc.checkpoint import CheckpointManager
from ai.model import create_model
from ai.predict import Predictor, compare_with_hpc
from utils.metrics import compute_all_metrics, format_metrics_report
from utils.visualization import Visualizer


def main():
    parser = argparse.ArgumentParser(description='Evaluate AI Model')
    parser.add_argument('--model-path', '-m', type=str, default='./models/best_model.pt',
                       help='Path to trained model')
    parser.add_argument('--config-path', '-c', type=str, default='./models/config.json',
                       help='Path to model config')
    parser.add_argument('--data-dir', '-d', type=str, default='./data',
                       help='Data directory')
    parser.add_argument('--output-dir', '-o', type=str, default='./results',
                       help='Output directory')
    parser.add_argument('--n-test-sims', type=int, default=5,
                       help='Number of test simulations')
    parser.add_argument('--particles', '-n', type=int, default=100,
                       help='Particles for new test simulations')
    parser.add_argument('--steps', type=int, default=100,
                       help='Steps to predict')
    parser.add_argument('--seed', type=int, default=9999,
                       help='Random seed for test simulations')
    
    args = parser.parse_args()
    
    # Paths
    model_path = Path(args.model_path)
    config_path = Path(args.config_path)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if not model_path.exists():
        print(f"Error: Model not found at {model_path}")
        sys.exit(1)
    
    print("=" * 60)
    print("AI MODEL EVALUATION")
    print("=" * 60)
    
    # Load config
    if config_path.exists():
        with open(config_path) as f:
            config = json.load(f)
        model_type = config['model_type']
        model_config = config['model_config']
        print(f"  Model Type: {model_type}")
    else:
        print("Warning: Config not found, defaulting to LSTM")
        model_type = 'lstm'
        model_config = {
            'input_dim': 6,
            'hidden_dim': 128,
            'n_layers': 3,
            'output_dim': 6
        }
    
    # Create model and predictor
    print("\nLoading model...")
    model = create_model(model_type, **model_config)
    predictor = Predictor(model, str(model_path))
    
    # Generate test simulations (fresh data not seen during training)
    print(f"\nGenerating {args.n_test_sims} test simulations...")
    test_results = []
    
    visualizer = Visualizer(str(output_dir / 'plots'))
    
    for i in range(args.n_test_sims):
        print(f"\n  Test {i+1}/{args.n_test_sims}")
        
        # Run HPC simulation
        sim = NBodySimulator(
            n_particles=args.particles,
            box_size=10.0,
            dt=0.001,
            seed=args.seed + i
        )
        hpc_states = sim.run(args.steps, save_interval=1, verbose=False)
        
        # Convert to trajectory format
        hpc_trajectory = {
            'positions': np.stack([s['positions'] for s in hpc_states]),
            'velocities': np.stack([s['velocities'] for s in hpc_states]),
            'masses': hpc_states[0]['masses']
        }
        
        # Create history for LSTM
        seq_len = 10
        if len(hpc_states) > seq_len:
            history = np.stack([
                np.concatenate([hpc_states[j]['positions'], 
                               hpc_states[j]['velocities']], axis=-1)
                for j in range(seq_len)
            ])
        else:
            history = None
        
        # AI prediction
        init_pos = hpc_trajectory['positions'][seq_len]
        init_vel = hpc_trajectory['velocities'][seq_len]
        masses = hpc_trajectory['masses']
        
        prediction_steps = args.steps - seq_len - 1
        
        ai_result = predictor.predict_rollout(
            init_pos, init_vel, masses, 
            n_steps=prediction_steps,
            history=history
        )
        
        # Get corresponding HPC ground truth
        hpc_pos = hpc_trajectory['positions'][seq_len:seq_len+prediction_steps+1]
        hpc_vel = hpc_trajectory['velocities'][seq_len:seq_len+prediction_steps+1]
        
        # Compute metrics
        metrics = compute_all_metrics(
            ai_result['positions'][:len(hpc_pos)],
            ai_result['velocities'][:len(hpc_vel)],
            hpc_pos,
            hpc_vel,
            masses
        )
        
        test_results.append(metrics)
        
        print(f"    Position RMSE: {metrics['position_rmse']:.6e}")
        print(f"    Velocity RMSE: {metrics['velocity_rmse']:.6e}")
        
        # Visualize first test case
        if i == 0:
            # Trajectory comparison
            visualizer.plot_comparison(
                hpc_pos, 
                ai_result['positions'][:len(hpc_pos)],
                title=f"Test {i+1}: HPC vs AI Comparison",
                save_name=f'comparison_test_{i+1}.png',
                show=False
            )
            
            # Error over time
            pos_rmse_per_step = np.sqrt(np.mean(
                (ai_result['positions'][:len(hpc_pos)] - hpc_pos) ** 2, 
                axis=(1, 2)
            ))
            vel_rmse_per_step = np.sqrt(np.mean(
                (ai_result['velocities'][:len(hpc_vel)] - hpc_vel) ** 2, 
                axis=(1, 2)
            ))
            
            visualizer.plot_error_over_time(
                pos_rmse_per_step,
                vel_rmse_per_step,
                title=f"Test {i+1}: Prediction Error Over Time",
                save_name=f'error_over_time_test_{i+1}.png',
                show=False
            )
    
    # Aggregate metrics
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
    
    # Print report
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
        'model_type': model_type,
        'n_test_simulations': args.n_test_sims,
        'n_particles': args.particles,
        'n_steps': args.steps,
        'average_metrics': avg_metrics,
        'per_simulation_metrics': test_results
    }
    
    results_path = output_dir / 'evaluation_results.json'
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    print(f"\n  Results saved to: {results_path}")
    print(f"  Plots saved to:   {output_dir / 'plots'}")
    print("=" * 60)


if __name__ == '__main__':
    main()
