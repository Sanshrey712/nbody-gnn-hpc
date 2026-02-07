"""
Accuracy Metrics for N-Body Predictions.

Implements:
- RMSE (Root Mean Square Error)
- MAE (Mean Absolute Error)
- Energy conservation error
- Momentum conservation error
- Trajectory divergence metrics
"""

import numpy as np
from typing import Dict, Tuple, Optional


def compute_rmse(predicted: np.ndarray, 
                 target: np.ndarray,
                 per_particle: bool = False) -> np.ndarray:
    """
    Compute Root Mean Square Error.
    
    Args:
        predicted: Predicted values
        target: Ground truth values
        per_particle: If True, return RMSE per particle
        
    Returns:
        RMSE value(s)
    """
    diff = predicted - target
    
    if per_particle:
        # RMSE per particle (across time and dimensions)
        return np.sqrt(np.mean(diff ** 2, axis=(0, -1)))
    else:
        # Overall RMSE
        return np.sqrt(np.mean(diff ** 2))


def compute_mae(predicted: np.ndarray, 
                target: np.ndarray,
                per_particle: bool = False) -> np.ndarray:
    """
    Compute Mean Absolute Error.
    
    Args:
        predicted: Predicted values
        target: Ground truth values
        per_particle: If True, return MAE per particle
        
    Returns:
        MAE value(s)
    """
    diff = np.abs(predicted - target)
    
    if per_particle:
        return np.mean(diff, axis=(0, -1))
    else:
        return np.mean(diff)


def compute_energy_error(positions: np.ndarray,
                         velocities: np.ndarray,
                         masses: np.ndarray,
                         G: float = 6.67430e-11,
                         softening: float = 1e-9) -> Tuple[np.ndarray, float]:
    """
    Compute total energy at each timestep and relative energy error.
    
    Args:
        positions: (n_steps, n_particles, 3) positions
        velocities: (n_steps, n_particles, 3) velocities
        masses: (n_particles,) masses
        G: Gravitational constant
        softening: Softening parameter
        
    Returns:
        (energy_per_step, relative_error)
    """
    n_steps, n_particles, _ = positions.shape
    energies = np.zeros(n_steps)
    
    for t in range(n_steps):
        # Kinetic energy
        v_squared = np.sum(velocities[t] ** 2, axis=1)
        kinetic = 0.5 * np.sum(masses * v_squared)
        
        # Potential energy
        potential = 0.0
        for i in range(n_particles):
            for j in range(i + 1, n_particles):
                r = np.linalg.norm(positions[t, i] - positions[t, j])
                r = np.sqrt(r**2 + softening**2)
                potential -= G * masses[i] * masses[j] / r
        
        energies[t] = kinetic + potential
    
    # Relative energy error (compared to initial)
    relative_error = np.abs((energies - energies[0]) / energies[0])
    
    return energies, float(np.max(relative_error))


def compute_momentum_error(velocities: np.ndarray,
                           masses: np.ndarray) -> Tuple[np.ndarray, float]:
    """
    Compute total momentum at each timestep and relative error.
    
    Args:
        velocities: (n_steps, n_particles, 3) velocities
        masses: (n_particles,) masses
        
    Returns:
        (momentum_magnitude_per_step, relative_error)
    """
    n_steps = velocities.shape[0]
    
    # Total momentum per step
    momentum = np.zeros((n_steps, 3))
    for t in range(n_steps):
        momentum[t] = np.sum(masses[:, np.newaxis] * velocities[t], axis=0)
    
    momentum_mag = np.linalg.norm(momentum, axis=1)
    
    # Relative error
    initial_mag = max(momentum_mag[0], 1e-10)  # Avoid division by zero
    relative_error = np.abs((momentum_mag - momentum_mag[0]) / initial_mag)
    
    return momentum_mag, float(np.max(relative_error))


def compute_trajectory_divergence(predicted_pos: np.ndarray,
                                   target_pos: np.ndarray) -> Dict[str, float]:
    """
    Compute various trajectory divergence metrics.
    
    Args:
        predicted_pos: Predicted positions (n_steps, n_particles, 3)
        target_pos: Ground truth positions (n_steps, n_particles, 3)
        
    Returns:
        Dictionary of metrics
    """
    n_steps, n_particles, _ = predicted_pos.shape
    
    # Distance between predicted and true positions per step
    distances = np.sqrt(np.sum((predicted_pos - target_pos) ** 2, axis=-1))  # (n_steps, n_particles)
    
    # Mean distance over particles per step
    mean_dist_per_step = np.mean(distances, axis=1)
    
    # Max distance over particles per step
    max_dist_per_step = np.max(distances, axis=1)
    
    # Lyapunov-like exponent (rate of divergence)
    # Fit exponential growth to mean distance
    log_dist = np.log(mean_dist_per_step + 1e-10)
    steps = np.arange(n_steps)
    
    # Simple linear fit to log distance
    if n_steps > 1:
        slope, _ = np.polyfit(steps, log_dist, 1)
    else:
        slope = 0.0
    
    return {
        'mean_rmse': float(compute_rmse(predicted_pos, target_pos)),
        'final_rmse': float(np.sqrt(np.mean(distances[-1] ** 2))),
        'mean_distance': float(np.mean(mean_dist_per_step)),
        'max_distance': float(np.max(max_dist_per_step)),
        'divergence_rate': float(slope),  # Lyapunov-like exponent
        'distances_per_step': mean_dist_per_step.tolist()
    }


def compute_all_metrics(predicted_pos: np.ndarray,
                        predicted_vel: np.ndarray,
                        target_pos: np.ndarray,
                        target_vel: np.ndarray,
                        masses: np.ndarray) -> Dict:
    """
    Compute all accuracy metrics.
    
    Args:
        predicted_pos: Predicted positions
        predicted_vel: Predicted velocities
        target_pos: Ground truth positions
        target_vel: Ground truth velocities
        masses: Particle masses
        
    Returns:
        Dictionary with all metrics
    """
    metrics = {}
    
    # Basic errors
    metrics['position_rmse'] = float(compute_rmse(predicted_pos, target_pos))
    metrics['position_mae'] = float(compute_mae(predicted_pos, target_pos))
    metrics['velocity_rmse'] = float(compute_rmse(predicted_vel, target_vel))
    metrics['velocity_mae'] = float(compute_mae(predicted_vel, target_vel))
    
    # Trajectory divergence
    divergence = compute_trajectory_divergence(predicted_pos, target_pos)
    metrics.update({f'trajectory_{k}': v for k, v in divergence.items()})
    
    # Energy conservation (for predicted trajectory)
    try:
        pred_energy, pred_energy_error = compute_energy_error(
            predicted_pos, predicted_vel, masses
        )
        target_energy, target_energy_error = compute_energy_error(
            target_pos, target_vel, masses
        )
        metrics['predicted_energy_error'] = pred_energy_error
        metrics['target_energy_error'] = target_energy_error
    except Exception:
        metrics['predicted_energy_error'] = float('nan')
        metrics['target_energy_error'] = float('nan')
    
    # Momentum conservation
    try:
        _, pred_momentum_error = compute_momentum_error(predicted_vel, masses)
        _, target_momentum_error = compute_momentum_error(target_vel, masses)
        metrics['predicted_momentum_error'] = pred_momentum_error
        metrics['target_momentum_error'] = target_momentum_error
    except Exception:
        metrics['predicted_momentum_error'] = float('nan')
        metrics['target_momentum_error'] = float('nan')
    
    return metrics


def format_metrics_report(metrics: Dict) -> str:
    """
    Format metrics as a readable report.
    
    Args:
        metrics: Dictionary of metrics
        
    Returns:
        Formatted string report
    """
    lines = [
        "=" * 50,
        "N-BODY PREDICTION ACCURACY REPORT",
        "=" * 50,
        "",
        "BASIC METRICS",
        "-" * 30,
        f"  Position RMSE:     {metrics.get('position_rmse', 'N/A'):.6e}",
        f"  Position MAE:      {metrics.get('position_mae', 'N/A'):.6e}",
        f"  Velocity RMSE:     {metrics.get('velocity_rmse', 'N/A'):.6e}",
        f"  Velocity MAE:      {metrics.get('velocity_mae', 'N/A'):.6e}",
        "",
        "TRAJECTORY ANALYSIS",
        "-" * 30,
        f"  Final Step RMSE:   {metrics.get('trajectory_final_rmse', 'N/A'):.6e}",
        f"  Mean Distance:     {metrics.get('trajectory_mean_distance', 'N/A'):.6e}",
        f"  Max Distance:      {metrics.get('trajectory_max_distance', 'N/A'):.6e}",
        f"  Divergence Rate:   {metrics.get('trajectory_divergence_rate', 'N/A'):.6e}",
        "",
        "PHYSICS CONSERVATION",
        "-" * 30,
        f"  Predicted Energy Error:   {metrics.get('predicted_energy_error', 'N/A'):.2%}",
        f"  Target Energy Error:      {metrics.get('target_energy_error', 'N/A'):.2%}",
        f"  Predicted Momentum Error: {metrics.get('predicted_momentum_error', 'N/A'):.2%}",
        f"  Target Momentum Error:    {metrics.get('target_momentum_error', 'N/A'):.2%}",
        "",
        "=" * 50
    ]
    
    return "\n".join(lines)
