"""
N-Body Gravitational Simulation with HPC Optimizations.

This module implements high-performance N-body simulation using:
- Numba JIT compilation for near-C performance
- Vectorized NumPy operations
- Optional Barnes-Hut algorithm for O(N log N) complexity
"""

import numpy as np
from numba import jit, prange
from typing import Tuple, Optional
import multiprocessing as mp
from functools import partial


# Physical constants
G = 6.67430e-11  # Gravitational constant
SOFTENING = 1e-9  # Softening parameter to avoid singularities


@jit(nopython=True, parallel=True, fastmath=True)
def compute_accelerations_direct(positions: np.ndarray, 
                                  masses: np.ndarray,
                                  softening: float = SOFTENING) -> np.ndarray:
    """
    Compute gravitational accelerations using direct O(N²) method.
    Optimized with Numba JIT and parallel execution.
    
    Args:
        positions: (N, 3) array of particle positions
        masses: (N,) array of particle masses
        softening: Softening parameter to avoid singularities
        
    Returns:
        (N, 3) array of accelerations
    """
    n = positions.shape[0]
    accelerations = np.zeros_like(positions)
    
    for i in prange(n):
        ax, ay, az = 0.0, 0.0, 0.0
        xi, yi, zi = positions[i, 0], positions[i, 1], positions[i, 2]
        
        for j in range(n):
            if i != j:
                dx = positions[j, 0] - xi
                dy = positions[j, 1] - yi
                dz = positions[j, 2] - zi
                
                # Distance with softening
                r2 = dx*dx + dy*dy + dz*dz + softening*softening
                r = np.sqrt(r2)
                r3 = r * r2
                
                # Gravitational acceleration
                factor = G * masses[j] / r3
                ax += factor * dx
                ay += factor * dy
                az += factor * dz
        
        accelerations[i, 0] = ax
        accelerations[i, 1] = ay
        accelerations[i, 2] = az
    
    return accelerations


@jit(nopython=True, fastmath=True)
def leapfrog_step(positions: np.ndarray,
                  velocities: np.ndarray,
                  accelerations: np.ndarray,
                  masses: np.ndarray,
                  dt: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Perform one leapfrog integration step.
    Leapfrog is symplectic and conserves energy well for gravitational systems.
    
    Args:
        positions: Current positions (N, 3)
        velocities: Current velocities (N, 3)
        accelerations: Current accelerations (N, 3)
        masses: Particle masses (N,)
        dt: Time step
        
    Returns:
        Tuple of (new_positions, new_velocities, new_accelerations)
    """
    # Half-step velocity update
    velocities_half = velocities + 0.5 * dt * accelerations
    
    # Full-step position update
    new_positions = positions + dt * velocities_half
    
    # Compute new accelerations (will be done outside this function)
    # This is a placeholder - actual computation happens in the simulator
    
    return new_positions, velocities_half, accelerations


@jit(nopython=True, fastmath=True)
def compute_total_energy(positions: np.ndarray,
                         velocities: np.ndarray,
                         masses: np.ndarray,
                         softening: float = SOFTENING) -> Tuple[float, float, float]:
    """
    Compute total kinetic, potential, and total energy of the system.
    
    Returns:
        Tuple of (kinetic_energy, potential_energy, total_energy)
    """
    n = positions.shape[0]
    
    # Kinetic energy: 0.5 * m * v^2
    kinetic = 0.0
    for i in range(n):
        v2 = (velocities[i, 0]**2 + velocities[i, 1]**2 + velocities[i, 2]**2)
        kinetic += 0.5 * masses[i] * v2
    
    # Potential energy: -G * m1 * m2 / r
    potential = 0.0
    for i in range(n):
        for j in range(i + 1, n):
            dx = positions[j, 0] - positions[i, 0]
            dy = positions[j, 1] - positions[i, 1]
            dz = positions[j, 2] - positions[i, 2]
            r = np.sqrt(dx*dx + dy*dy + dz*dz + softening*softening)
            potential -= G * masses[i] * masses[j] / r
    
    return kinetic, potential, kinetic + potential


class NBodySimulator:
    """
    High-performance N-body gravitational simulator.
    
    Features:
    - Direct O(N²) or Barnes-Hut O(N log N) force calculation
    - Numba JIT-compiled kernels
    - Leapfrog symplectic integration
    - Checkpoint saving for AI training
    """
    
    def __init__(self,
                 n_particles: int = 1000,
                 box_size: float = 1.0,
                 mass_range: Tuple[float, float] = (1e10, 1e12),
                 dt: float = 1e-3,
                 softening: float = SOFTENING,
                 use_barnes_hut: bool = False,
                 theta: float = 0.5,
                 seed: Optional[int] = None):
        """
        Initialize the N-body simulator.
        
        Args:
            n_particles: Number of particles
            box_size: Size of the simulation box
            mass_range: (min_mass, max_mass) for random initialization
            dt: Time step
            softening: Softening parameter
            use_barnes_hut: Whether to use Barnes-Hut algorithm
            theta: Barnes-Hut opening angle (smaller = more accurate)
            seed: Random seed for reproducibility
        """
        self.n_particles = n_particles
        self.box_size = box_size
        self.dt = dt
        self.softening = softening
        self.use_barnes_hut = use_barnes_hut
        self.theta = theta
        self.seed = seed
        
        # Initialize random state
        if seed is not None:
            np.random.seed(seed)
        
        # Initialize particles
        self.positions = (np.random.rand(n_particles, 3) - 0.5) * box_size
        self.velocities = (np.random.rand(n_particles, 3) - 0.5) * 0.1 * box_size
        self.masses = np.random.uniform(mass_range[0], mass_range[1], n_particles)
        
        # Compute initial accelerations
        self.accelerations = self._compute_accelerations()
        
        # Simulation state
        self.time = 0.0
        self.step_count = 0
        
        # History for checkpointing
        self.history = []
    
    def _compute_accelerations(self) -> np.ndarray:
        """Compute accelerations using selected method."""
        if self.use_barnes_hut:
            from .barnes_hut import BarnesHutTree
            tree = BarnesHutTree(self.positions, self.masses, self.theta)
            return tree.compute_accelerations_jit()
        else:
            return compute_accelerations_direct(self.positions, self.masses, self.softening)
    
    def step(self) -> None:
        """Advance simulation by one time step using leapfrog integration."""
        # Half-step velocity update
        self.velocities += 0.5 * self.dt * self.accelerations
        
        # Full-step position update
        self.positions += self.dt * self.velocities
        
        # Compute new accelerations
        self.accelerations = self._compute_accelerations()
        
        # Complete velocity update
        self.velocities += 0.5 * self.dt * self.accelerations
        
        # Update time
        self.time += self.dt
        self.step_count += 1
    
    def run(self, n_steps: int, save_interval: int = 1, verbose: bool = True) -> list:
        """
        Run simulation for n_steps.
        
        Args:
            n_steps: Number of steps to run
            save_interval: Save state every N steps
            verbose: Print progress
            
        Returns:
            List of saved states
        """
        states = []
        
        # Save initial state
        states.append(self.get_state())
        
        for i in range(n_steps):
            self.step()
            
            if (i + 1) % save_interval == 0:
                states.append(self.get_state())
                
            if verbose and (i + 1) % max(1, n_steps // 10) == 0:
                energy = self.get_energy()
                print(f"Step {i+1}/{n_steps}, Time: {self.time:.4f}, Energy: {energy[2]:.6e}")
        
        self.history = states
        return states
    
    def get_state(self) -> dict:
        """Get current simulation state as a dictionary."""
        return {
            'positions': self.positions.copy(),
            'velocities': self.velocities.copy(),
            'accelerations': self.accelerations.copy(),
            'masses': self.masses.copy(),
            'time': self.time,
            'step': self.step_count
        }
    
    def set_state(self, state: dict) -> None:
        """Restore simulation from a state dictionary."""
        self.positions = state['positions'].copy()
        self.velocities = state['velocities'].copy()
        self.accelerations = state['accelerations'].copy()
        self.masses = state['masses'].copy()
        self.time = state['time']
        self.step_count = state['step']
    
    def get_energy(self) -> Tuple[float, float, float]:
        """Get current system energy (kinetic, potential, total)."""
        return compute_total_energy(self.positions, self.velocities, 
                                    self.masses, self.softening)
    
    @classmethod
    def create_solar_system(cls, scale: float = 1.0) -> 'NBodySimulator':
        """Create a simplified solar system simulation."""
        sim = cls(n_particles=9, box_size=50.0, dt=0.01)
        
        # Simplified solar system (sun + 8 planets)
        # Masses in solar masses, distances in AU
        bodies = [
            ('Sun', 1.0, 0.0, 0.0),
            ('Mercury', 1.66e-7, 0.39, 47.87),
            ('Venus', 2.45e-6, 0.72, 35.02),
            ('Earth', 3.00e-6, 1.0, 29.78),
            ('Mars', 3.23e-7, 1.52, 24.07),
            ('Jupiter', 9.55e-4, 5.2, 13.07),
            ('Saturn', 2.86e-4, 9.58, 9.69),
            ('Uranus', 4.37e-5, 19.22, 6.81),
            ('Neptune', 5.15e-5, 30.05, 5.43)
        ]
        
        sim.masses = np.array([b[1] for b in bodies]) * 1.989e30 * scale
        sim.positions = np.zeros((9, 3))
        sim.velocities = np.zeros((9, 3))
        
        for i, (name, mass, dist, vel) in enumerate(bodies):
            sim.positions[i, 0] = dist * 1.496e11 * scale  # AU to meters
            sim.velocities[i, 1] = vel * 1000 * scale      # km/s to m/s
        
        sim.accelerations = sim._compute_accelerations()
        return sim
    
    @classmethod
    def create_galaxy_collision(cls, n_per_galaxy: int = 500) -> 'NBodySimulator':
        """Create two colliding galaxy simulation."""
        n_total = 2 * n_per_galaxy
        sim = cls(n_particles=n_total, box_size=100.0, dt=0.01)
        
        # Galaxy 1 - centered at (-20, 0, 0)
        theta1 = np.random.rand(n_per_galaxy) * 2 * np.pi
        r1 = np.random.exponential(5.0, n_per_galaxy)
        sim.positions[:n_per_galaxy, 0] = -20 + r1 * np.cos(theta1)
        sim.positions[:n_per_galaxy, 1] = r1 * np.sin(theta1)
        sim.positions[:n_per_galaxy, 2] = np.random.randn(n_per_galaxy) * 0.5
        
        # Galaxy 2 - centered at (20, 0, 0)
        theta2 = np.random.rand(n_per_galaxy) * 2 * np.pi
        r2 = np.random.exponential(5.0, n_per_galaxy)
        sim.positions[n_per_galaxy:, 0] = 20 + r2 * np.cos(theta2)
        sim.positions[n_per_galaxy:, 1] = r2 * np.sin(theta2)
        sim.positions[n_per_galaxy:, 2] = np.random.randn(n_per_galaxy) * 0.5
        
        # Set velocities for collision
        sim.velocities[:n_per_galaxy, 0] = 2.0
        sim.velocities[n_per_galaxy:, 0] = -2.0
        
        # Add circular rotation within each galaxy
        for i in range(n_per_galaxy):
            r = np.sqrt(sim.positions[i, 0]**2 + sim.positions[i, 1]**2)
            if r > 0:
                sim.velocities[i, 0] += -sim.positions[i, 1] / r * 0.5
                sim.velocities[i, 1] += sim.positions[i, 0] / r * 0.5
        
        sim.accelerations = sim._compute_accelerations()
        return sim


def run_parallel_simulations(configs: list, n_workers: int = None) -> list:
    """
    Run multiple simulations in parallel using multiprocessing.
    
    Args:
        configs: List of simulation configuration dictionaries
        n_workers: Number of worker processes (default: CPU count)
        
    Returns:
        List of simulation results (histories)
    """
    if n_workers is None:
        n_workers = mp.cpu_count()
    
    def run_single(config):
        sim = NBodySimulator(**config.get('init', {}))
        states = sim.run(**config.get('run', {}))
        return states
    
    with mp.Pool(n_workers) as pool:
        results = pool.map(run_single, configs)
    
    return results
