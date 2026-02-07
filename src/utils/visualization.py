"""
Visualization utilities for N-body simulations and AI predictions.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import Axes3D
from typing import Dict, List, Optional, Tuple
from pathlib import Path
import os


class Visualizer:
    """
    Visualization tools for N-body simulations.
    """
    
    def __init__(self, output_dir: str = "./results/plots"):
        """
        Initialize visualizer.
        
        Args:
            output_dir: Directory to save plots
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Style settings
        plt.style.use('seaborn-v0_8-darkgrid')
        self.colors = plt.cm.viridis(np.linspace(0, 1, 10))
    
    def plot_trajectory_3d(self,
                           positions: np.ndarray,
                           title: str = "N-Body Trajectory",
                           particle_indices: Optional[List[int]] = None,
                           save_name: Optional[str] = None,
                           show: bool = True) -> plt.Figure:
        """
        Plot 3D particle trajectories.
        
        Args:
            positions: Trajectory (n_steps, n_particles, 3)
            title: Plot title
            particle_indices: Which particles to plot (None = all)
            save_name: Filename to save
            show: Whether to display
            
        Returns:
            Matplotlib figure
        """
        fig = plt.figure(figsize=(12, 10))
        ax = fig.add_subplot(111, projection='3d')
        
        n_steps, n_particles, _ = positions.shape
        
        if particle_indices is None:
            particle_indices = range(min(n_particles, 50))  # Limit for visibility
        
        colors = plt.cm.rainbow(np.linspace(0, 1, len(particle_indices)))
        
        for i, idx in enumerate(particle_indices):
            ax.plot(positions[:, idx, 0],
                   positions[:, idx, 1],
                   positions[:, idx, 2],
                   color=colors[i],
                   alpha=0.7,
                   linewidth=0.5)
            
            # Mark start and end
            ax.scatter(*positions[0, idx], color=colors[i], s=30, marker='o')
            ax.scatter(*positions[-1, idx], color=colors[i], s=30, marker='x')
        
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_title(title)
        
        if save_name:
            plt.savefig(self.output_dir / save_name, dpi=150, bbox_inches='tight')
        
        if show:
            plt.show()
        
        return fig
    
    def plot_comparison(self,
                        hpc_positions: np.ndarray,
                        ai_positions: np.ndarray,
                        title: str = "HPC vs AI Prediction",
                        particle_indices: Optional[List[int]] = None,
                        save_name: Optional[str] = None,
                        show: bool = True) -> plt.Figure:
        """
        Compare HPC and AI trajectories side by side.
        """
        fig = plt.figure(figsize=(16, 6))
        
        # HPC trajectory
        ax1 = fig.add_subplot(131, projection='3d')
        ax1.set_title('HPC Ground Truth')
        
        # AI trajectory
        ax2 = fig.add_subplot(132, projection='3d')
        ax2.set_title('AI Prediction')
        
        # Overlay
        ax3 = fig.add_subplot(133, projection='3d')
        ax3.set_title('Overlay (HPC=solid, AI=dashed)')
        
        n_particles = hpc_positions.shape[1]
        if particle_indices is None:
            particle_indices = range(min(n_particles, 20))
        
        colors = plt.cm.rainbow(np.linspace(0, 1, len(particle_indices)))
        
        for i, idx in enumerate(particle_indices):
            # HPC
            ax1.plot(hpc_positions[:, idx, 0],
                    hpc_positions[:, idx, 1],
                    hpc_positions[:, idx, 2],
                    color=colors[i], alpha=0.7, linewidth=0.8)
            
            # AI
            ax2.plot(ai_positions[:, idx, 0],
                    ai_positions[:, idx, 1],
                    ai_positions[:, idx, 2],
                    color=colors[i], alpha=0.7, linewidth=0.8)
            
            # Overlay
            ax3.plot(hpc_positions[:, idx, 0],
                    hpc_positions[:, idx, 1],
                    hpc_positions[:, idx, 2],
                    color=colors[i], alpha=0.7, linewidth=0.8)
            ax3.plot(ai_positions[:, idx, 0],
                    ai_positions[:, idx, 1],
                    ai_positions[:, idx, 2],
                    color=colors[i], alpha=0.7, linewidth=0.8, linestyle='--')
        
        for ax in [ax1, ax2, ax3]:
            ax.set_xlabel('X')
            ax.set_ylabel('Y')
            ax.set_zlabel('Z')
        
        plt.suptitle(title)
        plt.tight_layout()
        
        if save_name:
            plt.savefig(self.output_dir / save_name, dpi=150, bbox_inches='tight')
        
        if show:
            plt.show()
        
        return fig
    
    def plot_error_over_time(self,
                              position_rmse: np.ndarray,
                              velocity_rmse: np.ndarray,
                              title: str = "Prediction Error Over Time",
                              save_name: Optional[str] = None,
                              show: bool = True) -> plt.Figure:
        """
        Plot prediction error as function of time.
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        
        steps = np.arange(len(position_rmse))
        
        # Position RMSE
        ax1.plot(steps, position_rmse, 'b-', linewidth=2, label='Position RMSE')
        ax1.fill_between(steps, 0, position_rmse, alpha=0.3)
        ax1.set_xlabel('Time Step')
        ax1.set_ylabel('RMSE')
        ax1.set_title('Position Error')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Velocity RMSE
        ax2.plot(steps, velocity_rmse, 'r-', linewidth=2, label='Velocity RMSE')
        ax2.fill_between(steps, 0, velocity_rmse, alpha=0.3)
        ax2.set_xlabel('Time Step')
        ax2.set_ylabel('RMSE')
        ax2.set_title('Velocity Error')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.suptitle(title)
        plt.tight_layout()
        
        if save_name:
            plt.savefig(self.output_dir / save_name, dpi=150, bbox_inches='tight')
        
        if show:
            plt.show()
        
        return fig
    
    def plot_training_history(self,
                               history: Dict,
                               title: str = "Training History",
                               save_name: Optional[str] = None,
                               show: bool = True) -> plt.Figure:
        """
        Plot training and validation loss curves.
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        
        epochs = range(1, len(history['train_loss']) + 1)
        
        # Loss curves
        ax1.plot(epochs, history['train_loss'], 'b-', label='Train Loss', linewidth=2)
        if history.get('val_loss') and not all(np.isnan(history['val_loss'])):
            ax1.plot(epochs, history['val_loss'], 'r-', label='Val Loss', linewidth=2)
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.set_title('Loss Curves')
        ax1.legend()
        ax1.set_yscale('log')
        ax1.grid(True, alpha=0.3)
        
        # Learning rate
        ax2.plot(epochs, history['learning_rate'], 'g-', linewidth=2)
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Learning Rate')
        ax2.set_title('Learning Rate Schedule')
        ax2.set_yscale('log')
        ax2.grid(True, alpha=0.3)
        
        plt.suptitle(title)
        plt.tight_layout()
        
        if save_name:
            plt.savefig(self.output_dir / save_name, dpi=150, bbox_inches='tight')
        
        if show:
            plt.show()
        
        return fig
    
    def plot_energy_conservation(self,
                                  hpc_energy: np.ndarray,
                                  ai_energy: np.ndarray,
                                  title: str = "Energy Conservation",
                                  save_name: Optional[str] = None,
                                  show: bool = True) -> plt.Figure:
        """
        Compare energy conservation between HPC and AI.
        """
        fig, ax = plt.subplots(figsize=(10, 6))
        
        steps = np.arange(len(hpc_energy))
        
        # Normalize to initial energy
        hpc_norm = hpc_energy / hpc_energy[0]
        ai_norm = ai_energy / ai_energy[0]
        
        ax.plot(steps, hpc_norm, 'b-', linewidth=2, label='HPC')
        ax.plot(steps, ai_norm, 'r--', linewidth=2, label='AI')
        ax.axhline(y=1.0, color='k', linestyle=':', alpha=0.5, label='Initial')
        
        ax.set_xlabel('Time Step')
        ax.set_ylabel('Normalized Total Energy')
        ax.set_title(title)
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        if save_name:
            plt.savefig(self.output_dir / save_name, dpi=150, bbox_inches='tight')
        
        if show:
            plt.show()
        
        return fig
    
    def create_animation(self,
                         positions: np.ndarray,
                         interval: int = 50,
                         save_name: Optional[str] = None) -> FuncAnimation:
        """
        Create animation of particle motion.
        
        Args:
            positions: Trajectory (n_steps, n_particles, 3)
            interval: Milliseconds between frames
            save_name: Filename to save (must end in .gif or .mp4)
            
        Returns:
            Matplotlib animation
        """
        fig = plt.figure(figsize=(10, 10))
        ax = fig.add_subplot(111, projection='3d')
        
        n_steps, n_particles, _ = positions.shape
        
        # Set axis limits
        all_pos = positions.reshape(-1, 3)
        margin = 0.1 * (all_pos.max() - all_pos.min())
        ax.set_xlim(all_pos[:, 0].min() - margin, all_pos[:, 0].max() + margin)
        ax.set_ylim(all_pos[:, 1].min() - margin, all_pos[:, 1].max() + margin)
        ax.set_zlim(all_pos[:, 2].min() - margin, all_pos[:, 2].max() + margin)
        
        # Initialize scatter plot
        colors = plt.cm.viridis(np.linspace(0, 1, n_particles))
        scatter = ax.scatter(positions[0, :, 0],
                            positions[0, :, 1],
                            positions[0, :, 2],
                            c=colors, s=20)
        
        title = ax.set_title('Step 0')
        
        def update(frame):
            scatter._offsets3d = (positions[frame, :, 0],
                                 positions[frame, :, 1],
                                 positions[frame, :, 2])
            title.set_text(f'Step {frame}')
            return scatter, title
        
        anim = FuncAnimation(fig, update, frames=n_steps,
                            interval=interval, blit=False)
        
        if save_name:
            filepath = self.output_dir / save_name
            if save_name.endswith('.gif'):
                anim.save(filepath, writer='pillow', fps=1000//interval)
            else:
                anim.save(filepath, writer='ffmpeg', fps=1000//interval)
            print(f"Saved animation to {filepath}")
        
        return anim
    
    def plot_particle_distribution(self,
                                    positions: np.ndarray,
                                    step: int = -1,
                                    title: str = "Particle Distribution",
                                    save_name: Optional[str] = None,
                                    show: bool = True) -> plt.Figure:
        """
        Plot histogram of particle positions at a given step.
        """
        fig, axes = plt.subplots(1, 3, figsize=(15, 4))
        
        pos = positions[step]
        labels = ['X', 'Y', 'Z']
        
        for i, (ax, label) in enumerate(zip(axes, labels)):
            ax.hist(pos[:, i], bins=30, alpha=0.7, color=self.colors[i])
            ax.set_xlabel(f'{label} Position')
            ax.set_ylabel('Count')
            ax.set_title(f'{label} Distribution')
        
        plt.suptitle(title)
        plt.tight_layout()
        
        if save_name:
            plt.savefig(self.output_dir / save_name, dpi=150, bbox_inches='tight')
        
        if show:
            plt.show()
        
        return fig
