"""
Checkpoint management for saving and loading simulation states.

Supports multiple formats:
- HDF5 (default, efficient for large data)
- NumPy .npz files
- JSON (for metadata only)
"""

import numpy as np
import h5py
import json
import os
from pathlib import Path
from typing import Dict, List, Optional, Union
from datetime import datetime


class CheckpointManager:
    """
    Manages saving and loading of simulation checkpoints.
    
    Checkpoints include:
    - Particle positions, velocities, accelerations
    - Masses
    - Simulation time and step count
    - Configuration metadata
    """
    
    def __init__(self, 
                 checkpoint_dir: str = "./data/checkpoints",
                 format: str = "hdf5"):
        """
        Initialize checkpoint manager.
        
        Args:
            checkpoint_dir: Directory to save checkpoints
            format: Save format ('hdf5' or 'npz')
        """
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.format = format
    
    def save_state(self, 
                   state: Dict,
                   name: str,
                   metadata: Optional[Dict] = None) -> str:
        """
        Save a single simulation state.
        
        Args:
            state: State dictionary with positions, velocities, etc.
            name: Name for this checkpoint
            metadata: Optional metadata to include
            
        Returns:
            Path to saved checkpoint
        """
        if self.format == "hdf5":
            return self._save_hdf5(state, name, metadata)
        else:
            return self._save_npz(state, name, metadata)
    
    def _save_hdf5(self, state: Dict, name: str, metadata: Optional[Dict]) -> str:
        """Save state in HDF5 format."""
        filepath = self.checkpoint_dir / f"{name}.h5"
        
        with h5py.File(filepath, 'w') as f:
            # Save arrays
            for key, value in state.items():
                if isinstance(value, np.ndarray):
                    f.create_dataset(key, data=value, compression='gzip')
                elif isinstance(value, (int, float)):
                    f.attrs[key] = value
            
            # Save metadata
            if metadata:
                meta_group = f.create_group('metadata')
                for key, value in metadata.items():
                    if isinstance(value, (int, float, str)):
                        meta_group.attrs[key] = value
                    else:
                        meta_group.attrs[key] = json.dumps(value)
            
            f.attrs['created_at'] = datetime.now().isoformat()
        
        return str(filepath)
    
    def _save_npz(self, state: Dict, name: str, metadata: Optional[Dict]) -> str:
        """Save state in NumPy npz format."""
        filepath = self.checkpoint_dir / f"{name}.npz"
        
        # Separate arrays and scalars
        arrays = {k: v for k, v in state.items() if isinstance(v, np.ndarray)}
        scalars = {k: v for k, v in state.items() if isinstance(v, (int, float))}
        
        # Add scalars as 0-d arrays
        for k, v in scalars.items():
            arrays[f"scalar_{k}"] = np.array(v)
        
        # Add metadata
        if metadata:
            arrays['metadata_json'] = np.array(json.dumps(metadata))
        
        np.savez_compressed(filepath, **arrays)
        return str(filepath)
    
    def load_state(self, name: str) -> Dict:
        """
        Load a checkpoint state.
        
        Args:
            name: Checkpoint name (without extension)
            
        Returns:
            State dictionary
        """
        # Try HDF5 first
        hdf5_path = self.checkpoint_dir / f"{name}.h5"
        if hdf5_path.exists():
            return self._load_hdf5(hdf5_path)
        
        # Try npz
        npz_path = self.checkpoint_dir / f"{name}.npz"
        if npz_path.exists():
            return self._load_npz(npz_path)
        
        raise FileNotFoundError(f"Checkpoint '{name}' not found")
    
    def _load_hdf5(self, filepath: Path) -> Dict:
        """Load state from HDF5 format."""
        state = {}
        
        with h5py.File(filepath, 'r') as f:
            # Load arrays
            for key in f.keys():
                if key != 'metadata':
                    state[key] = f[key][:]
            
            # Load scalar attributes
            for key in f.attrs.keys():
                if key != 'created_at':
                    state[key] = f.attrs[key]
            
            # Load metadata if present
            if 'metadata' in f:
                state['metadata'] = {}
                for key in f['metadata'].attrs.keys():
                    value = f['metadata'].attrs[key]
                    try:
                        state['metadata'][key] = json.loads(value)
                    except (json.JSONDecodeError, TypeError):
                        state['metadata'][key] = value
        
        return state
    
    def _load_npz(self, filepath: Path) -> Dict:
        """Load state from npz format."""
        data = np.load(filepath, allow_pickle=True)
        state = {}
        
        for key in data.files:
            if key.startswith('scalar_'):
                state[key[7:]] = data[key].item()
            elif key == 'metadata_json':
                state['metadata'] = json.loads(str(data[key]))
            else:
                state[key] = data[key]
        
        return state
    
    def save_trajectory(self,
                        states: List[Dict],
                        name: str,
                        metadata: Optional[Dict] = None) -> str:
        """
        Save a complete trajectory (multiple states).
        
        Args:
            states: List of state dictionaries
            name: Name for this trajectory
            metadata: Optional metadata
            
        Returns:
            Path to saved trajectory
        """
        filepath = self.checkpoint_dir / f"{name}_trajectory.h5"
        
        with h5py.File(filepath, 'w') as f:
            n_steps = len(states)
            f.attrs['n_steps'] = n_steps
            
            # Get dimensions from first state
            n_particles = states[0]['positions'].shape[0]
            
            # Create datasets for trajectories
            pos_data = f.create_dataset('positions', 
                                        shape=(n_steps, n_particles, 3),
                                        dtype='float64',
                                        compression='gzip')
            vel_data = f.create_dataset('velocities',
                                        shape=(n_steps, n_particles, 3),
                                        dtype='float64',
                                        compression='gzip')
            acc_data = f.create_dataset('accelerations',
                                        shape=(n_steps, n_particles, 3),
                                        dtype='float64',
                                        compression='gzip')
            
            times = []
            steps = []
            
            # Fill datasets
            for i, state in enumerate(states):
                pos_data[i] = state['positions']
                vel_data[i] = state['velocities']
                acc_data[i] = state['accelerations']
                times.append(state.get('time', i))
                steps.append(state.get('step', i))
            
            f.create_dataset('times', data=np.array(times))
            f.create_dataset('steps', data=np.array(steps))
            f.create_dataset('masses', data=states[0]['masses'])
            
            # Metadata
            if metadata:
                meta_group = f.create_group('metadata')
                for key, value in metadata.items():
                    if isinstance(value, (int, float, str)):
                        meta_group.attrs[key] = value
                    else:
                        meta_group.attrs[key] = json.dumps(value)
            
            f.attrs['created_at'] = datetime.now().isoformat()
        
        return str(filepath)
    
    def load_trajectory(self, name: str) -> Dict:
        """
        Load a trajectory.
        
        Args:
            name: Trajectory name (without _trajectory.h5)
            
        Returns:
            Dictionary with trajectory data
        """
        filepath = self.checkpoint_dir / f"{name}_trajectory.h5"
        
        if not filepath.exists():
            raise FileNotFoundError(f"Trajectory '{name}' not found")
        
        with h5py.File(filepath, 'r') as f:
            trajectory = {
                'positions': f['positions'][:],
                'velocities': f['velocities'][:],
                'accelerations': f['accelerations'][:],
                'times': f['times'][:],
                'steps': f['steps'][:],
                'masses': f['masses'][:],
                'n_steps': f.attrs['n_steps']
            }
            
            if 'metadata' in f:
                trajectory['metadata'] = {}
                for key in f['metadata'].attrs.keys():
                    value = f['metadata'].attrs[key]
                    try:
                        trajectory['metadata'][key] = json.loads(value)
                    except (json.JSONDecodeError, TypeError):
                        trajectory['metadata'][key] = value
        
        return trajectory
    
    def list_checkpoints(self) -> List[str]:
        """List all available checkpoints."""
        checkpoints = []
        
        for f in self.checkpoint_dir.iterdir():
            if f.suffix in ['.h5', '.npz']:
                name = f.stem.replace('_trajectory', ' (trajectory)')
                checkpoints.append(name)
        
        return sorted(checkpoints)
    
    def trajectory_exists(self, name: str) -> bool:
        """Check if a trajectory file exists."""
        filepath = self.checkpoint_dir / f"{name}_trajectory.h5"
        return filepath.exists()

    
    def delete_checkpoint(self, name: str) -> bool:
        """Delete a checkpoint."""
        for ext in ['.h5', '.npz', '_trajectory.h5']:
            filepath = self.checkpoint_dir / f"{name}{ext}"
            if filepath.exists():
                filepath.unlink()
                return True
        return False


def create_training_dataset(trajectories: List[Dict],
                           output_path: str,
                           sequence_length: int = 10,
                           stride: int = 1,
                           masses: Optional[np.ndarray] = None) -> str:
    """
    Create a training dataset from multiple trajectories.
    
    Uses chunked HDF5 writing to minimize memory usage.
    
    Generates (input_sequence, target) pairs for training.
    
    Args:
        trajectories: List of trajectory dictionaries
        output_path: Path to save the dataset
        sequence_length: Number of timesteps in input sequence
        stride: Stride between samples
        
    Returns:
        Path to saved dataset
    """
    # First pass: count total samples to pre-allocate HDF5 dataset
    total_samples = 0
    sample_shape_input = None
    sample_shape_target = None
    
    for traj in trajectories:
        n_steps = traj['n_steps']
        n_samples = max(0, (n_steps - sequence_length) // stride)
        total_samples += n_samples
        
        if sample_shape_input is None and n_samples > 0:
            n_particles = traj['positions'].shape[1]
            sample_shape_input = (sequence_length, n_particles, 6)
            sample_shape_target = (n_particles, 6)
    
    if total_samples == 0:
        raise ValueError("No samples could be created from trajectories")
    
    # Create output file with pre-allocated datasets
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with h5py.File(output_path, 'w') as f:
        # Create chunked, resizable datasets
        inputs_ds = f.create_dataset(
            'inputs',
            shape=(total_samples,) + sample_shape_input,
            dtype='float32',
            compression='gzip',
            compression_opts=4,
            chunks=(min(100, total_samples),) + sample_shape_input
        )
        
        targets_ds = f.create_dataset(
            'targets',
            shape=(total_samples,) + sample_shape_target,
            dtype='float32',
            compression='gzip',
            compression_opts=4,
            chunks=(min(100, total_samples),) + sample_shape_target
        )
        
        # Second pass: write samples directly to HDF5
        sample_idx = 0
        for traj in trajectories:
            positions = traj['positions']
            velocities = traj['velocities']
            n_steps = traj['n_steps']
            
            for i in range(0, n_steps - sequence_length, stride):
                # Input: sequence of states
                input_seq = np.concatenate([
                    positions[i:i+sequence_length],
                    velocities[i:i+sequence_length]
                ], axis=-1).astype(np.float32)
                
                # Target: next state
                target = np.concatenate([
                    positions[i+sequence_length],
                    velocities[i+sequence_length]
                ], axis=-1).astype(np.float32)
                
                inputs_ds[sample_idx] = input_seq
                targets_ds[sample_idx] = target
                sample_idx += 1
        
        f.attrs['sequence_length'] = sequence_length
        f.attrs['n_samples'] = total_samples
        f.attrs['created_at'] = datetime.now().isoformat()
        
        # Save masses if provided
        if masses is not None:
            f.create_dataset('masses', data=masses.astype(np.float32))
    
    print(f"Created dataset with {total_samples} samples at {output_path}")
    return str(output_path)

