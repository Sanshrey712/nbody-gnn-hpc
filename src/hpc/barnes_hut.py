"""
Barnes-Hut Tree Algorithm for O(N log N) N-body simulation.

The Barnes-Hut algorithm uses an octree to approximate distant particle
groups as single massive bodies, reducing complexity from O(N²) to O(N log N).
"""

import numpy as np
from numba import jit, prange
from numba.experimental import jitclass
from numba import types
from typing import Optional, List, Tuple


# Physical constants
G = 6.67430e-11
SOFTENING = 1e-9


@jit(nopython=True, fastmath=True)
def compute_cell_properties(positions: np.ndarray, 
                            masses: np.ndarray,
                            indices: np.ndarray) -> Tuple[np.ndarray, float]:
    """
    Compute center of mass and total mass for a cell.
    
    Returns:
        (center_of_mass, total_mass)
    """
    total_mass = 0.0
    com = np.zeros(3)
    
    for idx in indices:
        if idx >= 0:  # Valid particle index
            m = masses[idx]
            total_mass += m
            com[0] += m * positions[idx, 0]
            com[1] += m * positions[idx, 1]
            com[2] += m * positions[idx, 2]
    
    if total_mass > 0:
        com /= total_mass
    
    return com, total_mass


class OctreeNode:
    """
    A node in the Barnes-Hut octree.
    
    Each node represents a cubic cell in 3D space containing
    either a single particle (leaf) or multiple particles (internal).
    """
    
    def __init__(self, center: np.ndarray, size: float):
        """
        Initialize octree node.
        
        Args:
            center: Center of the cubic cell
            size: Side length of the cell
        """
        self.center = center
        self.size = size
        
        # Node properties
        self.mass = 0.0
        self.center_of_mass = np.zeros(3)
        self.particle_index = -1  # Index of particle if leaf, -1 if internal
        
        # Children (8 octants)
        self.children: List[Optional['OctreeNode']] = [None] * 8
        self.is_leaf = True
        self.is_empty = True
        self.node_id = -1  # For flattening
    
    def get_octant(self, position: np.ndarray) -> int:
        """
        Determine which octant a position falls into.
        
        Returns:
            Octant index (0-7)
        """
        octant = 0
        if position[0] > self.center[0]:
            octant += 1
        if position[1] > self.center[1]:
            octant += 2
        if position[2] > self.center[2]:
            octant += 4
        return octant
    
    def get_child_center(self, octant: int) -> np.ndarray:
        """Get the center of a child octant."""
        offset = self.size / 4
        child_center = self.center.copy()
        
        if octant & 1:
            child_center[0] += offset
        else:
            child_center[0] -= offset
            
        if octant & 2:
            child_center[1] += offset
        else:
            child_center[1] -= offset
            
        if octant & 4:
            child_center[2] += offset
        else:
            child_center[2] -= offset
        
        return child_center
    
    def insert(self, position: np.ndarray, mass: float, index: int, 
               positions: np.ndarray, masses: np.ndarray) -> None:
        """
        Insert a particle into the octree.
        
        Args:
            position: Particle position
            mass: Particle mass
            index: Particle index in the main arrays
            positions: All particle positions (for reinsertion)
            masses: All particle masses (for reinsertion)
        """
        if self.is_empty:
            # First particle in this cell - make it a leaf
            self.particle_index = index
            self.mass = mass
            self.center_of_mass = position.copy()
            self.is_leaf = True
            self.is_empty = False
            
        elif self.is_leaf:
            # Cell already has a particle - need to subdivide
            old_index = self.particle_index
            old_pos = positions[old_index]
            old_mass = masses[old_index]
            
            self.is_leaf = False
            self.particle_index = -1
            
            # Reinsert old particle
            octant = self.get_octant(old_pos)
            if self.children[octant] is None:
                child_center = self.get_child_center(octant)
                self.children[octant] = OctreeNode(child_center, self.size / 2)
            self.children[octant].insert(old_pos, old_mass, old_index, positions, masses)
            
            # Insert new particle
            octant = self.get_octant(position)
            if self.children[octant] is None:
                child_center = self.get_child_center(octant)
                self.children[octant] = OctreeNode(child_center, self.size / 2)
            self.children[octant].insert(position, mass, index, positions, masses)
            
            # Update node mass and center of mass
            self._update_mass_properties()
            
        else:
            # Internal node - recurse into appropriate child
            octant = self.get_octant(position)
            if self.children[octant] is None:
                child_center = self.get_child_center(octant)
                self.children[octant] = OctreeNode(child_center, self.size / 2)
            self.children[octant].insert(position, mass, index, positions, masses)
            
            # Update node mass and center of mass
            self._update_mass_properties()
    
    def _update_mass_properties(self) -> None:
        """Update total mass and center of mass from children."""
        self.mass = 0.0
        self.center_of_mass = np.zeros(3)
        
        for child in self.children:
            if child is not None and not child.is_empty:
                self.mass += child.mass
                self.center_of_mass += child.mass * child.center_of_mass
        
        if self.mass > 0:
            self.center_of_mass /= self.mass
    
    def compute_acceleration(self, position: np.ndarray, 
                             theta: float,
                             softening: float = SOFTENING) -> np.ndarray:
        """
        Compute gravitational acceleration at a position using Barnes-Hut.
        
        Args:
            position: Position to compute acceleration at
            theta: Opening angle threshold
            softening: Softening parameter
            
        Returns:
            Acceleration vector
        """
        if self.is_empty:
            return np.zeros(3)
        
        # Vector from position to center of mass
        r = self.center_of_mass - position
        dist = np.sqrt(np.sum(r * r) + softening * softening)
        
        # Check if we can use the multipole approximation
        if self.is_leaf or (self.size / dist < theta):
            # Use center of mass approximation
            if dist > softening:  # Avoid self-interaction
                return G * self.mass * r / (dist ** 3)
            else:
                return np.zeros(3)
        else:
            # Need to recurse into children
            acceleration = np.zeros(3)
            for child in self.children:
                if child is not None:
                    acceleration += child.compute_acceleration(position, theta, softening)
            return acceleration


class BarnesHutTree:
    """
    Barnes-Hut octree for efficient N-body force calculation.
    
    Reduces complexity from O(N²) to O(N log N) by approximating
    distant particle groups as single massive bodies.
    """
    
    def __init__(self, 
                 positions: np.ndarray, 
                 masses: np.ndarray,
                 theta: float = 0.5):
        """
        Build Barnes-Hut tree from particle data.
        
        Args:
            positions: (N, 3) array of particle positions
            masses: (N,) array of particle masses
            theta: Opening angle threshold (smaller = more accurate, slower)
        """
        self.positions = positions
        self.masses = masses
        self.theta = theta
        self.n_particles = len(masses)
        
        # Compute bounding box
        min_coords = np.min(positions, axis=0)
        max_coords = np.max(positions, axis=0)
        center = (min_coords + max_coords) / 2
        size = np.max(max_coords - min_coords) * 1.01  # Slight padding
        
        # Build tree
        self.root = OctreeNode(center, size)
        for i in range(self.n_particles):
            self.root.insert(positions[i], masses[i], i, positions, masses)
    
    def compute_accelerations(self) -> np.ndarray:
        """
        Compute gravitational accelerations for all particles (slow Python version).
        
        Returns:
            (N, 3) array of accelerations
        """
        accelerations = np.zeros_like(self.positions)
        
        for i in range(self.n_particles):
            accelerations[i] = self.root.compute_acceleration(
                self.positions[i], self.theta
            )
        
        return accelerations

    def flatten(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Flatten the tree into arrays for JIT compilation.
        
        Returns:
            Tuple of (cell_coms, cell_masses, cell_sizes, cell_children, cell_is_leaf)
        """
        # First, count nodes to allocate arrays
        node_count = 0
        nodes_to_visit = [self.root]
        while nodes_to_visit:
            node = nodes_to_visit.pop()
            node.node_id = node_count
            node_count += 1
            for child in node.children:
                if child is not None:
                    nodes_to_visit.append(child)
        
        # Allocate arrays
        cell_coms = np.zeros((node_count, 3))
        cell_masses = np.zeros(node_count)
        cell_sizes = np.zeros(node_count)
        cell_children = np.full((node_count, 8), -1, dtype=np.int64)
        cell_is_leaf = np.zeros(node_count, dtype=np.bool_)
        
        # Populate arrays
        nodes_to_visit = [self.root]
        while nodes_to_visit:
            node = nodes_to_visit.pop()
            idx = node.node_id
            
            cell_coms[idx] = node.center_of_mass
            cell_masses[idx] = node.mass
            cell_sizes[idx] = node.size
            cell_is_leaf[idx] = node.is_leaf
            
            for i, child in enumerate(node.children):
                if child is not None:
                    cell_children[idx, i] = child.node_id
                    nodes_to_visit.append(child)
                    
        return cell_coms, cell_masses, cell_sizes, cell_children, cell_is_leaf

    def compute_accelerations_jit(self) -> np.ndarray:
        """
        Compute accelerations using JIT-compiled flattened tree.
        
        Returns:
            (N, 3) array of accelerations
        """
        # Flatten tree
        flat_tree = self.flatten()
        
        # Call JIT function
        return barnes_hut_accelerations_flat(
            self.positions,
            self.masses,
            *flat_tree,
            self.theta
        )
    
    def compute_accelerations_parallel(self, n_workers: int = None) -> np.ndarray:
        """
        Compute accelerations in parallel using multiprocessing.
        
        Args:
            n_workers: Number of worker processes
            
        Returns:
            (N, 3) array of accelerations
        """
        import multiprocessing as mp
        from functools import partial
        
        if n_workers is None:
            n_workers = mp.cpu_count()
        
        def compute_single(idx, positions, root, theta):
            return root.compute_acceleration(positions[idx], theta)
        
        # Note: For true parallelism, we'd need to serialize the tree
        # This is a simplified version that uses sequential computation
        # A production version would use shared memory or serialize/deserialize
        
        accelerations = np.zeros_like(self.positions)
        for i in range(self.n_particles):
            accelerations[i] = self.root.compute_acceleration(
                self.positions[i], self.theta
            )
        
        return accelerations


@jit(nopython=True, parallel=True, fastmath=True)
def barnes_hut_accelerations_flat(positions: np.ndarray,
                                   masses: np.ndarray,
                                   cell_coms: np.ndarray,
                                   cell_masses: np.ndarray,
                                   cell_sizes: np.ndarray,
                                   cell_children: np.ndarray,
                                   cell_is_leaf: np.ndarray,
                                   theta: float,
                                   softening: float = SOFTENING) -> np.ndarray:
    """
    Numba-compatible Barnes-Hut acceleration computation using flattened tree representation.
    
    This is an alternative implementation that can be fully JIT-compiled for maximum performance.
    
    Note: Tree must be pre-built and flattened before calling this function.
    """
    n = positions.shape[0]
    accelerations = np.zeros_like(positions)
    
    for i in prange(n):
        # Stack-based tree traversal
        stack = np.zeros(100, dtype=np.int64)  # Max tree depth
        stack_ptr = 0
        stack[stack_ptr] = 0  # Root node
        
        ax, ay, az = 0.0, 0.0, 0.0
        pi = positions[i]
        
        while stack_ptr >= 0:
            node_idx = stack[stack_ptr]
            stack_ptr -= 1
            
            if node_idx < 0:
                continue
            
            # Vector from particle to cell center of mass
            dx = cell_coms[node_idx, 0] - pi[0]
            dy = cell_coms[node_idx, 1] - pi[1]
            dz = cell_coms[node_idx, 2] - pi[2]
            
            r2 = dx*dx + dy*dy + dz*dz + softening*softening
            r = np.sqrt(r2)
            
            # Check opening angle criterion
            if cell_is_leaf[node_idx] or (cell_sizes[node_idx] / r < theta):
                # Use multipole approximation
                if r > softening:
                    factor = G * cell_masses[node_idx] / (r * r2)
                    ax += factor * dx
                    ay += factor * dy
                    az += factor * dz
            else:
                # Push children onto stack
                for c in range(8):
                    child = cell_children[node_idx, c]
                    if child >= 0:
                        stack_ptr += 1
                        stack[stack_ptr] = child
        
        accelerations[i, 0] = ax
        accelerations[i, 1] = ay
        accelerations[i, 2] = az
    
    return accelerations
