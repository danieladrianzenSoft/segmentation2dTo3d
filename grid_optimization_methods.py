import time
import numpy as np

def process_particles(particles, voxel_centers, grid_size, downsample_factor=1.0, max_particles=None):
    """
    Process particles by applying downsampling and/or surface extraction.

    Parameters:
        particles (dict): Dictionary mapping particle labels to their voxel indices.
        voxel_centers (ndarray): Array of voxel center coordinates.
        grid_size (tuple): Shape of the 3D grid.
        downsample_factor (float): Fraction of voxels to retain for each particle (0 < factor <= 1).
        surface_only (bool): If True, only retain surface voxels of particles.
        max_particles (int or None): Limit the number of particles to process.

    Returns:
        dict: Processed particle data.
    """
    # Limit the number of particles
    if max_particles is not None:
        particles = dict(list(particles.items())[:max_particles])

    # Apply downsampling if needed
    if downsample_factor < 1.0:
        print(f"Downsampling particles to {downsample_factor * 100:.1f}%...")
        particles = downsample_particles(particles, downsample_factor)

    return particles


def extract_surface_voxels(particles, grid_size):
    """
    Efficiently extract surface voxels for particles using 1D voxel indices,
    leveraging a boolean mask and batch processing for neighbor checks.

    Parameters:
        particles (dict): A dictionary where keys are particle labels and values are 1D voxel indices belonging to each particle.
        grid_size (tuple): The size of the grid as (nx, ny, nz).

    Returns:
        dict: A dictionary where keys are particle labels and values are the 1D indices of surface voxels.
    """

    # Start timing surface extraction
    surface_extraction_start_time = time.time()

    nx, ny, nz = grid_size
    flat_grid_size = nx * ny * nz

    # Precompute 1D neighbor offsets
    neighbor_offsets = np.array([-1, +1, -nx, +nx, -nx * ny, +nx * ny])

    surface_particles = {}

    for label, voxel_indices in particles.items():
        if len(voxel_indices) == 0:
            continue

        # Validate voxel_indices
        if np.any((voxel_indices < 0) | (voxel_indices >= flat_grid_size)):
            print(f"Out-of-bounds voxel indices detected: {voxel_indices}")

        # Create a boolean mask for voxel membership
        voxel_mask = np.zeros(flat_grid_size, dtype=bool)
        voxel_mask[voxel_indices] = True

        # Expand voxel indices to include neighbors
        neighbors = voxel_indices[:, None] + neighbor_offsets  # Shape: (N, 6)

        # Filter out-of-bounds neighbors
        valid_neighbors = np.clip(neighbors, 0, flat_grid_size - 1)  # Prevent out-of-bounds access

        # # Filter out-of-bounds neighbors
        # valid_neighbors_mask = (neighbors >= 0) & (neighbors < flat_grid_size)
        # valid_neighbors = neighbors[valid_neighbors_mask]  # Flattened array of valid neighbors

        # Identify surface voxels
        # A voxel is on the surface if any of its neighbors are not in the particle
        is_surface = ~voxel_mask[valid_neighbors].reshape(len(voxel_indices), -1).all(axis=1)

        # Store surface voxels for this particle
        surface_particles[label] = voxel_indices[is_surface]
    
    # End surface extraction timer
    surface_extraction_end_time = time.time()
    surface_extraction_duration = surface_extraction_end_time - surface_extraction_start_time
    print(f"Time taken for surface voxel extraction: {surface_extraction_duration:.2f} seconds")

    return surface_particles

def downsample_particles(particles, factor=0.1):
    """
    Downsample the voxel data for each particle.

    Parameters:
        particles (dict): Dictionary mapping particle labels to their voxel indices.
        factor (float): Fraction of voxels to retain for each particle (0 < factor <= 1).

    Returns:
        dict: A dictionary with downsampled voxel indices for each particle.
    """
    if not (0 < factor <= 1):
        raise ValueError("Downsampling factor must be between 0 and 1.")
    
    downsampled_particles = {}
    for particle_label, voxel_indices in particles.items():
        n_samples = max(1, int(len(voxel_indices) * factor))  # At least 1 sample
        sampled_indices = np.random.choice(voxel_indices, size=n_samples, replace=False)
        downsampled_particles[particle_label] = sampled_indices

    return downsampled_particles