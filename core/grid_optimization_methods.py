import time
import numpy as np
import open3d as o3d
from scipy.spatial import cKDTree

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
    Extracts surface voxels for particles in a 3D voxel grid, ensuring no wraparound artifacts.
    
    Parameters:
        particles (dict): Keys are particle labels, values are 1D voxel indices of each particle.
        grid_size (tuple): The grid dimensions (nx, ny, nz).
    
    Returns:
        dict: A dictionary where keys are particle labels and values are the 1D indices of surface voxels.
    """
    surface_extraction_start_time = time.time()

    nx, ny, nz = grid_size
    flat_grid_size = nx * ny * nz

    # Define neighbor offsets (6-connected neighborhood)
    neighbor_offsets = np.array([-1, +1, -nx, +nx, -nx * ny, +nx * ny], dtype=np.int32)

    surface_particles = {}

    for label, voxel_indices in particles.items():
        if len(voxel_indices) == 0:
            continue

        # Create boolean mask of particle membership
        voxel_mask = np.zeros(flat_grid_size, dtype=bool)
        voxel_mask[voxel_indices] = True

        # Get voxel coordinates (x, y, z)
        voxel_x = voxel_indices % nx
        voxel_y = (voxel_indices // nx) % ny
        voxel_z = (voxel_indices // (nx * ny)) % nz

        # **🔥 PREVENT Wraparound Before Applying Neighbor Offsets**
        valid_x_neg = voxel_x > 0  # Can move left (-1)
        valid_x_pos = voxel_x < nx - 1  # Can move right (+1)
        valid_y_neg = voxel_y > 0  # Can move forward (-nx)
        valid_y_pos = voxel_y < ny - 1  # Can move backward (+nx)
        valid_z_neg = voxel_z > 0  # Can move down (-nx * ny)
        valid_z_pos = voxel_z < nz - 1  # Can move up (+nx * ny)

        # Build valid neighbor mask for each offset
        valid_neighbors_mask = np.stack([
            valid_x_neg, valid_x_pos, 
            valid_y_neg, valid_y_pos, 
            valid_z_neg, valid_z_pos
        ], axis=1)  # Shape: (N, 6)

        # Apply offsets **only where valid**
        neighbors = np.where(valid_neighbors_mask, voxel_indices[:, None] + neighbor_offsets, -1)

        # **🔥 STRICT PREVENTION of x=0 WRAPAROUND**
        no_x_wraparound = (
            (voxel_x[:, None] > 0) | (neighbor_offsets != -1)
        ) & (
            (voxel_x[:, None] < nx - 1) | (neighbor_offsets != +1)
        )

        # **🔥 Apply wraparound fix strictly at x=0**
        valid_neighbors_mask &= no_x_wraparound

        # Apply final filtering
        valid_neighbors = np.where(valid_neighbors_mask, neighbors, -1)

        # Lookup only valid neighbors in voxel_mask
        valid_lookup_mask = valid_neighbors >= 0
        valid_neighbors_present = np.zeros_like(valid_neighbors, dtype=bool)
        valid_neighbors_present[valid_lookup_mask] = voxel_mask[valid_neighbors[valid_lookup_mask]]

        # **Surface Condition: If ANY valid neighbor is missing, it's a surface voxel**
        is_surface = ~valid_neighbors_present.all(axis=1)

        # **🔥 Keep Valid Surface Voxels at `x=0`, Remove ONLY Extraneous**
        extraneous_voxels = is_surface & (voxel_x == 0) & ~valid_neighbors_present.any(axis=1)
        is_surface &= ~extraneous_voxels  # Remove only misplaced ones

        # **Final Safety Check: No Valid Voxels Were Removed**
        num_removed = np.sum(extraneous_voxels)
        if num_removed > 0:
            print(f"⚠️ Warning: Removed {num_removed} misplaced voxels at x=0.")

        # Store surface voxels
        surface_particles[label] = voxel_indices[is_surface]

    surface_extraction_end_time = time.time()
    print(f"Surface extraction completed in {surface_extraction_end_time - surface_extraction_start_time:.2f} seconds.")

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

def downsample_pointcloud(points, labels, voxel_size=1):
    """
    Downsamples the point cloud using voxel grid downsampling.

    Parameters:
        points (np.ndarray): (N, 3) array of 3D points.
        labels (np.ndarray): (N,) array of point labels.
        voxel_size (float): Voxel grid size for downsampling.

    Returns:
        tuple: (downsampled_points, downsampled_labels)
    """
    # Convert points to Open3D format
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)

    # Apply voxel downsampling
    pcd = pcd.voxel_down_sample(voxel_size=voxel_size)

    # Retrieve downsampled points
    downsampled_points = np.asarray(pcd.points)

    # Find nearest labels for downsampled points (approximate)
    tree = cKDTree(points)
    _, nearest_indices = tree.query(downsampled_points, k=1)
    downsampled_labels = labels[nearest_indices]

    return downsampled_points, downsampled_labels