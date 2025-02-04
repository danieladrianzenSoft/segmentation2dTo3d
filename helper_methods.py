import random
import time
from scipy.ndimage import label
import numpy as np
import warnings
import math
from scipy.spatial import cKDTree

from grid_optimization_methods import downsample_particles

def get_centered_grid(bounds, dx):
    """
    Generate the 3D grid of voxel centers and calculate the grid size. Maps to real 3d coordinates.

    Parameters:
        bounds (tuple): The bounds of the domain in real space as (xMin, xMax, yMin, yMax, zMin, zMax).
        dx (float): The size of each voxel in real space.

    Returns:
        tuple: A tuple containing:
            - numpy.ndarray: Array of voxel centers with shape (N, 3), where N is the total number of voxels.
            - tuple: Grid size (nz, ny, nx) corresponding to the number of grid points along each dimension.
    """
    x_min, x_max, y_min, y_max, z_min, z_max = bounds

    # Calculate the grid size
    nx = int((x_max - x_min) / dx)
    ny = int((y_max - y_min) / dx)
    nz = int((z_max - z_min) / dx)
    grid_size = (nx, ny, nz)

    # Generate voxel centers directly
    z = np.linspace(z_min + dx / 2, z_max - dx / 2, nz)
    y = np.linspace(y_min + dx / 2, y_max - dx / 2, ny)
    x = np.linspace(x_min + dx / 2, x_max - dx / 2, nx)

    # Create the 3D grid
    zzz, yyy, xxx = np.meshgrid(z, y, x, indexing="ij")

    # Flatten and combine into voxel centers
    voxel_centers = np.column_stack((xxx.ravel(),  # Flatten in column-major order for z-first
                                     yyy.ravel(),
                                     zzz.ravel()))
    
    return voxel_centers, grid_size

def voxelize_dat_particles(centers, radii, voxel_size=1):
    """
    Voxelize spherical particles from .dat file data into a structured grid.
    
    Parameters:
        centers (numpy.ndarray): Array of x, y, z coordinates of particle centers.
        radii (numpy.ndarray): Array of particle radii.
        voxel_size (float): Size of each voxel. Default is 1.
    
    Returns:
        dict: A dictionary containing:
            - 'voxel_size': The size of each voxel.
            - 'domain_size': The shape of the 3D domain (nx, ny, nz).
            - 'voxel_count': Total number of voxels in the domain.
            - 'particles': A dictionary mapping particle IDs to 1D indices of voxels.
    """
    # Start timing for voxelization
    voxelization_start_time = time.time()

    # Determine the bounds of the grid from the particle data
    x_min = np.floor(centers[:, 0] - radii).min()
    x_max = np.ceil(centers[:, 0] + radii).max()
    y_min = np.floor(centers[:, 1] - radii).min()
    y_max = np.ceil(centers[:, 1] + radii).max()
    z_min = np.floor(centers[:, 2] - radii).min()
    z_max = np.ceil(centers[:, 2] + radii).max()

    # Define the grid bounds
    bounds = (x_min, x_max, y_min, y_max, z_min, z_max)

    # Generate the grid of voxel centers and grid size
    voxel_centers, grid_size = get_centered_grid(bounds, voxel_size)
    nx, ny, nz = grid_size  # grid_size is now (nx, ny, nz)
    domain_size = (nx * voxel_size, ny * voxel_size, nz * voxel_size)  # Real-world dimensions (x, y, z)
    flat_grid_size = nx * ny * nz

    # Map 3D indices to 1D indices
    def coords_to_index(x, y, z):
        return x + y * nx + z * (nx * ny)

    particles = {}

    # Voxelize each particle
    for i, (center, radius) in enumerate(zip(centers, radii)):
        # Calculate the bounding box for the particle in voxel coordinates
        x_min_vox = int((center[0] - radius - bounds[0]) / voxel_size)
        x_max_vox = int((center[0] + radius - bounds[0]) / voxel_size)
        y_min_vox = int((center[1] - radius - bounds[2]) / voxel_size)
        y_max_vox = int((center[1] + radius - bounds[2]) / voxel_size)
        z_min_vox = int((center[2] - radius - bounds[4]) / voxel_size)
        z_max_vox = int((center[2] + radius - bounds[4]) / voxel_size)

        # Generate grid coordinates for the bounding box
        x_range = np.arange(x_min_vox, x_max_vox + 1)
        y_range = np.arange(y_min_vox, y_max_vox + 1)
        z_range = np.arange(z_min_vox, z_max_vox + 1)
        xx, yy, zz = np.meshgrid(x_range, y_range, z_range, indexing="ij")

        # Flatten the coordinates
        voxel_coords = np.column_stack((xx.ravel(), yy.ravel(), zz.ravel()))

        # Map voxel coordinates back to real space
        voxel_positions = voxel_coords * voxel_size + np.array([bounds[0], bounds[2], bounds[4]])

        # Calculate the distance from each voxel to the sphere center
        distances = np.linalg.norm(voxel_positions - center, axis=1)

        # Keep only voxels inside the sphere
        inside_sphere = distances <= radius
        voxel_coords = voxel_coords[inside_sphere]

        # Filter out-of-bounds voxel coordinates
        valid_mask = (
            (voxel_coords[:, 0] >= 0) & (voxel_coords[:, 0] < nx) &
            (voxel_coords[:, 1] >= 0) & (voxel_coords[:, 1] < ny) &
            (voxel_coords[:, 2] >= 0) & (voxel_coords[:, 2] < nz)
        )
        voxel_coords = voxel_coords[valid_mask]

        # Convert valid voxel coordinates to 1D indices
        voxel_indices = coords_to_index(voxel_coords[:, 0], voxel_coords[:, 1], voxel_coords[:, 2])

        # # Ensure indices are within bounds
        # assert np.all((voxel_indices >= 0) & (voxel_indices < flat_grid_size)), "Out-of-bounds indices found!"
        # voxel_indices = voxel_indices[(voxel_indices >= 0) & (voxel_indices < flat_grid_size)]

        # Store the voxel indices for the particle
        particles[str(i + 1)] = voxel_indices

    # Calculate total voxel count
    voxel_count = len(voxel_centers)

    # End voxelization timer
    voxelization_end_time = time.time()
    voxelization_duration = voxelization_end_time - voxelization_start_time
    print(f"Time taken for voxelization: {voxelization_duration:.2f} seconds")

    return {
        "voxel_size": voxel_size,
        "domain_size": domain_size,
        "voxel_count": voxel_count,
        "particles": particles,
    }

def determine_axis(config):
    """
    Selects an axis for slicing based on the configuration.

    Parameters:
        config (dict): Configuration dictionary with keys:
                       - "random_axis" (bool): Whether to choose the axis randomly.
                       - "axis" (str): Default axis to use if not random.

    Returns:
        str: The chosen axis ('x', 'y', or 'z').
    """
    if config.get("random_axis", False):
        chosen_axis = random.choice(['x', 'y', 'z'])
        print(f"Randomly selected axis for slicing: {chosen_axis}")
    else:
        chosen_axis = config.get("axis", 'z')
        print(f"Using configured axis for slicing: {chosen_axis}")
    return chosen_axis


def determine_num_slices(grid_size, voxel_size, config):
    """
    Determine the number of slices to extract based on the configuration.

    Parameters:
        config (dict): Configuration dictionary containing:
                       - grid_size
                       - voxel_size
                       - random_slice_spacing
                       - num_slices
                       - max_slices
                       - axis

    Returns:
        int: Number of slices to extract.
    """
    if config.get("random_slice_spacing") is not None:
        random_slice_spacing = config["random_slice_spacing"]

        if random_slice_spacing is True:
            slice_unit_spacing = random.randint(2, 6)
            num_slices = calculate_num_slices(
                grid_size=grid_size,
                voxel_size=voxel_size,
                slice_unit_spacing=slice_unit_spacing,
                max_slices=config["max_slices"],
                axis=config["axis"]
            )

            return num_slices, slice_unit_spacing
    
    return config.get("num_slices", 0), None

def calculate_num_slices(grid_size, voxel_size, slice_unit_spacing=None, max_slices=150, axis='z'):
    """
    Calculate the number of slices based on slice_unit_spacing or grid size.

    Parameters:
        grid_size (tuple): Shape of the 3D grid (nx, ny, nz).
        voxel_size (float): Size of each voxel in real-world units.
        slice_unit_spacing (int or None): Real domain spacing between slices in voxel units.
        max_slices (int): Maximum allowable number of slices.
        axis (str): Axis along which the slices are taken ('x', 'y', 'z').

    Returns:
        int: Number of slices.
    """
    axis_indices = {'x': 0, 'y': 1, 'z': 2}
    axis_index = axis_indices[axis]

    # Get the grid size along the specified axis
    grid_length = grid_size[axis_index]

    if slice_unit_spacing is not None:
        # Ensure slice_unit_spacing is a valid random integer between 2 and 10
        # if not (1 <= slice_unit_spacing <= 5):
        #     raise ValueError("slice_unit_spacing must be an integer between 2 and 10.")
        
        # Calculate the number of slices based on spacing
        num_slices = (grid_length // slice_unit_spacing) - 1
    else:
        raise ValueError("slice_unit_spacing must be provided for this calculation.")

    # Clip the number of slices to the maximum allowed
    num_slices = min(num_slices, max_slices)
    
    print(f"Number of slices to extract: {num_slices}")

    return num_slices

def extract_slices(particles, grid_size, voxel_size, num_slices=5, axis='z'):
    """
    Extract slices at voxel indices in grid space and find intersecting particles.

    Parameters:
        particles (dict): Dictionary mapping particle labels to 1D voxel indices.
        grid_size (tuple): Shape of the 3D voxel grid (nx, ny, nz).
        voxel_size (float): Size of each voxel in real-world units.
        num_slices (int): Number of slices to extract.
        axis (str): Axis along which to extract slices ('x', 'y', 'z').

    Returns:
        list: List of 2D numpy arrays representing slices.
        list: List of real-world slice midpoints.
    """
    # Start timing slice extraction
    slice_extraction_start_time = time.time()
    
    # Map axis name to its index
    axis_indices = {'x': 0, 'y': 1, 'z': 2}
    axis_index = axis_indices[axis]

    # Determine slice shape and strides
    if axis == 'z':
        slice_shape = (grid_size[1], grid_size[0])  # (Y, X)
        stride = grid_size[0] * grid_size[1]  # Steps for moving in Z
    elif axis == 'y':
        slice_shape = (grid_size[2], grid_size[0])  # (Z, X)
        stride = grid_size[0]  # Steps for moving in Y
    elif axis == 'x':
        slice_shape = (grid_size[2], grid_size[1])  # (Z, Y)
        stride = 1  # Steps for moving in X

    # Generate evenly spaced slice indices
    slice_indices = np.linspace(0, grid_size[axis_index] - 1, num_slices, dtype=int)
    slice_midpoints = slice_indices * voxel_size + voxel_size / 2

    # Initialize all slices at once using NumPy (batch processing)
    slices = np.zeros((num_slices, *slice_shape), dtype=np.uint16)

    # Convert particle data into single NumPy arrays for efficiency
    all_particle_labels = []
    all_voxel_indices = []

    for particle_label, voxel_indices in particles.items():
        all_particle_labels.append(np.full(len(voxel_indices), particle_label, dtype=np.uint16))
        all_voxel_indices.append(voxel_indices)

    # Flatten all particles into a single NumPy array (batch operation)
    all_particle_labels = np.concatenate(all_particle_labels)
    all_voxel_indices = np.concatenate(all_voxel_indices)

    # Compute slice positions for all voxels
    voxel_slice_positions = (all_voxel_indices // stride) % grid_size[axis_index]

    # Process all slices in a vectorized manner
    for i, slice_idx in enumerate(slice_indices):
        # Find which voxels belong to this slice
        mask = voxel_slice_positions == slice_idx
        filtered_indices = all_voxel_indices[mask]
        filtered_labels = all_particle_labels[mask]

        # Ensure all three indices are defined in every case
        x_indices = (filtered_indices % grid_size[0])  # Default
        y_indices = (filtered_indices // grid_size[0]) % grid_size[1]  # Default
        z_indices = filtered_indices // (grid_size[0] * grid_size[1])  # Default

        if axis == 'z':
            # Z-axis slices → Use (Y, X)
            slices[i, y_indices, x_indices] = filtered_labels
        elif axis == 'y':
            # Y-axis slices → Use (Z, X)
            slices[i, z_indices, x_indices] = filtered_labels
        elif axis == 'x':
            # X-axis slices → Use (Z, Y)
            slices[i, z_indices, y_indices] = filtered_labels
            
    # End surface extraction timer
    slice_extraction_end_time = time.time()
    slice_extraction_duration = slice_extraction_end_time - slice_extraction_start_time
    print(f"Time taken for slice voxel extraction: {slice_extraction_duration:.2f} seconds")

    return list(slices), slice_midpoints

def extract_slices_old(particles, voxel_centers, voxel_size, grid_size, num_slices=5, axis='z'):
    """
    Extract slices at midpoint voxel indices in grid space and find intersecting particles.

    Parameters:
        particles (dict): Dictionary mapping particle labels to voxel indices.
        voxel_centers (ndarray): Array of voxel center coordinates (N x 3).
        voxel_size (float): Size of each voxel in real-world units.
        grid_size (tuple): Shape of the 3D grid (nx, ny, nz).
        num_slices (int): Number of slices to extract.
        axis (str): Axis along which to extract slices ('x', 'y', 'z').

    Returns:
        list: List of 2D numpy arrays representing slices.
        list: List of real-world z-coordinates for the slices.
    """
    # Map axis name to its index
    axis_indices = {'x': 0, 'y': 1, 'z': 2}
    axis_index = axis_indices[axis]

    # Determine orthogonal axes, numpy defines y,z
    if axis == 'z':
        slice_shape = (grid_size[1], grid_size[0])  # y vs. x
        coord_x, coord_y = 0, 1
    elif axis == 'y':
        slice_shape = (grid_size[2], grid_size[0])  # z vs. x
        coord_x, coord_y = 0, 2
    elif axis == 'x':
        slice_shape = (grid_size[2], grid_size[1])  # z vs. y
        coord_x, coord_y = 1, 2


    # Divide the domain into slabs in real-world space
    domain_min = 0
    domain_max = grid_size[axis_index] * voxel_size
    slab_boundaries = np.linspace(domain_min, domain_max, num_slices + 1)
    slice_midpoints = (slab_boundaries[:-1] + slab_boundaries[1:]) / 2  # Real-world midpoints

    # Precompute minimum values for orthogonal axes
    min_x = voxel_centers[:, coord_x].min()
    min_y = voxel_centers[:, coord_y].min()

    # Debug: Print slice indices and midpoints
    # print(f"Slice Indices (grid space): {slice_indices}")
    # print(f"Slice Midpoints (real-world): {slice_midpoints}")

    # Initialize the result
    slices = []

    # Iterate through slice indices
    for real_coord in slice_midpoints:
        # Initialize a 2D grid for this slice
        slice_data = np.zeros(slice_shape, dtype=int)

        # Iterate through particles
        for particle_label, voxel_indices in particles.items():
            # Filter voxel centers belonging to the current particle
            particle_centers = voxel_centers[voxel_indices]

            # Check if voxels are at the same grid level as the current slice
            in_slice_mask = (particle_centers[:, axis_index] >= real_coord - voxel_size / 2) & \
                            (particle_centers[:, axis_index] < real_coord + voxel_size / 2)
            coords_in_slice = particle_centers[in_slice_mask]

            # Fill the slice grid
            # for coord in coords_in_slice:
            #     x, y = int(coord[coord_x] / voxel_size), int(coord[coord_y] / voxel_size)
            #     if 0 <= x < slice_data.shape[1] and 0 <= y < slice_data.shape[0]:
            #         slice_data[y, x] = particle_label
            
            # Fill the slice grid
            for coord in coords_in_slice:
                x = int((coord[coord_x] - min_x) / voxel_size)
                y = int((coord[coord_y] - min_y) / voxel_size)
                if 0 <= x < slice_data.shape[1] and 0 <= y < slice_data.shape[0]:
                    slice_data[y, x] = particle_label


        slices.append(slice_data)

    return slices, slice_midpoints

def select_representative_slices(slices, slice_coordinates, num_to_select=5):
    """
    Select a fixed number of representative slices from the full set of slices.

    Parameters:
        slices (list): List of 2D numpy arrays representing all slices.
        slice_coordinates (list): List of real-world slice midpoints corresponding to the slices.
        num_to_select (int): Number of slices to select for plotting.

    Returns:
        tuple: A tuple containing:
               - selected_slices (list): List of 2D numpy arrays of the selected slices.
               - selected_coordinates (list): List of midpoints corresponding to the selected slices.
    """
    total_slices = len(slices)
    if num_to_select >= total_slices:
        # If fewer slices are available, return them all
        return slices, slice_coordinates

    # Select indices evenly spaced across the available slices
    selected_indices = np.linspace(0, total_slices - 1, num=num_to_select, dtype=int)

    # Extract the selected slices and their coordinates
    selected_slices = [slices[i] for i in selected_indices]
    selected_coordinates = [slice_coordinates[i] for i in selected_indices]

    return selected_slices, selected_coordinates

def validate_voxel_coordinates(coords, domain_size):
    """
    Validate that all voxel coordinates are within the domain.

    Parameters:
        coords (numpy.ndarray): Array of voxel coordinates with shape (N, 3).
        domain_size (tuple): The size of the domain (x_max, y_max, z_max).

    Returns:
        None: Prints a warning if any voxels are outside the domain.
    """
    x_max, y_max, z_max = domain_size
    invalid_x = (coords[:, 0] < 0) | (coords[:, 0] > x_max)
    invalid_y = (coords[:, 1] < 0) | (coords[:, 1] > y_max)
    invalid_z = (coords[:, 2] < 0) | (coords[:, 2] > z_max)

    invalid_voxels = invalid_x | invalid_y | invalid_z
    if np.any(invalid_voxels):
        warnings.warn(
            f"Warning: {np.sum(invalid_voxels)} voxels are outside the domain. "
            "Check your parsing or plotting logic."
        )
        # Optionally print details of invalid voxels
        invalid_coords = coords[invalid_voxels]
        print(f"Invalid voxel coordinates:\n{invalid_coords}")

def check_particle_connectivity(particles, voxel_size, grid_shape):
    """
    Check if all voxels within each particle are spatially connected.

    Parameters:
        particles (dict): Dictionary mapping particle labels to their voxel indices.
        voxel_size (float): The size of each voxel.
        grid_shape (tuple): Shape of the 3D grid (Nx, Ny, Nz dimensions).

    Returns:
        None
    """
    def create_voxel_grid(voxel_indices, grid_shape):
        """
        Create a binary 3D grid with voxels set to 1 for the given indices.

        Parameters:
            voxel_indices (numpy.ndarray): 1D array of voxel indices.
            grid_shape (tuple): Shape of the 3D grid.

        Returns:
            numpy.ndarray: Binary 3D grid with active voxels.
        """
        grid = np.zeros(grid_shape, dtype=int)
        coords = np.unravel_index(voxel_indices, grid_shape)
        grid[coords] = 1
        return grid

    disconnected_particles = []

    for particle_label, voxel_indices in particles.items():
        # Create a binary grid for the particle
        voxel_grid = create_voxel_grid(voxel_indices, grid_shape)

        # Check connectivity
        labeled_grid, num_clusters = label(voxel_grid)
        if num_clusters > 1:
            disconnected_particles.append((particle_label, num_clusters))

    # Output results
    if disconnected_particles:
        print("Disconnected particles found:")
        for particle_label, num_clusters in disconnected_particles:
            print(f"  Particle {particle_label} contains {num_clusters} disconnected components.")
    else:
        print("All particles are fully connected.")

