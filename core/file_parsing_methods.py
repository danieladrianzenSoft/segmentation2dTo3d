import json
import os
import numpy as np
from core.ml_data_generation_methods import save_voxelized_data_as_json
from core.voxelization_helper_methods import voxelize_dat_particles
from core.visualization_methods import plot_spheres_plotly
from scipy.ndimage import binary_erosion

def parse_file(config, selected_file):
    """
    Parse a single file and extract necessary information.

    Parameters:
        config (dict): Configuration dictionary.
        selected_file (str): Path to the selected file.

    Returns:
        tuple: (selected_file, particles, voxel_size, domain_size)
    """
    print(f"Selected: {os.path.basename(selected_file)}")

    if selected_file.endswith(".dat"):
        centers, radii = parse_dat_file(selected_file)
        if centers.size == 0 or radii.size == 0:
            print(f"Skipping processing for file {selected_file}: No valid particles found.")
            return
        voxel_data = voxelize_dat_particles(
            centers, radii, voxel_size=config["voxelization_dx"]
        )

        if config.get("dat_to_json", False):
            # Save voxelized representation as JSON
            output_dir = config.get("output_dir", "./FlattenedDataJson")
            json_path = save_voxelized_data_as_json(voxel_data, selected_file, output_dir=output_dir)
            # plot_spheres_plotly(centers, radii)

        return (
            voxel_data["particles"],
            voxel_data["voxel_size"],
            voxel_data["domain_size"],
        )

    elif selected_file.endswith(".json"):
        parsed_data = parse_json_file(selected_file)
        return (
            parsed_data["particles"],
            parsed_data["voxel_size"],
            parsed_data["domain_size"],
        )
    elif selected_file.endswith(".npz"):
        parsed_data = parse_npz_file(selected_file)
        return (
            parsed_data["particles"],
            parsed_data["voxel_size"],
            parsed_data["domain_size"],
        )
    else:
        raise ValueError(f"Unsupported file type for {selected_file}")
    
def parse_dat_file(filepath):
    """
    Parse a .dat file to extract x, y, z coordinates and radii of spheres.

    Parameters:
        filepath (str): Path to the .dat file.

    Returns:
        tuple: A tuple containing:
            - centers (numpy.ndarray): Array of x, y, z coordinates.
            - radii (numpy.ndarray): Array of radii.
    """
    centers = []
    radii = []

    with open(filepath, 'r') as file:
        for line_number, line in enumerate(file, start=1):
            line = line.strip()

            # Skip empty lines or lines starting with non-numeric characters
            if not line or not line[0].isdigit():
                continue

            # Attempt to split and validate the line
            values = line.split()
            if len(values) != 4:
                continue

            try:
                # Convert values to floats
                x, y, z, r = map(float, values)
                centers.append([x, y, z])
                radii.append(r)
            except ValueError:
                print(f"Skipping line {line_number}: Could not convert all columns to float.")
                continue
    
    # Check if results are empty and return an empty tuple if no valid data
    if not centers or not radii:
        print(f"No valid data found in file: {filepath}. Skipping this file.")
        return np.array([]), np.array([])

    return np.array(centers), np.array(radii)

def parse_json_file(filepath):
    """
    Parse a JSON file to extract voxel data for particles and domain information.

    Parameters:
        filepath (str): Path to the JSON file.

    Returns:
        dict: A dictionary containing:
            - 'voxel_size': The size of each voxel.
            - 'domain_size': The shape of the 3D domain (x, y, z dimensions).
            - 'particles': A dictionary mapping particle labels to their 3D voxel positions.
            - 'particle_surfaces': A dictionary mapping particle labels to their surface voxel indices.
    """
    data = load_raw_json_data(filepath)
    domain_data = parse_raw_domain_data(data)
    voxel_size = domain_data["voxel_size"]
    domain_size = domain_data["domain_size"]
    voxel_count = domain_data["voxel_count"]
    bead_data = domain_data["bead_data"]

    # Parse particle data
    particles = {}
    particle_surfaces = {}
    for particle_label, voxel_ranges in bead_data.items():
        voxel_indices = []
        surface_indices = []
        for voxel_range in voxel_ranges:
            start, end = voxel_range
            # Adjust for MATLAB's 1-based indexing
            start -= 1
            end -= 1
            voxel_indices.extend(range(start, end + 1))  # Include the end index
            surface_indices.extend([start, end])  # Add the first and last indices of the range
        particles[particle_label] = np.array(voxel_indices)
        particle_surfaces[particle_label] = np.unique(surface_indices)  # Ensure uniqueness

    return {
        "voxel_size": voxel_size,
        "domain_size": domain_size,
        "voxel_count": voxel_count,
        "particles": particles,
        "particle_surfaces": particle_surfaces
    }

def parse_npz_file(filepath):
    """
    Parse an .npz file to extract voxel-based particle data and domain information.

    Parameters:
        filepath (str): Path to the .npz file.

    Returns:
        dict: A dictionary containing:
            - 'voxel_size': The size of each voxel.
            - 'domain_size': The shape of the 3D domain (x, y, z dimensions).
            - 'particles': A dictionary mapping particle labels to their 3D voxel positions.
            - 'particle_surfaces': A dictionary mapping particle labels to their surface voxel indices.
    """
    # Load .npz file
    data = np.load(filepath)

    # Extract voxel grid
    if "voxel_grid" not in data:
        raise KeyError(f"Missing 'voxel_grid' key in .npz file: {filepath}")
    voxel_grid = data["voxel_grid"]

    # Extract metadata
    voxel_size = float(data.get("voxel_size", 1.0))  # Default to 1.0 if not stored
    grid_size = tuple(data.get("grid_size", ()))  # Extracted from voxel grid

    # Validate extracted data
    if not isinstance(voxel_size, (int, float)):
        raise ValueError(f"Invalid voxel size: {voxel_size}")
    if len(grid_size) == 0:
        raise ValueError(f"Invalid grid size: {grid_size}")

    # Extract unique particle labels (excluding 0, which is background)
    unique_labels = np.unique(voxel_grid)
    unique_labels = unique_labels[unique_labels != 0]

    # Dictionary to store particle voxel indices
    particles = {}
    # particle_surfaces = {}

    for label in unique_labels:
        int_label = str(label)
        # Get all voxel indices where the label is present
        voxel_indices = np.flatnonzero(voxel_grid == label)
        particles[int_label] = voxel_indices  # Store raw voxel positions

        # Identify surface voxels
        # surface_mask = np.zeros_like(voxel_grid, dtype=bool)
        # surface_mask[tuple(voxel_indices.T)] = True  # Fill mask with this particle's voxels

        # Erode to find internal voxels, then subtract from original to get surface
        # eroded_mask = binary_erosion(surface_mask)
        # surface_voxels = voxel_indices[~eroded_mask[tuple(voxel_indices.T)]]
        # particle_surfaces[label] = surface_voxels  # Store surface positions

    return {
        "voxel_size": voxel_size,
        "domain_size": tuple(dim * voxel_size for dim in grid_size),
        "grid_size": grid_size,
        "particles": particles,
    }

def load_raw_json_data(filepath):
    with open(filepath, 'r') as f:
        data = json.load(f)

    if isinstance(data, list):
        if not data:
            raise ValueError("JSON file is empty or contains an empty list.")
        data = data[0]

    if not isinstance(data, dict):
        raise ValueError("Invalid JSON structure: Expected a dictionary at the root.")

    return data

def parse_raw_domain_data(data):
    # Extract relevant data
    voxel_size = data.get("voxel_size")
    domain_size = tuple(data.get("domain_size", ()))  # Ensure it's a tuple for easier usage
    voxel_count = data.get("voxel_count", None)  # Optional, not always required
    bead_data = data.get("bead_data", data.get("beads", None))

    # Validate domain_size
    if not all(isinstance(dim, (int, float)) and dim > 0 for dim in domain_size):
        raise ValueError(f"Invalid domain_size: {domain_size}")
    
    # Try extracting 'bead_data' first, then fall back to 'beads'
    if bead_data is None:
        raise KeyError("JSON file is missing both 'bead_data' and 'beads' keys.")

    # Convert domain_size to integers if necessary
    domain_size = tuple(int(dim) for dim in domain_size)

    return {
        "voxel_size": voxel_size,
        "domain_size": domain_size,
        "voxel_count": voxel_count,
        "bead_data": bead_data
    }