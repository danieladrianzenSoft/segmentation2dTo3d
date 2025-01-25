import json
import os
import numpy as np
from helper_methods import voxelize_dat_particles

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
        voxel_data = voxelize_dat_particles(
            centers, radii, voxel_size=config["voxelization_dx"]
        )
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
        for line in file:
            line = line.strip()
            if line.startswith('#'):
                continue  # Skip any line starting with '#'
            elif line:  # Process non-empty lines
                values = list(map(float, line.split()))
                if len(values) == 4:
                    x, y, z, r = values
                    centers.append([x, y, z])
                    radii.append(r)
                    
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
    with open(filepath, 'r') as file:
        data = json.load(file)

    # Handle list structure if the JSON root is a list
    if isinstance(data, list):
        if len(data) == 0:
            raise ValueError("JSON file is empty or contains an empty list.")
        data = data[0]  # Access the first element (assuming it contains the dictionary)

    # Validate required keys
    required_keys = ["voxel_size", "domain_size", "bead_data"]
    if not all(key in data for key in required_keys):
        raise KeyError(f"JSON file is missing required keys: {required_keys}")

    # Extract relevant data
    voxel_size = data["voxel_size"]
    domain_size = tuple(data["domain_size"])  # Ensure it's a tuple for easier usage
    voxel_count = data.get("voxel_count", None)  # Optional, not always required
    bead_data = data["bead_data"]

    # Validate domain_size
    if not all(isinstance(dim, int) and dim > 0 for dim in domain_size):
        raise ValueError(f"Invalid domain_size: {domain_size}")

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