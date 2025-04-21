import json
import math
import os
import numpy as np
from core.ml_data_generation_methods import save_voxelized_data_as_json
from core.voxelization_helper_methods import build_pore_data_dict, voxelize_dat_particles
from scipy.io import loadmat


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
        domain_type = parsed_data["domain_type"]

        if domain_type == "particle":
            return (
                parsed_data["particles"],
                parsed_data["particle_surfaces"],
                parsed_data["particles_metadata"],
                parsed_data["voxel_size"],
                parsed_data["domain_size"],
                "particle"
            )
        elif domain_type == "pore":
            return (
                parsed_data["pores"],
                parsed_data["pore_surfaces"],
                parsed_data["pores_metadata"],
                parsed_data["voxel_size"],
                parsed_data["domain_size"],
                "pore"
            )
        # return (
        #     parsed_data["particles"],
        #     parsed_data["voxel_size"],
        #     parsed_data["domain_size"],
        # )
    elif selected_file.endswith(".npz"):
        parsed_data = parse_npz_file(selected_file)
        return (
            parsed_data["particles"],
            parsed_data["voxel_size"],
            parsed_data["domain_size"],
        )
    elif selected_file.endswith(".mat"):
        parsed_data = parse_mat_file(selected_file)
        return (
            parsed_data["pores"],
            parsed_data["pores_metadata"],
            parsed_data["voxel_size"],
            parsed_data["domain_size"],
        )
    else:
        raise ValueError(f"Unsupported file type for {selected_file}")

def detect_domain_type(json_data):
    if "bead_data" in json_data:
        return "particle"
    elif "pores" in json_data:
        return "pore"
    else:
        raise ValueError("Unrecognized domain type: expected 'bead_data' or 'pores'.")
    
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
    Parse a JSON file to extract voxel data for particles or pores and domain info.

    Returns:
        dict with:
            - 'domain_type': 'particle' or 'pore'
            - 'voxel_size': float
            - 'domain_size': tuple of ints
            - 'voxel_count': int
            - 'particles' or 'pores': dict[label → np.array of voxel indices]
            - 'particle_surfaces' or 'pore_surfaces': dict[label → np.array of surface voxel indices]
    """
    data = load_raw_json_data(filepath)
    domain_type = detect_domain_type(data)
    domain_data = parse_raw_domain_data(data, domain_type)

    voxel_size = domain_data["voxel_size"]
    domain_size = domain_data["domain_size"]
    voxel_count = domain_data["voxel_count"]
    voxel_dict = domain_data["voxel_dict"]

    # Output keys
    data_key = "particles" if domain_type == "particle" else "pores"
    surface_key = "particle_surfaces" if domain_type == "particle" else "pore_surfaces"
    metadata_key = "particles_metadata" if domain_type == "particle" else "pores_metadata"
 
    # Parse voxel ranges into flattened lists and surface voxels
    structured_data = {}
    surface_data = {}

    for label, voxel_ranges in voxel_dict.items():
        voxel_indices = []
        surface_indices = []
        for start, end in voxel_ranges:
            # Adjust for 1-based MATLAB-style indexing
            start -= 1
            end -= 1
            voxel_indices.extend(range(start, end + 1))
            surface_indices.extend([start, end])
        structured_data[label] = np.array(voxel_indices, dtype=np.int32)
        surface_data[label] = np.unique(surface_indices)

    return {
        "domain_type": domain_type,
        "voxel_size": voxel_size,
        "domain_size": domain_size,
        "voxel_count": voxel_count,
        data_key: structured_data,
        surface_key: surface_data,
        metadata_key: data.get("pores_metadata", {}) if domain_type == "pore" else None
    }

# def parse_json_file(filepath):
#     """
#     Parse a JSON file to extract voxel data for particles and domain information.

#     Parameters:
#         filepath (str): Path to the JSON file.

#     Returns:
#         dict: A dictionary containing:
#             - 'voxel_size': The size of each voxel.
#             - 'domain_size': The shape of the 3D domain (x, y, z dimensions).
#             - 'particles': A dictionary mapping particle labels to their 3D voxel positions.
#             - 'particle_surfaces': A dictionary mapping particle labels to their surface voxel indices.
#     """
#     data = load_raw_json_data(filepath)
#     domain_data = parse_raw_domain_data(data)
#     voxel_size = domain_data["voxel_size"]
#     domain_size = domain_data["domain_size"]

#     domain_type = detect_domain_type(data)

#     voxel_count = domain_data["voxel_count"]
#     bead_data = domain_data["bead_data"]

#     # Parse particle data
#     particles = {}
#     particle_surfaces = {}
#     for particle_label, voxel_ranges in bead_data.items():
#         voxel_indices = []
#         surface_indices = []
#         for voxel_range in voxel_ranges:
#             start, end = voxel_range
#             # Adjust for MATLAB's 1-based indexing
#             start -= 1
#             end -= 1
#             voxel_indices.extend(range(start, end + 1))  # Include the end index
#             surface_indices.extend([start, end])  # Add the first and last indices of the range
#         particles[particle_label] = np.array(voxel_indices)
#         particle_surfaces[particle_label] = np.unique(surface_indices)  # Ensure uniqueness

#     return {
#         "voxel_size": voxel_size,
#         "domain_size": domain_size,
#         "voxel_count": voxel_count,
#         "particles": particles,
#         "particle_surfaces": particle_surfaces
#     }

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

def parse_mat_file(filepath):
    data = loadmat(filepath)
    pore_data = parse_raw_pore_data(data)
    
    voxel_size = pore_data["voxel_size"]
    domain_size = pore_data["domain_size"]
    pore_structs = pore_data["pore_data"]
    
    #domain_size = [domain_size[0][1], domain_size[0][3], domain_size[0][5]]
    domain_size = [domain_size[0][0], domain_size[0][1], domain_size[0][2], domain_size[0][3], domain_size[0][4], domain_size[0][5]]
    pores_full = [matlab_struct_to_dict(p) for p in pore_structs.flat]
    pore_ranges = build_pore_data_dict(pores_full)

    pores_metadata = {}
    for pore in pores_full:
        pore_id = pore["uniqueID"]
        metadata = {k: v for k, v in pore.items() if k != "indices"}
        pores_metadata[pore_id] = metadata

    return {
        "voxel_size": voxel_size[0][0],
        "domain_size": tuple(domain_size),
        "pores": pore_ranges,
        "pores_metadata": pores_metadata
    }

def matlab_struct_to_dict(mat_struct, exclude_fields=None):
    """
    Recursively convert a MATLAB struct to a Python dictionary.
    Optionally skip certain fields for custom handling.
    """
    exclude_fields = exclude_fields or []
    result = {}

    for field in mat_struct.dtype.names:
        if field in exclude_fields:
            result[field] = mat_struct[field]  # raw access
            continue

        elem = mat_struct[field][0, 0] if mat_struct[field].ndim == 2 else mat_struct[field]
        result[field] = parse_value(elem)

    return result

def parse_value(value):
    """
    Clean individual MATLAB values to Python types.
    """
    if isinstance(value, np.ndarray):
        if value.dtype == 'O':  # Possibly nested struct or cell array
            if value.size == 1:
                return parse_value(value.item())
            else:
                return [parse_value(v) for v in value.flat]
        else:
            # Try to squeeze and convert scalar arrays
            value = np.squeeze(value)
            if value.ndim == 0:
                return value.item()
            else:
                return value
    elif isinstance(value, bytes):
        return value.decode("utf-8")
    else:
        return value

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

def parse_raw_domain_data(data, domain_type):
    """
    Extract voxel domain data and metadata from a JSON dict, based on domain_type.

    Parameters:
        data (dict): Raw loaded JSON.
        domain_type (str): 'particle' or 'pore'

    Returns:
        dict with:
            - voxel_size
            - domain_size
            - voxel_count
            - voxel_dict (raw range-based representation)
    """
    voxel_size = data.get("voxel_size")
    domain_size = tuple(int(d) for d in data.get("domain_size", ()))
    voxel_count = data.get("voxel_count", None)

    # if not all(isinstance(d, (int, float)) and d > 0 for d in domain_size):
    #     raise ValueError(f"Invalid domain_size: {domain_size}")
    
    if not all(isinstance(d, (int, float)) for d in domain_size):
        raise ValueError(f"Invalid domain_size: {domain_size}")

    if domain_type == "particle":
        voxel_dict = data.get("bead_data") or data.get("beads")
    elif domain_type == "pore":
        voxel_dict = data.get("pores")
    else:
        raise ValueError(f"Unsupported domain_type: {domain_type}")

    if voxel_dict is None:
        raise KeyError(f"Missing voxel data for domain_type '{domain_type}'")

    return {
        "voxel_size": voxel_size,
        "domain_size": domain_size,
        "voxel_count": voxel_count,
        "voxel_dict": voxel_dict
    }

# def parse_raw_domain_data(data):
#     # Extract relevant data
#     voxel_size = data.get("voxel_size")
#     domain_size = tuple(data.get("domain_size", ()))  # Ensure it's a tuple for easier usage
#     voxel_count = data.get("voxel_count", None)  # Optional, not always required
#     bead_data = data.get("bead_data", data.get("beads", None))

#     # Validate domain_size
#     if not all(isinstance(dim, (int, float)) and dim > 0 for dim in domain_size):
#         raise ValueError(f"Invalid domain_size: {domain_size}")
    
#     # Try extracting 'bead_data' first, then fall back to 'beads'
#     if bead_data is None:
#         raise KeyError("JSON file is missing both 'bead_data' and 'beads' keys.")

#     # Convert domain_size to integers if necessary
#     domain_size = tuple(int(dim) for dim in domain_size)

#     return {
#         "voxel_size": voxel_size,
#         "domain_size": domain_size,
#         "voxel_count": voxel_count,
#         "bead_data": bead_data
#     }

def parse_raw_pore_data(data):
    # Extract relevant data
    voxel_size = data.get("dx")
    domain_size = tuple(data.get("domain", ()))  # Ensure it's a tuple for easier usage
    pore_data = data.get("subunits")

    if pore_data is None:
        raise KeyError("MAT file is missing 'subunits' key.")
    if voxel_size is None:
        raise KeyError("MAT file is missing 'dx' key.")
    if domain_size is None:
        raise KeyError("MAT file is missing 'domain' key.")
    
    return {
        "voxel_size": voxel_size,
        "domain_size": domain_size,
        "pore_data": pore_data
    }

def to_serializable(obj):
    if isinstance(obj, (np.integer, np.uint8, np.int32, np.int64)):
        return int(obj)
    elif isinstance(obj, (float, np.floating, np.float32, np.float64)):
        val = float(obj)
        if math.isnan(val) or math.isinf(val):
            return None
        return val
    elif isinstance(obj, (np.ndarray,)):
        return [to_serializable(v) for v in obj.tolist()]
    elif isinstance(obj, (bytes, np.bytes_)):
        return obj.decode('utf-8')
    elif isinstance(obj, dict):
        return {k: to_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [to_serializable(v) for v in obj]
    else:
        return obj  # fallback
