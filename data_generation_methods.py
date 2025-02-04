from datetime import datetime
import json
import os
from pathlib import Path
import time
import uuid
from matplotlib import pyplot as plt
from matplotlib.colors import ListedColormap
import cv2
import numpy as np
from PIL import Image

def save_slices_as_tiff(
        slices, 
        slice_coordinates, 
        voxel_size, 
        grid_size, 
        num_slices,
        slice_unit_spacing=None,
        axis='z', 
        output_dir_slices="slices", 
        output_dir_labels="labels",
        input_file_name=None,
        metadata_path="metadata.json",
        resolution_scale=1.0
    ):
    """
    Save slices as TIFF images and generate a metadata file.

    Parameters:
        slices (list): List of 2D numpy arrays representing slices.
        slice_coordinates (list): List of slice midpoints (real-world coordinates).
        voxel_size (float): Size of each voxel in real-world units.
        grid_size (tuple): Shape of the 3D grid (nx, ny, nz).
        axis (str): Axis along which the slices were taken ('x', 'y', 'z').
        output_dir_slices (str): Directory to save the TIFF files.
        output_dir_labels (str): Directory for the labels.
        label_file_name (str): Identifier for the original file (e.g., .dat or .json).
        num_slices (int): Number of slices extracted.
        slice_unit_spacing (int): Spacing between slices, if applicable.
        metadata_path (str): Path to the centralized metadata JSON file.
        resolution_scale (float): Scale factor for resolution (e.g., 0.8 for 80%)
    Returns:
        None
    """
        
    # Ensure output directory exists
    Path(output_dir_slices).mkdir(parents=True, exist_ok=True)

    cmap = ListedColormap(['black', '#00FF00'])  # Default fluorescent green
    greyscale_particle_color = 128
    make_gif = False
       
    # Ensure input_file_name is provided
    if not input_file_name:
        raise ValueError("input_file_name must be provided.")

    base_filename = Path(input_file_name).name  # Extract filename with extension
    base = Path(input_file_name).stem  # Extract filename without extension
    filename_label = os.path.join(output_dir_labels, f"{base}.npz")

    # Create a unique identifier for this dataset
    dataset_uuid = str(uuid.uuid4())

    # Initialize dataset-specific metadata
    dataset_metadata = {
        "filename": base_filename,
        "filename_label": filename_label,
        "uuid": dataset_uuid,
        "label": base,
        "voxel_size": voxel_size,
        "grid_size": grid_size,
        "axis": axis,
        "num_slices": num_slices,
        "slices": [],
    }

    if slice_unit_spacing is not None:
        dataset_metadata["slice_unit_spacing"] = slice_unit_spacing

    slice_paths = []

    for i, (slice_data, z_coord) in enumerate(zip(slices, slice_coordinates)):
        # Set all particles to the same grayscale color
        slice_data = np.where(slice_data > 0, greyscale_particle_color, 0).astype(np.uint8)

        # Construct the filename
        slice_filename = f"{base}_slice_{i:03d}.tiff" if base_filename else f"slice_{i:03d}.tiff"
        slice_path = Path(output_dir_slices) / slice_filename
        slice_paths.append(slice_path)

        # Save the slice as a TIFF file
        # fig, ax = plt.subplots(figsize=(5, 5))
        # ax.imshow(slice_data, cmap=cmap, origin='lower', interpolation='nearest', vmin=0, vmax=len(cmap.colors) - 1)
        # ax.axis('off')  # No axes, labels, or ticks
        # plt.savefig(slice_path, format="tiff", dpi=300, bbox_inches='tight', pad_inches=0)
        # plt.close()

        # Resize slice if resolution_scale is less than 1.0
        if resolution_scale != 1.0:
            original_shape = slice_data.shape
            new_shape = (int(original_shape[1] * resolution_scale), int(original_shape[0] * resolution_scale))
            slice_data = cv2.resize(slice_data, new_shape, interpolation=cv2.INTER_AREA)

        # Save the slice as a grayscale TIFF file
        slice_data = cv2.flip(slice_data, 0)  
        cv2.imwrite(str(slice_path), slice_data.astype(np.uint8))
		
        # Add slice info to metadata
        grid_position = {"x": 0, "y": 0, "z": 0}
        
        if axis == "z":
            grid_position["z"] = int(z_coord / voxel_size)
        elif axis == "y":
            grid_position["y"] = int(z_coord / voxel_size)
        elif axis == "x":
            grid_position["x"] = int(z_coord / voxel_size)

        dataset_metadata["slices"].append(
            {
                "filename": str(slice_path),
                "real_position": z_coord,
                "grid_position": grid_position,
            }
        )

        # Update the centralized metadata file
        central_metadata = {"files": []}
        if Path(metadata_path).exists():
            with open(metadata_path, "r") as meta_file:
                central_metadata = json.load(meta_file)

        # Check if the file already exists in metadata
        for i, existing_entry in enumerate(central_metadata["files"]):
            if existing_entry["filename"] == base_filename:
                # Overwrite existing entry
                central_metadata["files"][i] = dataset_metadata
                break
        else:
            # Add new entry
            central_metadata["files"].append(dataset_metadata)

        # Save the updated centralized metadata
        with open(metadata_path, "w") as meta_file:
            json.dump(central_metadata, meta_file, indent=4)

    print(f"Saved slices and updated centralized metadata at {metadata_path}")
    
    if make_gif == True:
        # Call the GIF helper
        gif_output_path = Path(output_dir_slices) / f"{base}_slices.gif"
        create_gif_from_slices(slice_paths, gif_output_path)

def create_gif_from_slices(slice_paths, output_gif_path, duration=200):
    """
    Create a GIF from a list of image paths.

    Parameters:
        slice_paths (list): List of file paths to the slice images.
        output_gif_path (str): Path to save the output GIF.
        duration (int): Duration per frame in milliseconds.

    Returns:
        None
    """
    # Open images and ensure they're compatible
    frames = [Image.open(path) for path in slice_paths]
    
    # Save as GIF
    frames[0].save(
        output_gif_path,
        save_all=True,
        append_images=frames[1:],
        duration=duration,
        loop=0  # Loop forever
    )
    print(f"GIF saved to {output_gif_path}")

def save_voxelized_data_as_json(voxel_data, original_dat_filepath, output_dir="json_outputs"):
    """
    Save voxelized .dat particle data as a structured .json file.

    Parameters:
        voxel_data (dict): Voxelized particle data output from `voxelize_dat_particles`.
        dat_file_path (str): Path to the original .dat file.
        output_dir (str): Directory to save the generated JSON file.

    Returns:
        str: Path to the saved JSON file.
    """

    json_creation_start_time = time.time()

    # Extract required information
    voxel_size = voxel_data["voxel_size"]
    domain_size = voxel_data["domain_size"]
    voxel_count = voxel_data["voxel_count"]
    particles = voxel_data["particles"]

    # Convert particles dictionary to "bead_data" format
    bead_data = {}
    bead_voxel_count = {}

    for pid, voxel_indices in particles.items():
        voxel_indices = np.sort(voxel_indices)  # Use NumPy for fast sorting

        # Detect discontinuities where index difference is > 1
        breaks = np.where(np.diff(voxel_indices) > 1)[0]  # Indices where gaps occur
        starts = np.insert(voxel_indices[breaks + 1], 0, voxel_indices[0])  # Start of ranges
        ends = np.append(voxel_indices[breaks], voxel_indices[-1])  # End of ranges
        
        # Store results
        bead_data[pid] = np.column_stack((starts, ends)).tolist()  # Convert to list of lists
        bead_voxel_count[pid] = len(voxel_indices)

    # Create metadata dictionary
    json_data = [{
        "bead_count": len(particles),
        "bead_voxel_count": bead_voxel_count,
        "created": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "data_type": "labeled",
        "domain_size": list(domain_size),  # Convert tuple to list for JSON compatibility
        "hip_file": str(original_dat_filepath),  # Store original .dat file location
        "voxel_count": voxel_count,
        "voxel_size": voxel_size,
        "bead_data": bead_data
    }]

    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Save JSON file with the same name as the .dat file
    json_filename = os.path.splitext(os.path.basename(original_dat_filepath))[0] + ".json"
    json_path = os.path.join(output_dir, json_filename)

    with open(json_path, "w") as json_file:
        json.dump(json_data, json_file, separators=(",", ":"))

    # End timing for data preparation
    json_creation_end_time = time.time()
    json_creation_duration = json_creation_end_time - json_creation_start_time
    print(f"Saved voxelized JSON: {json_path}")
    print(f"Time taken for method-specific data preparation: {json_creation_duration:.2f} seconds")

    return json_path

def create_and_save_voxel_grid(particles, grid_size, voxel_size, output_path, axis='z'):
    """
    Creates a 3D voxel grid with labeled particles using a 1D mask and saves it in a compressed .npz format.

    Parameters:
        particles (dict): Dictionary mapping particle labels to 1D voxel indices.
        grid_size (tuple): Shape of the 3D voxel grid (nx, ny, nz).
        voxel_size (float): Size of each voxel in real-world units.
        output_path (str): Path to save the compressed .npz file.
    """
    # Ensure the output directory exists
    output_dir = os.path.dirname(output_path)  # Extract directory from full file path
    os.makedirs(output_dir, exist_ok=True)  # Create directory if it doesn't exist

    # Initialize 1D voxel grid (background = 0)
    flat_grid_size = np.prod(grid_size)  # Total number of voxels
    voxel_grid_1d = np.zeros(flat_grid_size, dtype=np.uint16)

    # Assign particle labels efficiently using a mask
    for particle_label, voxel_indices in particles.items():
        voxel_grid_1d[voxel_indices] = int(particle_label)  # Assign label directly

    # Reshape into 3D voxel grid
    voxel_grid_3d = np.reshape(voxel_grid_1d, grid_size, order='F')

    # Flip along the correct axes to match bottom-left origin convention
    if axis == 'y':
        voxel_grid_3d = np.rot90(voxel_grid_3d, k=1, axes=(0, 2))
        # voxel_grid_3d = np.flip(voxel_grid_3d, axis=0)
    elif axis == 'x':
        voxel_grid_3d = np.rot90(voxel_grid_3d, k=1, axes=(1, 2))
        # voxel_grid_3d = np.flip(voxel_grid_3d, axis=1)
    else: 
        voxel_grid_3d = np.rot90(voxel_grid_3d, k=1, axes=(0, 1))
        # voxel_grid_3d = np.flip(voxel_grid_3d, axis=0)

    # voxel_grid_3d = np.transpose(voxel_grid_3d, (1, 0, 2))  # Swap X and Y axes
    # voxel_grid_3d = np.flip(voxel_grid_3d, axis=1)  # Flip X-axis to match correct orientation

    # Save the voxel grid in compressed format
    np.savez_compressed(output_path, voxel_grid=voxel_grid_3d, voxel_size=voxel_size, grid_size=grid_size)

    print(f"Saved voxel grid: {output_path} (Shape: {voxel_grid_3d.shape})")

