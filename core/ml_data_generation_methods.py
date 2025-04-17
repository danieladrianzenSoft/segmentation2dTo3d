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
from tqdm import tqdm
from plyfile import PlyData, PlyElement
from core.grid_optimization_methods import downsample_pointcloud

def voxel_grid_to_pointcloud(particles, grid_size, voxel_size, base_file_name, target_size=(512,512), axis='z', output_path="labels", output_format="npy"):
    """
    Converts a full 3D voxel grid into a point cloud (.ply) file using **grid-space coordinates**.

    Parameters:
        particles (dict): Dictionary mapping particle labels to 1D voxel indices.
        grid_size (tuple): Shape of the 3D grid (nx, ny, nz).
        base_file_name (str): Base name for output files.
        target_size (tuple): Final padded size (default: 512x512).
        axis (str): Axis along which slices were taken ('x', 'y', 'z').
        output_path (str): Directory to save `.ply` point cloud file.

    Returns:
        None
    """
    pc_start_time = time.time()

    print(f"🔹 Converting full 3D voxel data to point cloud (Grid Space)...")

    # Compute padding offsets
    pad_x, pad_y, pad_left, pad_top = apply_padding(grid_size, target_size, axis)

    # Prepare storage for all points
    all_points = []
    all_labels = []

    # Iterate over particles (batch processing all voxels per particle)
    for particle_label, voxel_indices in tqdm(particles.items(), desc="Processing Particles"):
        # Convert 1D indices into integer 3D grid coordinates
        x_indices = (voxel_indices % grid_size[0])  # X indices
        y_indices = (voxel_indices // grid_size[0]) % grid_size[1]  # Y indices
        z_indices = voxel_indices // (grid_size[0] * grid_size[1])  # Z indices

        # Adjust based on axis
        if axis == 'x':  
            points = np.column_stack((z_indices, y_indices, x_indices))  # (Z, Y, X) → Projected along X
        elif axis == 'y':  
            points = np.column_stack((x_indices, z_indices, y_indices))  # (X, Z, Y) → Projected along Y
        else:  
            points = np.column_stack((x_indices, y_indices, z_indices))  # (X, Y, Z) → Projected along Z (default)

        # Apply **consistent padding shift**
        if axis == 'z':
            points[:, 0] += pad_left  # Shift in X direction
            points[:, 1] += pad_top   # Shift in Y direction
        elif axis == 'y':
            points[:, 0] += pad_left  # Shift in X direction
            points[:, 2] += pad_top   # Shift in Z direction
        elif axis == 'x':
            points[:, 1] += pad_left  # Shift in Y direction
            points[:, 2] += pad_top   # Shift in Z direction

        # Store points and labels
        all_points.append(points)  # Shape: (num_voxels, 3)
        all_labels.append(np.full(len(points), particle_label, dtype=np.int32))  # Shape: (num_voxels,)

    # Concatenate all particles into a single array
    all_points = np.vstack(all_points)  # Shape: (total_voxels, 3)
    all_labels = np.concatenate(all_labels)  # Shape: (total_voxels,)

    initial_size = len(all_points)
    all_points, all_labels = downsample_pointcloud(all_points, all_labels, voxel_size=voxel_size)
    downsampled_size = len(all_points)

    print(f"✅ Extracted {initial_size} points and downsampled to {downsampled_size} for point cloud.")

    # Ensure output directory exists
    Path(output_path).mkdir(parents=True, exist_ok=True)

    if output_format == "ply":
        # Convert to structured numpy array for PLY export (store as integers)
        point_dtype = [('x', 'i4'), ('y', 'i4'), ('z', 'i4'), ('label', 'i4')]
        point_array = np.empty(len(all_points), dtype=point_dtype)
        point_array['x'], point_array['y'], point_array['z'], point_array['label'] = all_points[:, 0], all_points[:, 1], all_points[:, 2], all_labels

        # Define full file path
        output_file = os.path.join(output_path, f"{base_file_name}_label.ply")

        # Create a PLY file
        vertex_element = PlyElement.describe(point_array, "vertex")
        # PlyData([vertex_element], text=True).write(output_ply)
        PlyData([vertex_element], text=False).write(output_file)  # Binary Format (Smaller File)
        print(f"✅ Saved PLY file: {output_file}")
        
    elif output_format == "npy":
        # Define full file path
        output_file = os.path.join(output_path, f"{base_file_name}_label.npy")

        # Save as .npy file
        np.save(output_file, np.hstack((all_points, all_labels.reshape(-1, 1))))  # Shape: (N, 4)

        print(f"✅ Saved NPY file: {output_file}")
    elif output_format == "npz":
        # Define full file path
        output_file = os.path.join(output_path, f"{base_file_name}_label.npz")

        # Save as .npy file
        np.savez_compressed(output_file, np.hstack((all_points, all_labels.reshape(-1, 1))))  # Shape: (N, 4)
    
    else: 
        raise ValueError("❌ Invalid file format. Choose 'ply', 'npy' or 'npz.")

    # End pc timer
    pc_end_time = time.time()
    pc_duration = pc_end_time - pc_start_time
    print(f"Time taken for {output_format} file creation: {pc_duration:.2f} seconds")

def apply_padding(grid_size, target_size, axis='z'):
    """
    Computes padding offsets for voxel coordinates and images, ensuring slices and point cloud align.

    Parameters:
        grid_size (tuple): Shape of the original 3D grid (nx, ny, nz).
        target_size (tuple): Final padded size (512, 512).
        axis (str): Axis along which slices were taken ('x', 'y', or 'z').

    Returns:
        tuple: (pad_x, pad_y, pad_left_x, pad_top_y)
    """

    if axis == 'z':
        pad_x = max(0, target_size[0] - grid_size[0])  # Pad width (X)
        pad_y = max(0, target_size[1] - grid_size[1])  # Pad height (Y)
    elif axis == 'y':
        pad_x = max(0, target_size[0] - grid_size[0])  # Pad width (X)
        pad_y = max(0, target_size[1] - grid_size[2])  # Pad depth (Z)
    elif axis == 'x':
        pad_x = max(0, target_size[0] - grid_size[1])  # Pad height (Y)
        pad_y = max(0, target_size[1] - grid_size[2])  # Pad depth (Z)
    else:
        raise ValueError("Invalid axis. Choose from 'x', 'y', or 'z'.")

    pad_left_x = pad_x // 2
    pad_top_y = pad_y // 2

    return pad_x, pad_y, pad_left_x, pad_top_y

def rotate_pointcloud(points, axis='z', angle_deg=90):
    """
    Rotates a 3D point cloud around its bounding box center in **grid space**.

    Parameters:
        points (np.ndarray): (N, 3) array of 3D points in grid space.
        axis (str): Axis to rotate around ('x', 'y', or 'z').
        angle_deg (float): Rotation angle in degrees.

    Returns:
        np.ndarray: Rotated (N, 3) point cloud.
    """
    # Compute centroid **based on grid space**
    min_bound = np.min(points, axis=0)
    max_bound = np.max(points, axis=0)
    centroid = (min_bound + max_bound) / 2  # Midpoint of bounding box

    # Translate to center for rotation
    translated_points = points - centroid

    # Define rotation matrices
    angle_rad = np.radians(angle_deg)
    cos_a, sin_a = np.cos(angle_rad), np.sin(angle_rad)

    if axis == 'y':  # Rotate around Y → Swap X ↔ Z
        rotation_matrix = np.array([[cos_a, 0, sin_a],
                                    [0, 1, 0],
                                    [-sin_a, 0, cos_a]])
    elif axis == 'x':  # Rotate around X → Swap Y ↔ Z
        rotation_matrix = np.array([[1, 0, 0],
                                    [0, cos_a, -sin_a],
                                    [0, sin_a, cos_a]])
    else:  # Rotate around Z → Swap X ↔ Y
        rotation_matrix = np.array([[cos_a, -sin_a, 0],
                                    [sin_a, cos_a, 0],
                                    [0, 0, 1]])

    # Apply rotation
    rotated_points = np.dot(translated_points, rotation_matrix.T)

    # Translate back to original position
    rotated_points += centroid

    return rotated_points

def save_slices_mask3D(
    slices,
    slice_coordinates,  # Now in GRID SPACE
    voxel_size,
    grid_size,
    num_slices,
    base_file_name,
    slice_unit_spacing=None,
    axis='z',
    output_dir_slices="slices",
    output_dir_labels="labels",
    metadata=None,
    metadata_path="metadata_mask3D.json",
    particle_color=(0, 255, 0),
    target_size=(512, 512)
):
    """
    Save slices in **grid space** and generate Mask3D-compatible metadata.

    Parameters:
        slices (list): List of 2D numpy arrays in grid space.
        slice_coordinates (list): List of **grid-space** slice midpoints.
        voxel_size (float): Size of each voxel.
        grid_size (tuple): Shape of the 3D grid (nx, ny, nz).
        num_slices (int): Number of slices extracted.
        base_file_name (str): Base name for output files.
        axis (str): Slicing axis ('x', 'y', 'z').
        metadata (dict): Metadata object to be saved.
        metadata_path (str): Path for metadata JSON.
        target_size (tuple): Padded size (default: 512x512).

    Returns:
        None
    """

    slice_start_time = time.time()

    Path(output_dir_slices).mkdir(parents=True, exist_ok=True)

    # 🔹 Compute padding offsets in GRID SPACE (no voxel_size scaling)
    pad_x, pad_y, pad_left, pad_top = apply_padding(grid_size, target_size, axis)

    label_filename = f"{base_file_name}_label.ply"
    label_path = str(Path(output_dir_labels) / label_filename)

    dataset_metadata = {
        "uuid": str(uuid.uuid4()),
        "filename": base_file_name,
        "label_filename": label_filename,
        "grid_size": grid_size,
        "voxel_size": voxel_size,
        "axis": axis,
        "num_slices": num_slices,
        "slices": [],
    }

    if slice_unit_spacing is not None:
        dataset_metadata["slice_spacing"] = slice_unit_spacing

    # 🔹 Save slices and store metadata
    for i, grid_position in enumerate(slice_coordinates):
        real_position = grid_position * voxel_size  # Convert to real-world coordinates

        # 🔹 Convert slice labels to RGB format
        rgb_slice = np.zeros((*slices[i].shape, 3), dtype=np.uint8)  # Initialize as black

        # Assign particle color
        rgb_slice[slices[i] > 0] = particle_color

        # 🔹 Apply padding **consistently with point cloud**
        padded_slice = cv2.copyMakeBorder(
            rgb_slice,
            pad_top, pad_y - pad_top,
            pad_left, pad_x - pad_left,
            borderType=cv2.BORDER_CONSTANT,
            value=(0, 0, 0)  # Black padding
        )

        # 🔹 Flip Y-axis before saving to match 3D coordinate system
        # if axis=='x':
        #     padded_slice = cv2.flip(padded_slice, 0)

        # 🔹 Construct filenames
        slice_filename = f"{base_file_name}_slice_{i:03d}.tiff"
        slice_path = str(Path(output_dir_slices) / slice_filename)

        # 🔹 Save as RGB TIFF
        cv2.imwrite(str(slice_path), cv2.cvtColor(padded_slice, cv2.COLOR_RGB2BGR))

        # 🔹 Store metadata with both grid & real positions
        dataset_metadata["slices"].append({
            "filename": slice_filename,
            "grid_position": int(grid_position),  # Stored in grid space
            "real_position": float(real_position),       # Stored in real space
        })
    
    metadata["scaffolds"].append(dataset_metadata)

    with open(metadata_path, "w") as meta_file:
        json.dump(metadata, meta_file, indent=4)

    # End ply timer
    slice_end_time = time.time()
    slice_duration = slice_end_time - slice_start_time
    print(f"Time taken for slice image creation: {slice_duration:.2f} seconds")  
    print(f"✅ Saved slices and metadata at {metadata_path}")

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
        metadata=None,
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
        dataset_metadata["slice_spacing"] = slice_unit_spacing

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
        # central_metadata = {"files": []}
        # if Path(metadata_path).exists():
        #     with open(metadata_path, "r") as meta_file:
        #         central_metadata = json.load(meta_file)

        # # Check if the file already exists in metadata
        # for i, existing_entry in enumerate(central_metadata["files"]):
        #     if existing_entry["filename"] == base_filename:
        #         # Overwrite existing entry
        #         central_metadata["files"][i] = dataset_metadata
        #         break
        # else:
        #     # Add new entry
        #     central_metadata["files"].append(dataset_metadata)

        # Save the updated centralized metadata

    metadata["scaffolds"].append(dataset_metadata)

    with open(metadata_path, "w") as meta_file:
        json.dump(metadata, meta_file, indent=4)

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

def create_labels_mask3D(particles, grid_size, voxel_size, output_dir_labels, metadata_path, axis='z', target_size=(512, 512)):
    """
    Creates a 3D voxel grid with labeled particles and saves 2D slices as .npz files for Mask3D.

    Parameters:
        particles (dict): Dictionary mapping particle labels to 1D voxel indices.
        grid_size (tuple): Shape of the 3D voxel grid (nx, ny, nz).
        voxel_size (float): Size of each voxel in real-world units.
        output_dir_labels (str): Directory to save the label slices.
        metadata_path (str): Path to the Mask3D metadata file.
        axis (str): Axis along which to extract slices ('x', 'y', 'z').
        target_size (tuple): Final (width, height) for padding.

    Returns:
        None
    """

    # Ensure output directory exists
    Path(output_dir_labels).mkdir(parents=True, exist_ok=True)

    # Initialize 1D voxel grid (background = 0)
    flat_grid_size = np.prod(grid_size)  
    voxel_grid_1d = np.zeros(flat_grid_size, dtype=np.uint16)

    # Assign particle labels efficiently
    for particle_label, voxel_indices in particles.items():
        voxel_grid_1d[voxel_indices] = int(particle_label)

    # Reshape into 3D voxel grid
    voxel_grid_3d = np.reshape(voxel_grid_1d, grid_size, order='F')

    # **Rotate voxel grid** based on slicing axis
    if axis == 'y':
        voxel_grid_3d = np.rot90(voxel_grid_3d, k=1, axes=(0, 2))
    elif axis == 'x':
        voxel_grid_3d = np.rot90(voxel_grid_3d, k=1, axes=(1, 2))
    else:
        voxel_grid_3d = np.rot90(voxel_grid_3d, k=1, axes=(0, 1))

    print(f"🟢 Voxel grid rotated to match {axis}-axis slicing.")

    # Load existing metadata
    with open(metadata_path, "r") as f:
        metadata = json.load(f)

    # Find slices from metadata
    slices_metadata = metadata["slices"]

    # **Process each slice and save labels**
    for slice_info in tqdm(slices_metadata, desc="Saving label slices"):
        slice_filename = Path(slice_info["filename"]).name  # Extract base filename
        label_filename = slice_filename.replace(".tiff", ".npz")  # Convert to npz
        label_path = Path(output_dir_labels) / label_filename

        # Get slice index from metadata
        grid_position = slice_info["grid_position"]
        slice_idx = grid_position[axis]

        # Extract slice from voxel grid
        if axis == 'z':
            slice_data = voxel_grid_3d[:, :, slice_idx]
        elif axis == 'y':
            slice_data = voxel_grid_3d[:, slice_idx, :]
        elif axis == 'x':
            slice_data = voxel_grid_3d[slice_idx, :, :]

        # **Pad the label slice to match `512x512`**
        original_height, original_width = slice_data.shape
        pad_x = max(0, target_size[0] - original_width)
        pad_y = max(0, target_size[1] - original_height)

        pad_left = pad_x // 2
        pad_right = pad_x - pad_left
        pad_top = pad_y // 2
        pad_bottom = pad_y - pad_top

        padded_slice = np.pad(slice_data, ((pad_top, pad_bottom), (pad_left, pad_right)), mode='constant', constant_values=0)

        # Save label as `.npz`
        np.savez_compressed(label_path, voxel_grid=padded_slice, voxel_size=voxel_size)

        # Update metadata with label path
        slice_info["label_filename"] = str(label_path)

    # Save updated metadata
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=4)

    print(f"✅ Labels saved in {output_dir_labels} and metadata updated.")


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

