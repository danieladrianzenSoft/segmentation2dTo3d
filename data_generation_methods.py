import json
import os
from pathlib import Path
import uuid
from matplotlib import pyplot as plt
from matplotlib.colors import ListedColormap

def save_slices_as_tiff(
        slices, 
        slice_coordinates, 
        voxel_size, 
        grid_size, 
        num_slices,
        slice_unit_spacing=None,
        axis='z', 
        output_dir="slices", 
        label_file_name=None,
        metadata_path="metadata.json",
    ):
    """
    Save slices as TIFF images and generate a metadata file.

    Parameters:
        slices (list): List of 2D numpy arrays representing slices.
        slice_coordinates (list): List of slice midpoints (real-world coordinates).
        voxel_size (float): Size of each voxel in real-world units.
        grid_size (tuple): Shape of the 3D grid (nx, ny, nz).
        axis (str): Axis along which the slices were taken ('x', 'y', 'z').
        output_dir (str): Directory to save the TIFF files.
        label_file_name (str): Identifier for the original file (e.g., .dat or .json).
        num_slices (int): Number of slices extracted.
        slice_unit_spacing (int): Spacing between slices, if applicable.
        metadata_path (str): Path to the centralized metadata JSON file.
    Returns:
        None
    """
        
    # Ensure output directory exists
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    cmap = ListedColormap(['black', '#00FF00'])  # Default fluorescent green
    
	# Remove extension from label_file_name if provided
    if label_file_name:
        base_filename = Path(label_file_name).name  # Extract filename with extension
        label = Path(label_file_name).stem  # Extract filename without extension
    else:
        raise ValueError("label_file_name must be provided.")

    # Create a unique identifier for this dataset
    dataset_uuid = str(uuid.uuid4())

    # Initialize dataset-specific metadata
    dataset_metadata = {
        "filename": base_filename,
        "uuid": dataset_uuid,
        "label": label,
        "voxel_size": voxel_size,
        "grid_size": grid_size,
        "axis": axis,
        "num_slices": num_slices,
        "slices": [],
    }

    if slice_unit_spacing is not None:
        dataset_metadata["slice_unit_spacing"] = slice_unit_spacing

    for i, (slice_data, z_coord) in enumerate(zip(slices, slice_coordinates)):
        # Construct the filename
        slice_filename = f"{label}_slice_{i:03d}.tiff" if label_file_name else f"slice_{i:03d}.tiff"
        slice_path = Path(output_dir) / slice_filename

        # Save the slice as a TIFF file
        fig, ax = plt.subplots(figsize=(5, 5))
        ax.imshow(slice_data, cmap=cmap, origin='lower', interpolation='nearest', vmin=0, vmax=len(cmap.colors) - 1)
        ax.axis('off')  # No axes, labels, or ticks
        plt.savefig(slice_path, format="tiff", dpi=300, bbox_inches='tight', pad_inches=0)
        plt.close()
		
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