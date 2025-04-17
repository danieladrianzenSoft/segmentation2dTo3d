import json
import os
import numpy as np
from core.file_parsing_methods import parse_file
from core.segmentation_methods import load_tiff_slices, segment_particles, resample_3d_labels
from core.visualization_methods import plot_particles
from core.voxelization_helper_methods import coords_to_index, get_centered_grid

def load_metadata(metadata_path):
    """Loads the metadata JSON file."""
    with open(metadata_path, 'r') as file:
        return json.load(file)

def group_tiff_files_by_scaffold(metadata, slices_dir, labels_dir):
    """Groups TIFF slice files by scaffold based on metadata, adjusting paths."""
    scaffold_groups = {}
    for entry in metadata.get("files", []):
        label = entry["label"]
        label_file = os.path.join(labels_dir, os.path.basename(entry["filename"]).replace(".dat", ".npz").replace(".json",".npz"))
        if label not in scaffold_groups:
            scaffold_groups[label] = {
                "uuid": entry["uuid"],
                "voxel_size": entry["voxel_size"],
                "grid_size": entry["grid_size"],
                "axis": entry["axis"],
                "slices": [],
                "label_file": label_file,
                "slice_unit_spacing": entry["slice_unit_spacing"]
            }
        for slice_entry in entry["slices"]:
            slice_filename = os.path.basename(slice_entry["filename"])
            full_path = os.path.join(slices_dir, slice_filename)
            scaffold_groups[label]["slices"].append(full_path)
    return scaffold_groups

def load_and_plot_ground_truth(scaffold_label, scaffold_data, voxel_centers, config):
    """Loads and plots the ground truth 3D geometry if available."""
    truth_label_file = scaffold_data["label_file"]
    if os.path.exists(truth_label_file):
        result = parse_file(config, truth_label_file)

        if result is None: return

        # Unpack the result from parse_file
        particles, voxel_size, domain_size = result
        grid_size = tuple(int(dim / voxel_size) for dim in domain_size)

        print(f"Loaded ground truth labels from {truth_label_file}")
        plot_particles(
            particles=particles,
            voxel_centers=voxel_centers,
            grid_size=grid_size,
            voxel_size=scaffold_data["voxel_size"],
            domain_size=domain_size,
            surface_only=config.get("surface_only", False),
            slices=None,
            slice_coordinates=None,
            axis=scaffold_data["axis"],
            plot_method="pv_polydata",
            cmap=None,
            show_legend=config.get("show_legend", False),
        )
    else:
        print(f"Warning: No ground truth label {truth_label_file}")

def process_scaffold(scaffold_label, scaffold_data, config, output_dir):
    """Processes a single scaffold, loading TIFF slices and performing segmentation."""
    print(f"Processing scaffold: {scaffold_label}")
    
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Load TIFF slices
    volume = load_tiff_slices(scaffold_data["slices"])
    
    # Perform segmentation
    binary_volume = volume > 0  # Assume nonzero voxels are foreground
    labeled_volume = segment_particles(binary_volume)
    
    # Use voxel size from metadata
    dx_truth = scaffold_data["voxel_size"]
    voxel_sizes_truth = (dx_truth, dx_truth, dx_truth)
    voxel_sizes_segmented = (dx_truth, dx_truth, scaffold_data["slice_unit_spacing"])

    # Scale the volume in Z by duplicating/removing slices
    labeled_volume = resample_3d_labels(labeled_volume, voxel_sizes_segmented, voxel_sizes_truth)  # Resample with correct spacing
    
    # Save labeled volume
    output_label_file = os.path.join(output_dir, f"{scaffold_label}_segmented.npz")
    np.savez_compressed(output_label_file, labeled_volume)
    print(f"Segmentation complete. {labeled_volume.max()} particles identified. Saved to {output_label_file}")

    # Compute voxel centers for segmented volume
    nx, ny, nz = scaffold_data["grid_size"]
    bounds = (0, nx * dx_truth,
              0, ny * dx_truth,
              0, nz * dx_truth)
    voxel_centers, _ = get_centered_grid(bounds, dx_truth)

    # Load and plot ground truth if available
    load_and_plot_ground_truth(scaffold_label, scaffold_data, voxel_centers, config)

    # particles_1d = {label: np.array([coords_to_index(x, y, z, nx, ny) for x, y, z in np.argwhere(labeled_volume == label)]) for label in np.unique(labeled_volume) if label != 0}
    # particles_1d = {int(label): np.array([coords_to_index(x, y, z, nx, ny) for x, y, z in np.argwhere(labeled_volume == label)]) for label in np.unique(labeled_volume) if label != 0}
    particles_1d = {str(label): np.array([coords_to_index(x, y, z, nx, ny) for x, y, z in np.argwhere(labeled_volume == label)]) 
                for label in np.unique(labeled_volume) if label != 0}
       
    # Plot segmented result
    plot_particles(
        particles=particles_1d,
        voxel_centers=voxel_centers,
        grid_size=labeled_volume.shape,
        voxel_size=dx_truth,
        domain_size=np.array(labeled_volume.shape) * dx_truth,
        surface_only=config.get("surface_only", False),
        slices=None,
        slice_coordinates=None,
        axis=scaffold_data["axis"],
        plot_method="pv_polydata",
        cmap=None,
        show_legend=config.get("show_legend", False),
    )
    
    return labeled_volume

def run(config):
    # Load metadata
    metadata = load_metadata(config["metadata_path"])
    scaffold_groups = group_tiff_files_by_scaffold(metadata, config["slices_dir"], config["labels_dir"])
    
    # Process the selected scaffold
    scaffold_label = config["process_scaffold"]
    if scaffold_label in scaffold_groups:
        process_scaffold(scaffold_label, scaffold_groups[scaffold_label], config, config["output_dir"])
    else:
        print(f"Scaffold '{scaffold_label}' not found in metadata.")