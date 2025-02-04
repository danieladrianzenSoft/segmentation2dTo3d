import json
import os
from data_generation_methods import save_slices_as_tiff, create_and_save_voxel_grid
from file_parsing import parse_file
from grid_optimization_methods import process_particles
from helper_methods import determine_axis, determine_num_slices, extract_slices, get_centered_grid, select_representative_slices
from scraping_methods import get_files, validate_input
from summary_methods import summarize_slices
from visualization_methods import create_colormap, plot_particles, plot_slices_as_images, visualize_voxel_grid
from workflow_methods import get_workflow_config

def process_file(selected_file, config):
    # Select and parse file
    result = parse_file(config, selected_file)

    # Exit early if parse_file returns None
    if result is None: return

    # If we only want to convert .dat to .json, process and exit
    if config.get("dat_to_json", False): return
    
    # Unpack the result from parse_file
    particles, voxel_size, domain_size = result

    # Generate grid and process particles
    bounds = (0, domain_size[0], 0, domain_size[1], 0, domain_size[2])
    voxel_centers, grid_size = get_centered_grid(bounds, voxel_size)

    # Choose the slicing axis
    axis = determine_axis(config)

    # Determine output directories
    output_dir_labels = config.get("output_dir_labels", config.get("output_dir"))  # Use output_dir_labels if available, else fallback
    output_dir_slices = config.get("output_dir_slices", config.get("output_dir"))  # Use output_dir_slices if available, else fallback

    if config.get("generate_labels", False): 
        # Define output file path
        output_filename = os.path.join(output_dir_labels, os.path.basename(selected_file).replace(".json", ".npz"))

        # Call the new function to create & save the voxel grid
        create_and_save_voxel_grid(particles, grid_size, voxel_size, output_filename, axis=axis)
    
    if config.get("downsample_factor") and config.get("max_particles"):
        particles = process_particles(
            particles,
            voxel_centers,
            grid_size,
            downsample_factor=config["downsample_factor"],
            max_particles=config["max_particles"],
        )

    # Determine num_slices (if random_slice_spacing is True)
    num_slices, slice_unit_spacing = determine_num_slices(grid_size, voxel_size, config)

    # Create colormap
    cmap = create_colormap(particles)

    # Extract slices
    # slices, slice_coordinates = extract_slices(
    #     particles, voxel_centers, voxel_size, grid_size, num_slices=num_slices, axis=axis
    # )
    slices, slice_coordinates = extract_slices(particles, grid_size, voxel_size, num_slices=num_slices, axis=axis)


    if config.get("slices_debug_mode", False):
        summarize_slices(slices, slice_coordinates, grid_size, voxel_size, particles, axis=axis)

    # Plot or save slices
    if config.get("generate_images", False):
        save_slices_as_tiff(
			slices, 
			slice_coordinates, 
			voxel_size, 
			grid_size, 
            num_slices,
            slice_unit_spacing=slice_unit_spacing,
			axis=axis, 
			output_dir_slices=output_dir_slices,
            output_dir_labels=output_dir_labels,
			input_file_name=selected_file,
			metadata_path=config["metadata_path"]
		)
    else:
        # Select representative slices for plotting
        selected_slices, selected_slice_coordinates = select_representative_slices(slices, slice_coordinates, num_to_select=5)

        if (config.get("generate_labels"), False):
            visualize_voxel_grid(output_filename, axis=axis, slice_coordinates=selected_slice_coordinates, cmap=cmap)
        plot_slices_as_images(
            selected_slices, 
            selected_slice_coordinates, 
            voxel_size, 
            grid_size, 
            axis=axis, 
            cmap=cmap, 
            debug_mode=config.get("slices_debug_mode", False)
        )
        plot_particles(
            particles,
            voxel_centers,
            grid_size,
            voxel_size,
            domain_size,
            surface_only=config.get("surface_only", False),
            slices=selected_slices,
            slice_coordinates=selected_slice_coordinates,
            axis=axis,
            plot_method="pv_polydata",
            cmap=cmap,
            show_legend=config.get("show_legend", False),
        )

def main():
    # Workflow Configuration
    config = get_workflow_config(workflow_method='batch_generate_labels_slices')
    
    # Scrape folder and validate input
    dat_files, json_files = get_files(config["folder_path"])

    if config["batch_process"]:
        # Combine all .dat and .json files for batch processing
        files_to_process = dat_files + json_files
    else:
        # Validate input for single-file processing
        validate_input(config, dat_files, json_files)
        files_to_process = [
            dat_files[config["file_index"] - 1]
            if config["file_type"] == "dat"
            else json_files[config["file_index"] - 1]
        ]
    
    # Load metadata if restart is enabled
    processed_files = set()
    if config.get("batch_process", False) and config.get("restart", False) and os.path.exists(config["metadata_path"]):
        with open(config["metadata_path"], "r") as metadata_file:
            metadata = json.load(metadata_file)
            processed_files = {os.path.basename(entry["filename"]) for entry in metadata.get("files", [])}

    # Process each file
    for selected_file in files_to_process:
        file_basename = os.path.basename(selected_file)

        # Skip file if it has already been processed
        if config.get("restart", False) and file_basename in processed_files:
            print(f"Skipping already processed file: {file_basename}")
            continue

        print(f"Processing file: {selected_file}")
        process_file(selected_file, config)

if __name__ == "__main__":
    main()