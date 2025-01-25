from data_generation_methods import save_slices_as_tiff
from file_parsing import parse_file
from grid_optimization_methods import process_particles
from helper_methods import determine_axis, determine_num_slices, extract_slices, get_centered_grid, select_representative_slices
from scraping_methods import get_files, validate_input
from summary_methods import summarize_slices
from visualization_methods import create_colormap, plot_particles, plot_slices_as_images

def process_file(selected_file, config):
    # Select and parse file
    particles, voxel_size, domain_size = parse_file(config, selected_file)

    # Generate grid and process particles
    bounds = (0, domain_size[0], 0, domain_size[1], 0, domain_size[2])
    voxel_centers, grid_size = get_centered_grid(bounds, voxel_size)

    particles = process_particles(
        particles,
        voxel_centers,
        grid_size,
        downsample_factor=config["downsample_factor"],
        max_particles=config["max_particles"],
    )

    # Determine num_slices (if random_slice_spacing is True)
    num_slices, slice_unit_spacing = determine_num_slices(grid_size, voxel_size, config)

    # Choose the slicing axis
    axis = determine_axis(config)

    # Create colormap
    cmap = create_colormap(particles)

    # Extract slices
    slices, slice_coordinates = extract_slices(
        particles, voxel_centers, voxel_size, grid_size, num_slices=num_slices, axis=axis
    )

    if config["slices_debug_mode"]:
        summarize_slices(slices, slice_coordinates, grid_size, voxel_size, particles, axis=axis)

    # Plot or save slices
    if config["generate_images"]:
        save_slices_as_tiff(
			slices, 
			slice_coordinates, 
			voxel_size, 
			grid_size, 
            num_slices,
            slice_unit_spacing=slice_unit_spacing,
			axis=axis, 
			output_dir=config["output_dir"],
			label_file_name=selected_file,
			metadata_path=config["metadata_path"]
		)
    else:
        # Select representative slices for plotting
        selected_slices, selected_slice_coordinates = select_representative_slices(slices, slice_coordinates, num_to_select=5)
        plot_slices_as_images(
            selected_slices, selected_slice_coordinates, voxel_size, grid_size, axis=axis, cmap=cmap
        )
        plot_particles(
            particles,
            voxel_centers,
            grid_size,
            voxel_size,
            domain_size,
            surface_only=config["surface_only"],
            slices=selected_slices,
            slice_coordinates=selected_slice_coordinates,
            axis=axis,
            plot_method="pv_polydata",
            cmap=cmap,
            show_legend=config["show_legend"],
        )

def main():
    # Configuration
    config = {
        "folder_path": "./TestData",
        "batch_process": True,
        "file_type": "json",
        "file_index": 2,
        "downsample_factor": 1,
        "surface_only": True,
        "max_particles": None,
        "random_slice_spacing": True,
        "random_axis": True,
        "max_slices": 150,
        "num_slices": 5,
        "axis": "z",
        "show_legend": False,
        "voxelization_dx": 2,
        "slices_debug_mode": False,
        "generate_images": True,
        "output_dir": "slices_TestData",
        "metadata_path": "metadata_TestData.json"
    }
    
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

    # Process each file
    for selected_file in files_to_process:
        print(f"Processing file: {selected_file}")
        process_file(selected_file, config)

if __name__ == "__main__":
    main()