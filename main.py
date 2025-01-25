from data_generation_methods import save_slices_as_tiff
from file_parsing import parse_file
from grid_optimization_methods import process_particles
from helper_methods import extract_slices, get_centered_grid
from scraping_methods import get_files, validate_input
from summary_methods import summarize_slices
from visualization_methods import create_colormap, plot_particles, plot_slices_as_images

def main():
    # Configuration
    config = {
        "folder_path": "./TestData",
        "file_type": "json",
        "file_index": 2,
        "downsample_factor": 1,
        "surface_only": True,
        "max_particles": None,
        "num_slices": 5,
        "axis": "z",
        "show_legend": False,
        "voxelization_dx": 2,
        "slices_debug_mode": True,
        "generate_images": True,
        "output_dir": "slices_TestData",
        "metadata_path": "metadata_TestData.json"
    }

    # Scrape folder and validate input
    dat_files, json_files = get_files(config["folder_path"])
    validate_input(config, dat_files, json_files)

    # Select and parse file
    selected_file, particles, voxel_size, domain_size = parse_file(config, dat_files, json_files)

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

    # Create colormap
    cmap = create_colormap(particles)

    # Extract slices
    slices, slice_coordinates = extract_slices(
        particles, voxel_centers, voxel_size, grid_size, num_slices=config["num_slices"], axis=config["axis"]
    )

    if config["slices_debug_mode"]:
        summarize_slices(slices, slice_coordinates, grid_size, voxel_size, particles, axis=config["axis"])

    # Plot or save slices
    if config["generate_images"]:
        save_slices_as_tiff(
			slices, 
			slice_coordinates, 
			voxel_size, 
			grid_size, 
			axis='z', 
			output_dir=config["output_dir"],
			label_file_name=selected_file,
			metadata_path=config["metadata_path"]
		)
    else:
        plot_slices_as_images(
            slices, slice_coordinates, voxel_size, grid_size, axis=config["axis"], cmap=cmap
        )
        plot_particles(
            particles,
            voxel_centers,
            grid_size,
            voxel_size,
            domain_size,
            surface_only=config["surface_only"],
            slices=slices,
            slice_coordinates=slice_coordinates,
            axis=config["axis"],
            plot_method="pv_polydata",
            cmap=cmap,
            show_legend=config["show_legend"],
        )

if __name__ == "__main__":
    main()