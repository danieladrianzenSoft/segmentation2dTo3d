from core.file_parsing_methods import parse_file
from core.grid_optimization_methods import process_particles
from core.voxelization_helper_methods import get_centered_grid
from core.scraping_methods import get_files, select_input_file
from core.visualization_methods import create_colormap, plot_voxelized_domain

def process_file(selected_file, config):
    # Select and parse file
    result = parse_file(config, selected_file)

    # Exit early if parse_file returns None
    if result is None: return
    
    # Unpack the result from parse_file
    domain_data, surface_data, domain_metadata, voxel_size, domain_size, domain_type = result

    # Generate grid and process particles
    if (len(domain_size) == 3):
        bounds = (0, domain_size[0], 0, domain_size[1], 0, domain_size[2])
    else:
        bounds = tuple(domain_size)

    voxel_centers, grid_size = get_centered_grid(bounds, voxel_size)

    if domain_type == "pore":
        show_edge_pores = config.get("show_edge_pores", False)

        if not show_edge_pores and domain_metadata:
            to_remove = []

            for pore_id in domain_data:
                metadata = domain_metadata.get(pore_id, {})
                is_edge = metadata.get("edge", None)
                if is_edge == 1:
                    to_remove.append(pore_id)

            for pore_id in to_remove:
                del domain_data[pore_id]

            print(f"Filtered to {len(domain_data)} interior pores (excluding edge pores)")

            if not domain_data:
                print("⚠️ No interior pores found to plot.")
                return

    if config.get("downsample_factor") and config.get("max_particles"):
        domain_data = process_particles(
            domain_data,
            voxel_centers,
            grid_size,
            downsample_factor=config["downsample_factor"],
            max_particles=config["max_particles"],
        )
    
    if domain_type == "pore":
        label = "Pores (Interior Only)" if not config.get("show_edge_pores", False) else "Pores (All)"
    else:
        label = domain_type.capitalize()  # "Particle"

    # Create colormap
    cmap = create_colormap(domain_data)
    plot_voxelized_domain(
		domain_data=domain_data,
		voxel_centers=voxel_centers,
		grid_size=grid_size,
		voxel_size=voxel_size,
		domain_size=domain_size,
		config=config,
		cmap=cmap,
		label=label
	)
    
def run(config):   
    # Scrape folder and validate input
    dat_files, json_files, npz_files = get_files(config["folder_path"])

    selected_file = select_input_file(config, [dat_files, json_files, npz_files])
    process_file(selected_file=selected_file, config=config)


if __name__ == "__main__":
    from workflows.voxelize_into_json import get_config
    run(get_config())