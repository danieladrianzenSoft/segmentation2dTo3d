import os
from core.file_parsing_methods import parse_file
from core.mesh_generation_methods import generate_mesh_marching_cubes
from core.scraping_methods import get_files, select_input_file
from core.voxelization_helper_methods import get_centered_grid

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
    
	# Generate .glb file
    output_dir = config.get("output_dir")
    output_path = os.path.join(output_dir, f"{os.path.basename(selected_file).split('.')[0]}.glb")
   
    generate_mesh_marching_cubes(domain_data, filter_metadata(domain_metadata), voxel_centers, voxel_size, output_path, config=config)

def filter_metadata(metadata_dict):
        if metadata_dict is None:
            return {}
        filtered = {}
        for key, original_metadata in metadata_dict.items():
            filtered[key] = {
                "volume": original_metadata.get("volume"),
                "surfArea": original_metadata.get("surfArea"),
                "charLength": original_metadata.get("charLength"),
                "edge": original_metadata.get("edge"),
                "avgDoorDiam": original_metadata.get("avgDoorDiam"),
                "largestDoorDiam": original_metadata.get("largestDoorDiam"),
                "beadNeighbors": original_metadata.get("beadNeighbors"),
            }
        return filtered

def run(config):
    dat_files, json_files, npz_files = get_files(config["input_dir"])

    if config["batch_process"]:
        files_to_process = json_files
    else:
        selected_file = select_input_file(config, [dat_files, json_files, npz_files])
        files_to_process = [selected_file]

    for selected_file in files_to_process:
        print(f"Processing file: {selected_file}")
        process_file(selected_file, config)

if __name__ == "__main__":
    from workflows.mesh_generation import get_config
    run(get_config())