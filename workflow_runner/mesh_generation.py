import os
from core.file_parsing_methods import parse_file
from core.mesh_generation_methods import generate_mesh_marching_cubes
from core.scraping_methods import get_files, validate_input
from core.voxelization_helper_methods import get_centered_grid

def process_file(selected_file, config):
    # Select and parse file
    result = parse_file(config, selected_file)

    # Exit early if parse_file returns None
    if result is None: return
    
    # Unpack the result from parse_file
    particles, voxel_size, domain_size = result

    # Generate grid and process particles
    bounds = (0, domain_size[0], 0, domain_size[1], 0, domain_size[2])
    voxel_centers, grid_size = get_centered_grid(bounds, voxel_size)

    # Extract only surface voxels for efficiency
    # surface_particles = extract_surface_voxels(particles, grid_size)
    surface_particles = particles
    
	# Generate .glb file
    output_dir = config.get("output_dir", "./meshes")
    output_path = os.path.join(output_dir, f"{os.path.basename(selected_file).split('.')[0]}.glb")
    # generate_glb_file_convex_hull(surface_particles, voxel_centers, voxel_size, output_path)
    # generate_glb_file_poisson(surface_particles, voxel_centers, voxel_size, output_path, method="bpa")
    # generate_glb_file_delauney(
    #     surface_particles, 
    #     voxel_centers, 
    #     voxel_size, 
    #     output_path, 
    # )
    # generate_glb_file_from_surface(surface_particles, voxel_centers, voxel_size, output_path)
    generate_mesh_marching_cubes(surface_particles, voxel_centers, voxel_size, output_path)

def run(config):
    dat_files, json_files, npz_files = get_files(config["input_dir"])

    if config["batch_process"]:
        files_to_process = json_files
    else:
        validate_input(config, dat_files, json_files, npz_files)
        files_to_process = [json_files[config["file_index"] - 1]]

    for selected_file in files_to_process:
        print(f"Processing file: {selected_file}")
        process_file(selected_file, config)