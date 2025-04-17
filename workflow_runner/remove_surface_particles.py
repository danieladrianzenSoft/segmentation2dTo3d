import os
import json
from core.file_parsing_methods import load_raw_json_data, parse_raw_domain_data
from core.surface_particle_removal_methods import filter_incomplete_beads
from core.voxelization_helper_methods import get_centered_grid

def clean_surface_particles(input_dir, output_dir, filename, threshold, axis=2):
    file_path = os.path.join(input_dir, filename)
    
    data = load_raw_json_data(file_path)
    domain_data = parse_raw_domain_data(data)
    voxel_size = domain_data["voxel_size"]
    domain_size = domain_data["domain_size"]
    bead_data = domain_data["bead_data"]
    
    bounds = (0, domain_size[0], 0, domain_size[1], 0, domain_size[2])
    
    grid, _ = get_centered_grid(bounds, voxel_size)
    filtered_bead_data = filter_incomplete_beads(bead_data, grid, threshold, axis=axis)
    data["bead_data"] = filtered_bead_data
    data["bead_count"] = len(filtered_bead_data)
    # Filter voxel counts to only include remaining beads
    original_voxel_count = data.get("bead_voxel_count", {})
    data["bead_voxel_count"] = {
        k: v for k, v in original_voxel_count.items() if k in filtered_bead_data
    }

    # Determine output filename
    name, ext = os.path.splitext(filename)
    if input_dir == output_dir:
        output_filename = f"{name}_clean{ext}"
    else:
        output_filename = filename

    output_path = os.path.join(output_dir, output_filename)
    with open(output_path, 'w') as f:
        json.dump(data, f, separators=(",", ":"))

    print(f"Processed file saved to {output_path}")

def run(config):
    clean_surface_particles(
        input_dir=config["input_directory"],
        output_dir=config["output_directory"],
        filename=config["filename"],
        threshold=config["threshold"]
	)