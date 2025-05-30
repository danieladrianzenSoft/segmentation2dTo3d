
import os
import json
from pathlib import Path
from core.file_parsing_methods import parse_file, to_serializable
from core.scraping_methods import get_files, select_input_file

def process_file(config, file):
    result = parse_file(config, file)
    if result is None: return
    pores, pores_metadata, voxel_size, domain_size = result

    # for i, (pore_id, voxel_ranges) in enumerate(pores.items()):
    #     print(f"Pore ID: {pore_id}")
    #     print("Voxel ranges:", voxel_ranges[:5])  # First 5 ranges
    #     print()
    #     if i >= 1:  # stop after 2 items (i = 0 and 1)
    #         break
    data = {
        "voxel_size": voxel_size,
        "domain_size": domain_size,
        "pores": pores,
        "pores_metadata": pores_metadata
	}
    base_file_name = Path(file).stem
    save_pore_data(data, config, base_file_name)
    # print(list(pores[0].keys()))

    return

def save_pore_data(data, config, filename="pore_domain"):
    """
    Save selected keys from data into a single JSON file.
    """
    output_dir = config["output_dir"]
    os.makedirs(output_dir, exist_ok=True)

    # Determine what to save
    save_keys = config.get("save_keys", ["pores", "pores_metadata"])
    data_to_save = {
        "voxel_size": data["voxel_size"],
        "domain_size": data["domain_size"]
    }

    for key in save_keys:
        if key in data:
            data_to_save[key] = data[key]
        else:
            print(f"Warning: key '{key}' not found in data")

    # Write to file
    filename = filename + ".json"
    output_path = os.path.join(output_dir, filename)
    with open(output_path, "w") as f:
        json.dump(to_serializable(data_to_save), f, allow_nan=False, separators=(",", ":"))

    print(f"✅ Saved JSON to: {output_path}")

def run(config):
    # Scrape folder and validate input
    mat_files, = get_files(config["input_dir"], extensions=[".mat"])

    if config.get("batch_process", False):
        # Combine all .dat and .json files for batch processing
        files_to_process = mat_files
    else:
        # Validate input for single-file processing
        selected_file = select_input_file(config, mat_files)
        files_to_process = [selected_file]
    
    # Process each file
    for selected_file in files_to_process:
        print(f"Processing file: {selected_file}")
        process_file(config=config, file=selected_file)

