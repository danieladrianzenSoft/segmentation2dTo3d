import os
import sys

def list_files_by_extension(folder_path, extensions):
    """
    List all files in a folder with specific extensions.

    Parameters:
        folder_path (str): Path to the folder to scrape.
        extensions (list): List of file extensions to include (e.g., ['.dat', '.json']).

    Returns:
        list: A list of file paths with the specified extensions.
    """
    return [os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.endswith(tuple(extensions))]

def get_files(folder_path):

    print(f"Searching for .dat, .json and .npz files in: {folder_path}")

    all_files = list_files_by_extension(folder_path, [".dat", ".json", ".npz"])
    dat_files = [f for f in all_files if f.endswith(".dat")]
    json_files = [f for f in all_files if f.endswith(".json")]
    npz_files = [f for f in all_files if f.endswith(".npz")]


    print(f"Found {len(dat_files)} .dat file(s).")
    print(f"Found {len(json_files)} .json file(s).")
    print(f"Found {len(npz_files)} .npz file(s).")

    print(f"Total files found: {len(all_files)}")
    if not all_files:
        print("No files found in the specified folder.")
        sys.exit(1)
    return dat_files, json_files, npz_files


def validate_input(config, dat_files, json_files, npz_files):
    file_type = config["file_type"]
    file_index = config["file_index"]

    if file_type not in ["dat", "json", "npz"]:
        print("Invalid file type. Please choose 'dat', 'json' or 'npz'.")
        sys.exit(1)

    file_count = (
        len(dat_files) if file_type == "dat" 
        else len(npz_files) if file_type == "npz" 
        else len(json_files)
    )

    if not (1 <= file_index <= file_count):
        print(f"Invalid file index for {file_type} files. Choose a number between 1 and {file_count}.")
        sys.exit(1)