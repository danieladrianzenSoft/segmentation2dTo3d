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
    result = {ext: [] for ext in extensions}
    for f in os.listdir(folder_path):
        for ext in extensions:
            if f.endswith(ext):
                result[ext].append(os.path.join(folder_path, f))
    return result

def get_files(folder_path, extensions=None):
    """
    Search for files by extension in a given folder.

    Parameters:
        folder_path (str): Folder to search.
        extensions (list, optional): Extensions to search for. Default is ['.dat', '.json', '.npz'].

    Returns:
        tuple: A tuple of lists of file paths, in the same order as the extensions input.
    """

    if extensions is None:
        extensions = [".dat", ".json", ".npz"]

    print(f"Searching for files with extensions {extensions} in: {folder_path}")

    all_files = list_files_by_extension(folder_path, extensions)

    results = []
    total = 0
    for ext in extensions:
        files = all_files.get(ext, [])
        print(f"Found {len(files)} {ext} file(s).")
        results.append(files)
        total += len(files)

    print(f"Total files found: {total}")
    if total == 0:
        print("No files found in the specified folder.")
        sys.exit(1)

    return tuple(results)

def select_input_file(config, file_lists, extensions=None):
    """
    Select and validate the file to process.

    If 'filename' is provided in config, it will be used directly (after validation).
    Otherwise, selection is based on 'file_type' and 'file_index'.

    Parameters:
        config (dict): Must include 'file_type' and either 'file_index' or 'filename'.
        file_lists (list of lists): Output of get_files(...), converted to a list.
        extensions (list of str, optional): Corresponding extensions list (default = ['.dat', '.json', '.npz']).

    Returns:
        str: Path to the selected file.
    """
    # If extensions not provided, assume default
    if extensions is None:
        extensions = ['.dat', '.json', '.npz']

    # Use filename override if provided
    filename = config.get("filename")
    if filename and isinstance(filename, str):
        # If filename is not an absolute path, prepend input_dir
        if not os.path.isabs(filename):
            input_dir = config.get("input_dir", ".")
            filename = os.path.join(input_dir, filename)

        if not os.path.isfile(filename):
            print(f"Provided filename does not exist: {filename}")
            sys.exit(1)

        print(f"Using provided filename: {filename}")
        return filename

    # FALLBACK to file_type + index
    
    # Infer file_type if only one extension is available
    file_type = config.get("file_type")
    if not file_type:
        if len(extensions) == 1:
            file_type = extensions[0].lstrip('.')
            print(f"Inferred file_type = '{file_type}' from extensions")
        else:
            print("'file_type' must be specified when using multiple extensions.")
            sys.exit(1)

    file_index = config.get("file_index")
    if not isinstance(file_index, int):
        print("'file_index' must be an integer when using file_type selection.")
        sys.exit(1)

    ext = '.' + file_type

    # Default extension order
    if extensions is None:
        extensions = ['.dat', '.json', '.npz']

    if ext not in extensions:
        print(f"Extension '{ext}' not in provided extensions list.")
        sys.exit(1)

    idx = extensions.index(ext)
    file_list = file_lists[idx]

    if not (1 <= file_index <= len(file_list)):
        print(f"Invalid file index for {file_type}. Choose between 1 and {len(file_list)}.")
        sys.exit(1)

    return file_list[file_index - 1]