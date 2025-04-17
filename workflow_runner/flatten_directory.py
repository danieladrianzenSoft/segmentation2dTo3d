from core.directory_flattening_methods import flatten_directory 

def run(config):
    flatten_directory(
        directory=config["directory"],
        output_directory=config["output_directory"],
        file_types=config["file_types"],
        excluded_directories=config["excluded_directories"],
        excluded_files=config["excluded_files"]
    )