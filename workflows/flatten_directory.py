from pathlib import Path

def get_config(
    base_data_dir: Path = None,
    output_dir: Path = None
):
    # Resolve to repo root based on this file’s location
    repo_root = Path(__file__).resolve().parents[1]

    # Default paths assume test data in the repo
    default_input = repo_root / "data" / "ParticleDomainData"
    default_output = repo_root / "data" / "FlattenedData"

    config = {
        "directory": base_data_dir or default_input,
        "output_directory": output_dir or default_output,
        "file_types": [".dat", ".json"],
        "excluded_directories": ["old", "wrong", "alex", "gong", "combined", "consolidated"],
        "excluded_files": ["old", "wrong", "real", "alex", "gong", "combined", "consolidated"]
    }

    return config