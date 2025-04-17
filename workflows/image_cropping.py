from pathlib import Path
from core.path_manipulation_methods import resolve_path

def get_config(input_path: Path = None, output_path: Path = None):
    # Resolve to repo root based on this file’s location
    repo_root = Path(__file__).resolve().parents[1]
    image_dir = repo_root / "data" / "Images"

    # Default paths assume test data in the repo
    input_path = resolve_path(input_path, image_dir) if input_path else image_dir / "Duke_logo.png"
    output_path = resolve_path(output_path, image_dir) if output_path else image_dir / "Duke_logo_cropped.png"

    config = {
        "input_path": input_path,
        "output_path": output_path,
        "left": "15%",
        "right": "15%",
        "top": "15%",
        "bottom": "15%"
	}
    return config

