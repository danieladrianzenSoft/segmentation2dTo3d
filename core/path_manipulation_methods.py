from pathlib import Path

def resolve_path(path_str, default_dir: Path) -> Path:
    """
    Resolves a path intelligently:
    - If the input is an absolute path, returns it as-is.
    - If it's a relative filename (e.g., 'image.png'), returns default_dir / filename.
    - If it's already a Path object, handles it correctly.

    Parameters:
        path_str (str or Path): The user-provided path (can be relative or absolute).
        default_dir (Path): The base directory to resolve against if the path is relative.

    Returns:
        Path: An absolute Path object.
    """
    path = Path(path_str)
    return path if path.is_absolute() else (default_dir / path).resolve()