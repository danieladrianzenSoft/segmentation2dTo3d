"""Utilities to combine multiple .glb files into a single .glb (optionally Draco-compressed).

This keeps logic separate from the mesh generation code and re-uses the existing
`compress_glb` helper in `core.mesh_generation_methods` when compression is requested.
"""
from typing import List, Optional
import os
import glob
import trimesh
from core.mesh_generation_methods import compress_glb, color_scene_unique, distinguishable_colors, apply_material_color, apply_vertex_color
import numpy as np
import colorsys

def _natural_sort_key(s: str):
    """Return a key for natural/numeric-aware sorting (e.g., pore2 < pore10).

    Splits text into int and non-int chunks to allow numeric-aware comparison.
    """
    import re
    parts = re.split(r"(\d+)", s)
    key = []
    for p in parts:
        if p.isdigit():
            key.append(int(p))
        else:
            key.append(p.lower())
    return key


def list_glb_files(input_dir: str, pattern: str = "*.glb", numeric_sort: bool = True) -> List[str]:
    """Return sorted list of .glb files matching pattern inside input_dir.

    Args:
        numeric_sort: If True, use numeric-aware ordering (pore2 < pore10). Defaults to True.
    """
    files = glob.glob(os.path.join(input_dir, pattern))
    if numeric_sort:
        files = sorted(files, key=_natural_sort_key)
    else:
        files = sorted(files)
    return files


def unite_glb_files(
    input_dir: str,
    output_path: str,
    pattern: str = "*.glb",
    compress: bool = False,
    compression_level: int = 10,
    max_files: Optional[int] = None,
    start_index: Optional[int] = None,
    end_index: Optional[int] = None,
    color: bool = False,
    color_method: str = "per_mesh",  # 'material' (deprecated), 'vertex' (legacy), 'per_mesh' (fast per-mesh vertex coloring)
    alpha: float = 1.0,
    numeric_sort: bool = True,
    verbose: bool = True,
) -> str:
    """Combine multiple .glb files into a single .glb file.

    Args:
        input_dir: Directory containing .glb files (searched with `pattern`).
        output_path: Path to write combined .glb file.
        pattern: Glob pattern to match input files (default: "*.glb").
        compress: If True, run Draco compression via `gltf-pipeline` (Node.js).
        compression_level: Integer 0-10 passed through to the compressor.
        max_files: If provided, only use the first `max_files` files (applied after slicing).
        start_index: 0-based start index (alphabetical order) of files to include.
        end_index: 0-based end index (inclusive) of files to include.
        color: If True, apply visually-distinct colors to each pore geometry before export.
        alpha: Alpha value to use for per-vertex coloring.
        verbose: Print progress messages.

    Returns:
        Path to the saved .glb file (compressed path if compression requested).

    Raises:
        ValueError: if no .glb files are found or no geometry is loaded.
    """
    files = list_glb_files(input_dir, pattern, numeric_sort=numeric_sort)

    # Accept string inputs (CLI overrides) and normalize indices
    if start_index is not None:
        start_index = int(start_index)
        if start_index < 0:
            start_index = 0
    if end_index is not None:
        end_index = int(end_index)
        if end_index < 0:
            end_index = 0

    # Apply start/end slicing first (end is inclusive)
    if start_index is not None or end_index is not None:
        s = start_index if start_index is not None else 0
        e = (end_index + 1) if end_index is not None else None
        files = files[s:e]

    # Apply max_files cap after slicing
    if max_files is not None:
        max_files = int(max_files)
        files = files[:max_files]

    if len(files) == 0:
        raise ValueError(f"No .glb files found in {input_dir} matching {pattern}.")

    scene = trimesh.Scene()
    added = 0

    # Prepare colors for per-mesh coloring if requested
    # Use distinguishable_colors by default; fall back to HSV if it fails (e.g., very large n or constraints too tight).
    file_colors = None
    if color and color_method in ("material", "per_mesh"):
        try:
            file_colors = distinguishable_colors(len(files), 'w')
        except Exception:
            try:
                # Fast HSV-based color generation — very cheap and incremental
                def generate_colors(n):
                    hues = np.linspace(0, 1, n, endpoint=False)
                    cols = [colorsys.hsv_to_rgb(float(h), 0.65, 0.95) for h in hues]
                    return cols
                file_colors = generate_colors(len(files))
            except Exception:
                file_colors = [(0.8, 0.8, 0.8)] * len(files)

    total = len(files)
    # Always show a progress bar (tqdm if available, else a simple in-place counter)
    try:
        from tqdm import tqdm
        file_iter = enumerate(tqdm(files, desc="Combining meshes", unit="file"))
        use_manual_progress = False
    except Exception:
        file_iter = enumerate(files)
        use_manual_progress = True

    for idx, f in file_iter:
        # Manual progress fallback
        if use_manual_progress:
            if idx % max(1, total // 100) == 0 or idx == total - 1:
                print(f"Combining meshes: {idx+1}/{total}", end="\r", flush=True)

        try:
            # Force scene so we can iterate over contained geometry
            loaded = trimesh.load(f, force="scene")

            # Determine file-level color if per-mesh method selected
            file_color = None
            if file_colors is not None:
                file_color = file_colors[idx]

            if isinstance(loaded, trimesh.Trimesh):
                if color and file_color is not None:
                    if color_method in ("material", "per_mesh"):
                        # fast path: apply per-mesh vertex color (reliable export)
                        apply_vertex_color(loaded, file_color, alpha=alpha, verbose=False)
                    else:
                        # slower path will color scene after assembly
                        pass
                node_name = os.path.basename(f)
                scene.add_geometry(loaded, node_name=node_name)
                added += 1
            else:
                # loaded is a Scene
                for name, geom in loaded.geometry.items():
                    if color and file_color is not None:
                        if color_method in ("material", "per_mesh"):
                            apply_vertex_color(geom, file_color, alpha=alpha, verbose=False)
                        else:
                            pass
                    node_name = f"{os.path.basename(f)}_{name}"
                    scene.add_geometry(geom, node_name=node_name)
                    added += 1
        except Exception as e:
            if verbose:
                print(f"⚠️ Skipping {f}: {e}")
            continue

    # If we used the manual progress fallback, ensure we end with a newline
    if use_manual_progress and total > 0:
        print()

    if len(scene.geometry) == 0:
        raise ValueError("No geometry could be loaded from the provided .glb files.")

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    # Apply coloring if requested and using legacy per-scene vertex coloring
    if color and color_method == "vertex":
        try:
            color_scene_unique(scene, colors=None, alpha=alpha, verbose=verbose)
        except Exception as e:
            if verbose:
                print(f"⚠️ Coloring failed: {e}")

    scene.export(output_path, file_type="glb")
    if verbose:
        print(f"✅ Combined GLB saved: {output_path} (objects added: {len(scene.geometry)})")

    if compress:
        compressed_path = output_path.replace(".glb", "_compressed.glb")
        compress_glb(output_path, compressed_path, compression_level=compression_level)
        return compressed_path

    return output_path
