from core.mesh_combination_methods import unite_glb_files
import os


def run(config: dict):
    print(f"🔧 Running unite_meshes with config: {config}")
    input_dir = config.get("input_dir")
    output_dir = config.get("output_dir", "data/PoreMeshes")
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, config.get("output_name", "combined.glb"))

    result = unite_glb_files(
        input_dir=input_dir,
        output_path=output_path,
        pattern=config.get("pattern", "*.glb"),
        compress=config.get("compress", False),
        compress_in_place=config.get("compress_in_place", False),
        compression_level=config.get("compression_level", 10),
        max_files=config.get("max_files", None),
        start_index=config.get("start_index", None),
        end_index=config.get("end_index", None),
        color=config.get("color", False),
        color_method=config.get("color_method", "per_mesh"),
        alpha=config.get("alpha", 1.0),
        numeric_sort=config.get("numeric_sort", True),
        verbose=True,
    )

    print(f"🎯 Unite meshes result: {result}")
    return result
