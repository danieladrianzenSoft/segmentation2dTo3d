def get_config():
    return {
        "input_dir": "data/PoreMeshesIndividual/fd8ccdd0-2559-4848-8ac4-d375f53c0c75/pores",
        "output_dir": "data/PoreMeshes",
        "output_name": "combined_pores.glb",
        "pattern": "*.glb",
        "compress": True,
        "compress_in_place": True,
        "compression_level": 10,
        "max_files": None,
        "start_index": None,
        "end_index": None,
        "color": True,
        "color_method": "per_mesh",
        "alpha": 1.0,
        "numeric_sort": True,
    }
