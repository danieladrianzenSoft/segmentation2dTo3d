def get_config():
	config = {
		"folder_path": "./FlattenedData",
		"batch_process": True,
		"restart": True,
		"file_type": "json",
		"file_index": 2,
		"downsample_factor": 1,
		"surface_only": True,
		"max_particles": None,
		"random_slice_spacing": True,
		"random_axis": True,
		"max_slices": 150,
		"num_slices": 5,
		"axis": "z",
		"show_legend": False,
		"voxelization_dx": 2,
		"slices_debug_mode": False,
		"generate_images": True,
		"output_dir": "slices",
		"metadata_path": "metadata.json"
	}
	return config