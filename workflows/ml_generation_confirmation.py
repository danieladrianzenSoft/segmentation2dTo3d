def get_config():
	config = {
		"folder_path": "./TestData",
		"batch_process": False,
		"restart": False,
		"file_type": "json",
		"file_index": 3,
		"downsample_factor": 1,
		"surface_only": True,
		"max_particles": None,
		"random_slice_spacing": True,
		"random_axis": False,
		"max_slices": 150,
		"num_slices": 5,
		"axis": "z",
		"show_legend": False,
		"voxelization_dx": 2,
		"slices_debug_mode": False,
		"generate_images": False,
		"generate_labels": False,
		"output_dir_slices": "slices_TestData",
		"output_dir_labels": "labels_TestData",
		"metadata_path": "metadata_TestData.json",
		"visualize_label_slice_creation": True
	}
	return config