def get_config():
	config = {
		"folder_path": "/Users/mimc/Documents/MIMC/MaterialsAI/DanielAdrianzen/segmentation_2D3D/FlattenedDataJson",
		"output_dir_slices":  "/Users/mimc/Documents/MIMC/MaterialsAI/DanielAdrianzen/segmentation_2D3D/slices_512x512x128",
		"output_dir_labels": "/Users/mimc/Documents/MIMC/MaterialsAI/DanielAdrianzen/segmentation_2D3D/labels_512x512x128",
		"metadata_path": "/Users/mimc/Documents/MIMC/MaterialsAI/DanielAdrianzen/segmentation_2D3D/metadata_512x512x128.json",
		"batch_process": True,
		"restart": False,
		"file_type": "json",
		"file_index": 2,
		"downsample_factor": 1,
		"surface_only": True,
		"max_particles": None,
		"random_slice_spacing": False,
		"random_axis": False,
		"max_slices": 150,
		"num_slices": 128,
		"axis": "z",
		"show_legend": False,
		"voxelization_dx": 2,
		"slices_debug_mode": False,
		"generate_images": True,
		"generate_labels": True,
		"visualize_label_slice_creation": False
	}

	return config