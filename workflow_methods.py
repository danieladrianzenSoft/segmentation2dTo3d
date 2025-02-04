def get_workflow_config(workflow_method='test_generation'):
    if (workflow_method=='batch_generation'):
        config = {
            "folder_path": "./FlattenedData",
            "batch_process": True,
            "restart": False,
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
    elif (workflow_method=='batch_generation_restart'):
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
    elif (workflow_method=='standardize_json'): 
        config = {
            "folder_path": "./FlattenedData",
            "output_dir": "./FlattenedDataJson",
            "dat_to_json": True,
            "batch_process": True,
            "voxelization_dx": 2,
        }
    elif (workflow_method == 'batch_generate_labels_slices'):
        config = {
            "folder_path": "/Users/mimc/Documents/MIMC/MaterialsAI/DanielAdrianzen/segmentation_2D3D/FlattenedDataJson",
            "batch_process": True,
            "output_dir_slices": "/Users/mimc/Documents/MIMC/MaterialsAI/DanielAdrianzen/segmentation_2D3D/slices",
            "output_dir_labels": "/Users/mimc/Documents/MIMC/MaterialsAI/DanielAdrianzen/segmentation_2D3D/labels",
            "metadata_path": "/Users/mimc/Documents/MIMC/MaterialsAI/DanielAdrianzen/segmentation_2D3D/metadata.json",
            "generate_labels": True,
            "generate_images": True,
            "dat_to_json": False,
            "surface_only": True,
            "restart": False,
            "random_slice_spacing": True,
            "random_axis": True,
            "max_slices": 150,
            "num_slices": 5,
            "axis": "z",
            "voxelization_dx": 2,
            "slices_debug_mode": False,
    }
    elif (workflow_method == 'test_generate_labels'):
        config = {
            "folder_path": "/Users/mimc/Documents/MIMC/MaterialsAI/DanielAdrianzen/segmentation_2D3D/FlattenedDataJson",
            "batch_process": False,
            "output_dir": "Labels_TestData",
            "generate_labels": True,
            "dat_to_json": False,
            "surface_only": True,
            "file_type": "json",
            "file_index": 1,
            "num_slices": 50,
            "slices_debug_mode": True,
            "random_axis": False,
            "axis": "z",
            "generate_images": False,
            "metadata_path": "metadata_TestData.json"
        }
    elif (workflow_method=='test_generation'):
        config = {
            "folder_path": "./TestData",
            "batch_process": False,
            "restart": False,
            "file_type": "dat",
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
            "output_dir": "slices_TestData",
            "metadata_path": "metadata_TestData.json"
        }
    else:
        config = {
            "folder_path": "./TestData",
            "batch_process": False,
            "restart": False,
            "file_type": "dat",
            "file_index": 1,
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
            "slices_debug_mode": True,
            "generate_images": False,
            "output_dir": "slices_TestData",
            "metadata_path": "metadata_TestData.json"
        }

    return config