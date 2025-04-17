from pathlib import Path

def get_config(input_dir: Path = None):
	repo_root = Path(__file__).resolve().parents[1]
	domain_path = input_dir if input_dir else repo_root / "data" / "DomainTestData"
	output_dir = repo_root / "data" / "DomainsToMeshOutputs"

	config = {
		"folder_path": domain_path,
		"output_dir": output_dir,
		"batch_process": False,
		"restart": False,
		"file_type": "json",
		"file_index": 1,
		"downsample_factor": 1,
		"surface_only": True,
		"max_particles": None,
		"show_legend": False,
		"voxelization_dx": 2,
		"slices_debug_mode": True,
		"generate_images": False,
		"generate_labels": False,
		"metadata_path": "/Users/mimc/Documents/MIMC/MaterialsAI/DanielAdrianzen/segmentation_2D3D/metadata.json"
	}
	return config