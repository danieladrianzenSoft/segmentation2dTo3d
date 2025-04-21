from pathlib import Path

def get_config(input_dir: Path = None):
	repo_root = Path(__file__).resolve().parents[1]
	domain_path = input_dir if input_dir else repo_root / "data" / "ParticleDomainTestData"
	# domain_path = input_dir if input_dir else repo_root / "data" / "PoreDomainsToMesh" / "Json"

	config = {
		"folder_path": domain_path,
		"file_type": "json",
		"file_index": 1,
		"downsample_factor": 1,
		"surface_only": True,
		"max_particles": None,
		"show_legend": False,
		"show_edge_pores": False
	}
	return config