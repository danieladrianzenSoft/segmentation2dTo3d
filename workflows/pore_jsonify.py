"""
Objective: 
convert .mat files containing pore data into voxelized
jsons for future plotting and meshing purposes.

"""

from pathlib import Path

def get_config(input_dir: Path = None, output_dir: Path = None, filename: str = None):
	repo_root = Path(__file__).resolve().parents[1]
	# input_dir = input_dir if input_dir else repo_root / "data" / "PoreDomainsToMesh" / "MATLAB"
	# output_dir = output_dir if output_dir else repo_root / "data" / "PoreDomainsToMesh" / "Json"
	input_dir = "/Users/mimc/Documents/MIMC/MaterialsAI/DanielAdrianzen/lovamap_gateway_data/PorePathData/MissingDomains_111825/subunits"
	output_dir = "/Users/mimc/Documents/MIMC/MaterialsAI/DanielAdrianzen/lovamap_gateway_data/PorePathData/MissingDomains_111825/subunits_json"
	
	config = {
		"input_dir": input_dir,
		"output_dir": output_dir,
		# "filename": filename,
		"file_index": 1,
		"file_type": "mat",
		"batch_process": True
	}
	return config