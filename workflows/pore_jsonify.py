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
	# input_dir = "/Users/dzen/Documents/LOVAMAP/Data/Domains/SubunitsMatlab/Real"
	# output_dir = "/Users/dzen/Documents/LOVAMAP/Data/Domains/Subunits/Real"
	dir_name = "PhysicalContinuityFibroblasts_BioArxiv_SuarezArnedo"
	input_dir = "/Users/dzen/Library/CloudStorage/Box-Box/Lindsay Riley PhD/Electronic Notebook/Void Space Project/MIMC/Data/Domains/Subunits/Real/Matlab/" + dir_name
	output_dir = output_dir if output_dir else repo_root / "data" / "SubunitJsons" / dir_name

	config = {
		"input_dir": input_dir,
		"output_dir": output_dir,
		# "filename": filename,
		"file_index": 1,
		"file_type": "mat",
		"batch_process": True,
		"scrape_subdirectories": True,
	}
	return config