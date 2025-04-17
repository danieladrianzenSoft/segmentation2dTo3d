def get_config():
	config = {
		"input_directory": "../DomainsToMesh",
        "output_directory": "../DomainsToMesh",
        "filename": "labeledDomain_FEM_Beads-1_d70_s3.json",
        "threshold": 2  # Remove beads that touch top surface in n voxels
    }
	return config