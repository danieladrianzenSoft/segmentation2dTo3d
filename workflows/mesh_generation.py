from pathlib import Path

def get_config(input_dir: Path = None, output_dir: Path = None):
    repo_root = Path(__file__).resolve().parents[1]

    ## PORE
    # Test
    # input_dir = input_dir if input_dir else repo_root / "data" / "PoreDomainsToMesh" / "Json"
    # output_dir = output_dir if output_dir else repo_root / "data" / "PoreMeshes"

    # Real
    input_dir = "/Users/mimc/Documents/MIMC/MaterialsAI/DanielAdrianzen/lovamap_gateway_data/PorePathData/lovamap_outputs/subunits_json"
    output_dir = "/Users/mimc/Documents/MIMC/MaterialsAI/DanielAdrianzen/lovamap_gateway_data/PorePathData/lovamap_outputs/subunits_meshes"

    ## PARTICLES

    # Test
    # input_dir = input_dir if input_dir else repo_root / "data" / "ParticleDomainsToMesh"
    # output_dir = output_dir if output_dir else repo_root / "data" / "ParticleMeshes"

    # Real
    # input_dir = "/Users/dzen/Documents/LOVAMAP/DomainJsons/ReadyForMeshing"
    # output_dir = "/Users/dzen/Documents/LOVAMAP/DomainMeshes"

    config = {
        "input_dir": input_dir,
        "output_dir": output_dir,
        "file_index": 1,
        "file_type": "json",
        "batch_process": True,
        "show_edge_pores": True,
        "save_metadata": True,
        "save_mesh": True,
        # "filename": "labeledDomain_soft-spheres_100_v0.json"
    }
    
    return config