from pathlib import Path

def get_config(input_dir: Path = None, output_dir: Path = None):
    repo_root = Path(__file__).resolve().parents[1]

    ## PORE
    input_dir = input_dir if input_dir else repo_root / "data" / "PoreDomainsToMesh" / "Json"
    output_dir = output_dir if output_dir else repo_root / "data" / "PoreMeshes"

    ## PARTICLES
    # input_dir = input_dir if input_dir else repo_root / "data" / "ParticleDomainsToMesh"
    # output_dir = output_dir if output_dir else repo_root / "data" / "ParticleMeshes"

    # input_dir = "/Users/dzen/Documents/LOVAMAP/DomainJsons/RemainingToMesh",
    # output_dir = "/Users/dzen/Documents/LOVAMAP/DomainMeshes",

    config = {
        "input_dir": input_dir,
        "output_dir": output_dir,
        "file_index": 1,
        "file_type": "json",
        "batch_process": False,
        "show_edge_pores": False
    }
    
    return config