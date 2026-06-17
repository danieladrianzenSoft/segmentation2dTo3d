# Docker Workflow Reference

Reference for the public workflows exposed via the Docker image (`ghcr.io/seguralab/segmentation-workflows`). Use this when setting up programmatic endpoints that invoke these containers.

---

## General Usage

```bash
docker run --rm \
  -v /path/to/local/data:/data:rw \
  ghcr.io/seguralab/segmentation-workflows:latest \
  --workflow <workflow_name> \
  --config /app/configs/<workflow_name>.json \
  --set key1=value1 key2=value2
```

- `--config` loads the default config baked into the image
- `--set` overrides individual parameters (type-aware: booleans, ints, floats are auto-cast)
- Volume mount at `/data` is the standard I/O path

---

## mesh_generation

Converts JSON files describing scaffold geometries (particles or pores with voxel indices) into 3D mesh files (`.glb`) using Marching Cubes.

### Input

- A directory containing `.json` files, each representing a voxelized domain (scaffold geometry)
- Each JSON file contains entities (particles or pores) with their voxel indices, domain size, and voxel size

### Output

- Per-file `.glb` mesh (binary glTF) with all entities as named geometries
- Per-file `_metadata.json` with entity IDs and properties (volume, surface area, etc.)

### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `input_dir` | string | `/data/input` | Directory containing input JSON files |
| `output_dir` | string | `/data/output` | Directory where meshes are written |
| `file_type` | string | `"json"` | Input format: `"json"`, `"dat"`, `"npz"`, `"mat"` |
| `batch_process` | bool | `true` | Process all files in directory. Set `false` for single file (requires `file_index`) |
| `file_index` | int | `1` | Which file to process when `batch_process` is `false` |
| `show_edge_pores` | bool | `true` | Include edge pores in output. Only relevant for pore domains |
| `save_metadata` | bool | `true` | Generate `_metadata.json` alongside each mesh |
| `save_mesh` | bool | `true` | Generate `.glb` mesh file |
| `scrape_subdirectories` | bool | `false` | Recurse into subdirectories and mirror structure in output |

### Example

```bash
docker run --rm \
  -v /path/to/scaffold_jsons:/data/input:ro \
  -v /path/to/output:/data/output:rw \
  ghcr.io/seguralab/segmentation-workflows:latest \
  --workflow mesh_generation \
  --config /app/configs/mesh_generation.json
```

With overrides:

```bash
docker run --rm \
  -v /path/to/data:/data:rw \
  ghcr.io/seguralab/segmentation-workflows:latest \
  --workflow mesh_generation \
  --config /app/configs/mesh_generation.json \
  --set input_dir=/data/my_scaffolds output_dir=/data/meshes show_edge_pores=false
```

### Programmatic invocation (pseudocode)

```python
container = docker.run(
    image="ghcr.io/seguralab/segmentation-workflows:latest",
    volumes={input_path: "/data/input", output_path: "/data/output"},
    command=[
        "--workflow", "mesh_generation",
        "--config", "/app/configs/mesh_generation.json",
        "--set", f"input_dir=/data/input", f"output_dir=/data/output"
    ]
)
# Output: .glb and _metadata.json files written to output_path
```

---

## unite_meshes

Combines multiple individual `.glb` mesh files into a single combined `.glb` file. Supports optional Draco compression and per-mesh coloring.

### Input

- A directory containing individual `.glb` files (e.g., individual pore meshes)

### Output

- A single combined `.glb` file containing all input meshes as named geometries
- Optionally compressed via Draco (gltf-pipeline)

### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `input_dir` | string | `/data/PoreMeshesIndividual/POOL/pores` | Directory containing `.glb` files to combine |
| `output_dir` | string | `/data/out` | Directory where combined mesh is written |
| `output_name` | string | `"combined.glb"` | Filename for the combined output |
| `pattern` | string | `"*.glb"` | Glob pattern for matching input files |
| `start_index` | int\|null | `null` | Start index for file subset (0-based, inclusive) |
| `end_index` | int\|null | `null` | End index for file subset (inclusive) |
| `max_files` | int\|null | `null` | Maximum number of files to process |
| `compress` | bool | `true` | Apply Draco compression (requires gltf-pipeline) |
| `compress_in_place` | bool | `true` | Replace original with compressed version |
| `compression_level` | int | `10` | Draco compression level (1-10, higher = smaller file) |
| `color` | bool | `false` | Apply distinguishable colors to each mesh |
| `color_method` | string | `"per_mesh"` | Coloring strategy: `"per_mesh"` or `"per_vertex"` |
| `alpha` | float | `1.0` | Opacity (0.0 - 1.0) |
| `numeric_sort` | bool | `true` | Sort files numerically (pore2 before pore10) |
| `flip_yz` | bool | `false` | Swap Y and Z axes to convert Z-up (microscopy) to Y-up (glTF) |
| `verbose` | bool | `false` | Print detailed progress |

### Example

```bash
docker run --rm \
  -v /path/to/individual_meshes:/data/input:ro \
  -v /path/to/output:/data/out:rw \
  ghcr.io/seguralab/segmentation-workflows:latest \
  --workflow unite_meshes \
  --config /app/configs/unite_meshes.json \
  --set input_dir=/data/input output_dir=/data/out compress=true
```

Subset of files:

```bash
docker run --rm \
  -v /path/to/data:/data:rw \
  ghcr.io/seguralab/segmentation-workflows:latest \
  --workflow unite_meshes \
  --config /app/configs/unite_meshes.json \
  --set input_dir=/data/meshes output_dir=/data/combined start_index=0 end_index=50
```

### Programmatic invocation (pseudocode)

```python
container = docker.run(
    image="ghcr.io/seguralab/segmentation-workflows:latest",
    volumes={meshes_path: "/data/input", output_path: "/data/out"},
    command=[
        "--workflow", "unite_meshes",
        "--config", "/app/configs/unite_meshes.json",
        "--set", "input_dir=/data/input", "output_dir=/data/out",
        "compress=true", "color=false"
    ]
)
# Output: combined.glb (and possibly combined_compressed.glb) in output_path
```

---

## Exit Codes

- `0` — Success
- `1` — Error (check container logs/stderr for details)

## Notes

- The container runs as non-root (`appuser`). Ensure mounted output directories are writable.
- All `--set` values are type-cast based on the config defaults (e.g., `"true"` → `bool`, `"10"` → `int`).
- `null` in JSON configs means "not set" / use internal default behavior.
- The image includes gltf-pipeline for Draco compression — no extra setup needed.
