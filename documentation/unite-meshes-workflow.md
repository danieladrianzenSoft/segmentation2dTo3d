# Unite Meshes Workflow

The **unite_meshes** workflow combines multiple individual `.glb` mesh files
into a single combined `.glb` file. Supports optional Draco compression,
per-mesh coloring, and coordinate axis conversion.

---

## Quick Start

```bash
# Local
python -m workflows.unite_meshes

# Or via Docker (see docker-workflow-reference.md)
```

Configure via `configs/unite_meshes.json` or environment variables.

---

## High-Level Flow

```
                       unite_meshes workflow
                       =====================

  input_dir/*.glb
       │
       ▼
  list & sort files
  (natural/numeric order)
       │
       ▼
  [optional slice: start_index..end_index]
  [optional cap: max_files]
       │
       ▼
  for each .glb file:
       │
       ├──► load geometry (trimesh)
       │
       ├──► [flip_yz] swap Y ↔ Z axes
       │
       ├──► [color] apply per-mesh vertex color
       │         (distinguishable_colors / HSV fallback)
       │
       └──► add to combined Scene
                   │
                   ▼
            trimesh.Scene.export()
                   │
              ┌────┴────┐
              ▼         ▼
           .glb      [Draco
          (raw)    compressed]
```

---

## Coordinate System Convention

Microscopy data typically uses a **Z-up** coordinate system (X and Y form the
image plane, Z is the optical/stacking axis). The glTF 2.0 spec defines a
right-handed **Y-up** system (Y vertical, Z toward the viewer).

The **mesh_generation** workflow applies the Y↔Z swap automatically (see
[mesh-generation-workflow.md](mesh-generation-workflow.md)). When combining
meshes from external sources that are still in Z-up (microscopy) coordinates,
set `flip_yz: true` so the combined output follows the same Y-up convention as
meshes produced by mesh_generation.

---

## Configuration Reference

| Option              | Type      | Default          | Description                                                      |
|---------------------|-----------|------------------|------------------------------------------------------------------|
| `input_dir`         | path      | —                | Directory containing `.glb` files to combine                     |
| `output_dir`        | path      | `data/PoreMeshes`| Directory where the combined mesh is written                     |
| `output_name`       | string    | `"combined.glb"` | Filename for the combined output                                 |
| `pattern`           | string    | `"*.glb"`        | Glob pattern for matching input files                            |
| `start_index`       | int\|null | `null`           | Start index for file subset (0-based, inclusive)                 |
| `end_index`         | int\|null | `null`           | End index for file subset (inclusive)                             |
| `max_files`         | int\|null | `null`           | Maximum number of files to process                               |
| `compress`          | bool      | `true`           | Apply Draco compression (requires `gltf-pipeline`)               |
| `compress_in_place` | bool      | `true`           | Replace original with compressed version                         |
| `compression_level` | int       | `10`             | Draco compression level (1-10, higher = smaller file)            |
| `color`             | bool      | `true`           | Apply distinguishable colors to each mesh                        |
| `color_method`      | string    | `"per_mesh"`     | Coloring strategy: `"per_mesh"`, `"vertex"`, or `"material"`     |
| `alpha`             | float     | `1.0`            | Opacity (0.0 - 1.0)                                             |
| `numeric_sort`      | bool      | `true`           | Sort files numerically (pore2 before pore10)                     |
| `flip_yz`           | bool      | `false`          | Swap Y and Z axes to convert Z-up (microscopy) to Y-up (glTF)   |

### Example config (`configs/unite_meshes.json`)

```json
{
  "input_dir": "data/PoreMeshesIndividual/pores",
  "output_dir": "data/PoreMeshes",
  "output_name": "combined_pores.glb",
  "compress": true,
  "compress_in_place": true,
  "color": true,
  "color_method": "per_mesh",
  "flip_yz": false
}
```

---

## Docker Usage

```bash
docker run --rm \
  -v /path/to/individual_meshes:/data/input:ro \
  -v /path/to/output:/data/out:rw \
  ghcr.io/seguralab/segmentation-workflows:latest \
  --workflow unite_meshes \
  --config /app/configs/unite_meshes.json \
  --set input_dir=/data/input output_dir=/data/out compress=true

# With Y↔Z flip for Z-up source meshes:
docker run --rm \
  -v /path/to/individual_meshes:/data/input:ro \
  -v /path/to/output:/data/out:rw \
  ghcr.io/seguralab/segmentation-workflows:latest \
  --workflow unite_meshes \
  --config /app/configs/unite_meshes.json \
  --set input_dir=/data/input output_dir=/data/out flip_yz=true
```

---

## Dependencies

| Package          | Purpose                                |
|------------------|----------------------------------------|
| `trimesh`        | Scene loading, combination, export     |
| `numpy`          | Vertex array operations                |
| `gltf-pipeline`  | Draco compression (Node.js, via CLI)   |

---

## Error Handling

| Condition                          | Behavior                                     |
|------------------------------------|----------------------------------------------|
| No `.glb` files found              | `ValueError` raised                          |
| No geometry loaded from files      | `ValueError` raised                          |
| Individual file fails to load      | Skipped with warning                         |
| Draco compression fails            | Returns uncompressed `.glb` path             |
| Coloring fails                     | Continues without colors (warning printed)   |
