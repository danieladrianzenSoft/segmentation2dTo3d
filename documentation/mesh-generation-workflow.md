# Mesh Generation Workflow

The **mesh_generation** workflow converts voxelized domain data (particles/pores)
or sphere definitions into optimized 3D `.glb` mesh files with metadata. It
supports multiple input formats and two primary meshing approaches.

---

## Quick Start

```bash
# Single file
python -m workflows.mesh_generation

# Or via Docker (see docker-workflow-reference.md)
```

Configure via `configs/mesh_generation.json` or environment variables.

---

## High-Level Flow

```
                        mesh_generation workflow
                        =======================

  ┌──────────────────┐  ┌──────────┐  ┌──────────┐
  │ .dat / .txt / .csv│  │  .json   │  │  .npz    │
  │    (spheres)      │  │ (voxels) │  │ (grids)  │
  └────────┬──────────┘  └────┬─────┘  └────┬─────┘
           │                  │              │
           ▼                  ▼              ▼
      parse_dat_file    parse_json_file  parse_npz_file
           │                  │              │
           │                  ├──────────────┘
           │                  ▼
           │           get_centered_grid()
           │                  │
           │           [filter edge pores]
           │                  │
           ▼                  ▼
      generate_mesh      generate_mesh
      _from_spheres      _marching_cubes
           │                  │
           │      ┌───────────┤
           │      │           │
           ▼      ▼           ▼
        distinguishable   marching_cubes
        _colors()         + simplify_mesh
           │                  │
           ▼                  ▼
        sRGB → linear    sRGB → linear
        color conversion color conversion
           │                  │
           └───────┬──────────┘
                   ▼
             trimesh.Scene
                   │
              ┌────┴────┐
              ▼         ▼
           .glb      _metadata
        (Draco)       .json
```

---

## Supported Input Formats

### `.dat` / `.txt` / `.csv` — Sphere Definitions

Tabular text files with one sphere per line (four numeric columns: x, y, z, radius).
All three formats are parsed by `parse_dat_file()` with identical validation:

- `.dat` and `.txt` — whitespace-separated
- `.csv` — comma-separated

```
# .dat / .txt (whitespace-separated)
x   y   z   radius
1.5 2.0 3.1 0.75
4.2 1.8 2.9 0.60

# .csv (comma-separated)
1.5,2.0,3.1,0.75
4.2,1.8,2.9,0.60
```

Lines that are empty, start with non-numeric characters, or don't have exactly
four columns are skipped. This allows comment/header lines to be present.

**Pipeline:** `parse_dat_file()` → `generate_mesh_from_spheres()` (icospheres)

### `.json` — Voxelized Domains (Particles or Pores)

Structured JSON with voxel index ranges. The parser auto-detects the domain type:

- **Pore data** — has a `"pores"` key
- **Particle data** — has a `"bead_data"` or `"beads"` key

```
Voxel ranges stored as [start, end] pairs (1-based, MATLAB-style)
  → Converted to 0-based flat indices internally
```

**Pore metadata fields preserved:**

| Field             | Description                        |
|-------------------|------------------------------------|
| `volume`          | Pore volume                        |
| `surfArea`        | Surface area                       |
| `charLength`      | Characteristic length              |
| `edge`            | 0 = interior, 1 = boundary pore   |
| `avgDoorDiam`     | Average door/throat diameter       |
| `largestDoorDiam` | Largest door/throat diameter       |
| `beadNeighbors`   | Count of neighboring beads         |

**Pipeline:** `parse_json_file()` → `get_centered_grid()` → `generate_mesh_marching_cubes()`

### `.npz` — NumPy Voxel Grids

NumPy archive with a labeled 3D voxel grid:

- `voxel_grid` (required) — 3D array where each value is a particle label
- `voxel_size` (optional, default 1.0)
- `grid_size` (optional)

**Pipeline:** Same as `.json` after parsing.

---

## Meshing Approaches

### 1. Marching Cubes (`.json` / `.npz` input)

Converts voxelized entities into smooth triangle meshes.

```
For each entity (pore/particle):

  Voxel indices
       │
       ▼
  Map to 3D coords ──► Build sparse voxel
  (via voxel_centers)   grid with +1 padding
                              │
                              ▼
                     Gaussian smoothing (σ=1.2)
                              │
                              ▼
                     Marching Cubes (level=0.5)
                              │
                              ▼
                     Quadric decimation
                     (target: 10,000 faces)
                              │
                              ▼
                     Fix outward normals
                              │
                              ▼
                     Apply per-vertex color
                     (sRGB → linear RGB)
```

**Key parameters:**
- Gaussian sigma: `1.2` — smooths voxel staircase artifacts
- Marching Cubes isovalue: `0.5`
- Target faces: `10,000` per entity (via Open3D quadric decimation)
- Grid padding: `+1` on all sides to prevent boundary clipping

### 2. Icospheres (`.dat` / `.txt` / `.csv` input)

Generates meshes directly from sphere centroids and radii.

```
For each sphere:

  center (x,y,z) + radius
       │
       ▼
  Create icosphere
  (subdivisions=3 → 642 verts, 1280 faces)
       │
       ▼
  Translate to center, scale by radius
       │
       ▼
  Apply per-vertex color
  (sRGB → linear RGB)
```

---

## Color Generation

Colors are generated using a perceptually-uniform greedy algorithm that produces
the same palette as the `buildPoreColorMap` function in **lovamap_gw**.

```
distinguishable_colors(n)
─────────────────────────
1. Build 40³ = 64,000 RGB candidates
2. Remove grays (R ≈ G ≈ B within tolerance)
3. Convert to CIE-LAB color space (D65)
4. Filter by lightness (L_min=0, L_max=100 → full range)
5. Greedy farthest-point selection:
   ┌─────────────────────────────────────────────┐
   │  seed distances from background (white)      │
   │  for i = 1..n:                               │
   │      pick candidate with max(min_dist)       │
   │      update min_dist with new selection       │
   └─────────────────────────────────────────────┘
6. Fisher-Yates shuffle (mulberry32 PRNG, seed=0)
   → Same permutation as TS version
```

**Cross-platform consistency:** The Python implementation uses the same
sRGB→LAB conversion constants, grid ordering, squared Euclidean distances,
and `mulberry32` PRNG as the TypeScript version in lovamap_gw, ensuring
identical palettes for the same input count and seed.

**glTF color space:** All colors are converted from sRGB to **linear RGB**
before being written to the `.glb`, as required by the glTF 2.0 spec.
Viewers (Three.js, etc.) apply sRGB encoding for display automatically.

---

## Coordinate System Convention

Microscopy data typically uses a **Z-up** coordinate system (X and Y form the
image plane, Z is the optical/stacking axis). The glTF 2.0 spec, however,
defines a right-handed **Y-up** system (Y vertical, Z toward the viewer).

To produce standard-compliant `.glb` files that display correctly in any viewer
(Three.js, Blender, online viewers, etc.), the mesh generation workflows swap
Y and Z axes before export:

- **`mesh_generation`** — applies the Y↔Z swap automatically in
  `generate_mesh_marching_cubes()` when mapping voxel coordinates.
- **`unite_meshes`** — accepts an optional `flip_yz` parameter (default
  `false`). Set to `true` when combining meshes that are still in Z-up
  (microscopy) coordinates so the output follows the same Y-up convention.

---

## Output Files

### `.glb` — 3D Mesh

- **Format:** glTF 2.0 binary
- **Compression:** Draco (level 10), via `gltf-pipeline`
- **Contents:**
  - One geometry node per entity (named by entity ID)
  - Per-vertex colors in linear RGB + alpha
  - Triangle faces (decimated)
  - Vertex normals (outward-facing)

### `_metadata.json` — Entity Metadata

```json
{
  "ids": ["1", "2", "3"],
  "id_to_index": { "1": 0, "2": 1, "3": 2 },
  "metadata": {
    "1": {
      "volume": 1234.5,
      "surfArea": 456.7,
      "charLength": 78.9,
      "edge": 0,
      "avgDoorDiam": 12.3,
      "largestDoorDiam": 45.6,
      "beadNeighbors": 8
    }
  }
}
```

The `id_to_index` mapping corresponds to geometry ordering in the `.glb`
scene graph, allowing lovamap_gw to associate metadata with rendered meshes.

---

## Configuration Reference

| Option                  | Type   | Default | Description                                         |
|-------------------------|--------|---------|-----------------------------------------------------|
| `input_dir`             | path   | —       | Source directory for input files                     |
| `output_dir`            | path   | —       | Destination for `.glb` and metadata files            |
| `file_type`             | string | `json`  | Input format: `"json"`, `"dat"`, `"txt"`, `"csv"`, or `"npz"` |
| `file_index`            | int    | `1`     | 1-based index for single-file mode                   |
| `batch_process`         | bool   | `true`  | Process all files (`true`) or one by index (`false`) |
| `show_edge_pores`       | bool   | `true`  | Include pores touching the domain boundary           |
| `save_metadata`         | bool   | `true`  | Generate `_metadata.json` alongside `.glb`           |
| `save_mesh`             | bool   | `true`  | Save the `.glb` file                                 |
| `scrape_subdirectories` | bool   | `false` | Recursively search subdirectories for inputs         |
| `filename`              | string | —       | Override file selection with explicit path            |

### Example config (`configs/mesh_generation.json`)

```json
{
  "input_dir": "/path/to/input",
  "output_dir": "/path/to/output",
  "file_type": "json",
  "batch_process": true,
  "show_edge_pores": false,
  "save_metadata": true,
  "save_mesh": true,
  "scrape_subdirectories": true
}
```

---

## Dependencies

| Package          | Purpose                                |
|------------------|----------------------------------------|
| `trimesh`        | Scene construction, mesh operations    |
| `numpy`          | Array operations, grid generation      |
| `scipy`          | Gaussian smoothing, Delaunay           |
| `scikit-image`   | Marching Cubes algorithm               |
| `open3d`         | Quadric mesh decimation                |
| `pygltflib`      | GLB file handling                      |
| `gltf-pipeline`  | Draco compression (Node.js, via CLI)   |

---

## Error Handling

| Condition                          | Behavior                                     |
|------------------------------------|----------------------------------------------|
| Entity has < 10 voxels            | Skipped with warning                          |
| No valid entities after filtering  | `ValueError` raised                          |
| Mesh simplification fails          | Falls back to unsimplified mesh              |
| Draco compression fails            | Saves uncompressed `.glb` with warning       |
| `.dat`/`.txt`/`.csv` line has non-numeric data | Line skipped silently              |
| No files found in `input_dir`      | Exit with error message                      |
