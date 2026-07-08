# JSON Data Format Reference

Reference documentation for the voxelized segmentation JSON files produced by the LOVAMAP pipeline and consumed by `segmentation2dTo3d` for mesh generation, visualization, and downstream processing workflows.

---

## 1. JSON File Structure

Files may be wrapped in an outer array (`[{ ... }]`) or be a plain object (`{ ... }`). The parser (`load_raw_json_data`) handles both; if the root is an array, the first element is used.

### 1.1 Top-Level Fields

| Field | Type | Description |
|---|---|---|
| `voxel_size` | `float` | Edge length of each cubic voxel in real-world units (e.g. micrometers). |
| `domain_size` | `[float x6]` | Domain bounds: `[x_min, x_max, y_min, y_max, z_min, z_max]`. |
| `voxel_count` | `int` (optional) | Total number of voxels in the domain grid (`nx * ny * nz`). |
| `pores` **or** `bead_data`/`beads` | `dict` | The entity voxel data (see [Section 2](#2-entity-voxel-data)). Determines domain type. |
| `pores_metadata` | `dict` (optional) | Per-entity metadata, keyed by entity ID (see [Section 3](#3-entity-metadata)). |
| `bead_count` | `int` (optional) | Number of particles (present in particle-type files). |
| `bead_voxel_count` | `dict` (optional) | Entity ID to voxel count mapping. |
| `created` | `string` (optional) | ISO-style timestamp of when the file was generated. |
| `data_type` | `string` (optional) | Usually `"labeled"`. |
| `hip_file` | `string` (optional) | Path to the original source file (e.g. `.dat`). |

### 1.2 Domain Type Detection

The presence of specific keys determines the domain type:

- **`"bead_data"` or `"beads"`** in the JSON &rarr; **particle** domain
- **`"pores"`** in the JSON &rarr; **pore** domain

This is handled by `detect_domain_type()` in `core/file_parsing_methods.py`.

### 1.3 Example: Pore-Type JSON

```json
[{
  "voxel_size": 2,
  "domain_size": [0, 630, 0, 630, 0, 504],
  "voxel_count": 24948450,
  "pores": {
    "1-888cad98-51cc-3b8b-bb8c-b88efef38282": [
      [202702, 202730],
      [203015, 203045],
      [203070, 203075],
      [203328, 203360]
    ],
    "2-a1b2c3d4-...": [
      [100, 150],
      [200, 210]
    ]
  },
  "pores_metadata": {
    "1-888cad98-51cc-3b8b-bb8c-b88efef38282": {
      "num": 1,
      "uniqueID": "1-888cad98-...",
      "centerCoord": [308.68, 258.04, 229.98],
      "volume": 46631.58,
      "surfArea": 6139.52,
      "edge": 1,
      "avgDoorDiam": 18.92,
      "largestDoorDiam": 54.49,
      "beadNeighbors": [3, 4, 5]
    }
  }
}]
```

### 1.4 Example: Particle-Type JSON

```json
[{
  "bead_count": 500,
  "bead_voxel_count": {"1": 3421, "2": 2987},
  "created": "2026-01-15 10:30:00",
  "data_type": "labeled",
  "domain_size": [0, 500, 0, 500, 0, 400],
  "hip_file": "/path/to/original.dat",
  "voxel_count": 25000000,
  "voxel_size": 1,
  "bead_data": {
    "1": [[1000, 1050], [1365, 1400], [1680, 1730]],
    "2": [[5000, 5020], [5315, 5340]]
  }
}]
```

---

## 2. Entity Voxel Data

The core data structure maps each entity (pore or particle) to an array of **1D voxel index ranges**.

```
entity_id (string) -> [[start, end], [start, end], ...]
```

### 2.1 Index Convention: 1-Based (MATLAB-Style)

**The ranges in the JSON file use 1-based inclusive indexing** (MATLAB convention).

- `[start, end]` means all voxels from index `start` through `end`, inclusive.
- Both `start` and `end` are 1-based.

**When parsing in Python (0-based), subtract 1:**

```python
for start, end in voxel_ranges:
    start -= 1  # Convert to 0-based
    end -= 1    # Convert to 0-based
    voxel_indices.extend(range(start, end + 1))  # +1 because range() is exclusive
```

This is implemented in `parse_json_file()` at `core/file_parsing_methods.py:185-189`.

### 2.2 Why Ranges Instead of Individual Indices

Each entity may occupy thousands to millions of voxels. Storing each voxel index individually would make files enormous. Instead, **consecutive voxel indices are compressed into ranges**:

- If a particle occupies 1D indices `[100, 101, 102, 103, 150, 151, 152]`, this is stored as `[[100, 103], [150, 152]]` (1-based: `[[101, 104], [151, 153]]`).
- This achieves significant compression because voxels belonging to a single entity tend to cluster spatially, producing long consecutive runs in the 1D index space.

### 2.3 Converting Ranges Back to Full Index Lists

```python
def expand_ranges_to_indices(voxel_ranges):
    """Expand 1-based inclusive ranges to a 0-based index array."""
    indices = []
    for start, end in voxel_ranges:
        start_0 = start - 1  # 1-based -> 0-based
        end_0 = end - 1
        indices.extend(range(start_0, end_0 + 1))
    return np.array(indices, dtype=np.int32)
```

### 2.4 Converting Full Index Lists Back to Ranges

To re-encode indices into ranges (e.g. after filtering), use `convert_indices_to_ranges()` from `core/voxelization_helper_methods.py:518-534`:

```python
def convert_indices_to_ranges(indices):
    """Convert a flat array of 0-based indices into [[start, end], ...] ranges (0-based)."""
    indices = np.sort(np.unique(indices))
    if len(indices) == 0:
        return []
    ranges = []
    start = end = indices[0]
    for i in indices[1:]:
        if i == end + 1:
            end = i
        else:
            ranges.append([start, end])
            start = end = i
    ranges.append([start, end])
    return ranges
```

**Important:** This function produces 0-based ranges. If writing back to JSON (MATLAB convention), add 1 to both `start` and `end`:

```python
ranges_1based = [[s + 1, e + 1] for s, e in convert_indices_to_ranges(indices)]
```

---

## 3. Entity Metadata

Pore-type JSON files may include a `pores_metadata` dictionary keyed by entity ID. Common metadata fields:

| Field | Type | Description |
|---|---|---|
| `num` | `int` | Sequential pore number |
| `uniqueID` | `string` | UUID-style identifier (matches the entity key) |
| `centerCoord` | `[x, y, z]` | Centroid in real-world coordinates |
| `volume` | `float` | Pore volume |
| `surfArea` | `float` | Surface area |
| `charLength` | `float` | Characteristic length |
| `edge` | `int` | `0` = interior pore, `1` = touches domain boundary |
| `avgDoorDiam` | `float` | Average throat/door diameter |
| `largestDoorDiam` | `float` | Largest throat diameter |
| `beadNeighbors` | `[int, ...]` | IDs of neighboring beads/particles |
| `largestSphereDiam` | `float` | Diameter of largest inscribed sphere |
| `peaks` | `[int, ...]` | 1D voxel indices of local maxima in the EDT |
| `skeletonVoxs` | `[int, ...]` | 1D voxel indices of the medial axis skeleton |
| `outerSurfInds` | `[int, ...]` | 1D voxel indices of the outer surface |
| `ellipse_lengths` | `[a, b, c]` | Principal axis lengths of fitted ellipsoid |
| `isotropy` | `float` | Shape isotropy metric |

Not all fields will be present in every file. Particle-type files typically have no per-entity metadata.

---

## 4. The 3D Voxel Grid and 1D Index Mapping

This is the most critical concept for understanding and manipulating the data.

### 4.1 Grid Dimensions

The grid dimensions `(nx, ny, nz)` are computed from `domain_size` and `voxel_size`:

```python
x_min, x_max, y_min, y_max, z_min, z_max = domain_size
nx = int(round((x_max - x_min) / voxel_size))
ny = int(round((y_max - y_min) / voxel_size))
nz = int(round((z_max - z_min) / voxel_size))
```

**Example:** `domain_size = [0, 630, 0, 630, 0, 504]`, `voxel_size = 2`
&rarr; `nx = 315, ny = 315, nz = 252`, total voxels = `315 * 315 * 252 = 24,948,450`

### 4.2 Row-Major (C-Style) 1D Indexing

The 3D grid is flattened into a 1D array using **row-major order** where **X varies fastest**, then Y, then Z:

```
index_1d = x + y * nx + z * (nx * ny)
```

**Traversal order:** The 1D index increments as:
1. X from 0 to nx-1 (fastest axis)
2. Y from 0 to ny-1
3. Z from 0 to nz-1 (slowest axis)

This means consecutive 1D indices represent adjacent voxels along the X axis.

### 4.3 Forward Conversion: 3D to 1D

```python
def coords_to_index(x, y, z, nx, ny):
    """Convert 3D grid coordinates to a 1D index."""
    return x + y * nx + z * (nx * ny)
```

### 4.4 Inverse Conversion: 1D to 3D

```python
def index_to_coords(index, nx, ny):
    """Convert a 1D index to 3D grid coordinates."""
    x = index % nx
    y = (index // nx) % ny
    z = index // (nx * ny)
    return x, y, z
```

**Vectorized (NumPy) version:**

```python
x_indices = indices % nx
y_indices = (indices // nx) % ny
z_indices = indices // (nx * ny)
```

This is used throughout the codebase, e.g. `core/ml_data_generation_methods.py:45-47` and `core/voxelization_helper_methods.py:314-316`.

### 4.5 Worked Example

Grid: `nx=315, ny=315, nz=252`

| 1D Index | x | y | z | Calculation |
|---|---|---|---|---|
| 0 | 0 | 0 | 0 | `0 + 0*315 + 0*99225` |
| 1 | 1 | 0 | 0 | `1 + 0*315 + 0*99225` |
| 314 | 314 | 0 | 0 | `314 + 0*315 + 0*99225` |
| 315 | 0 | 1 | 0 | `0 + 1*315 + 0*99225` |
| 99225 | 0 | 0 | 1 | `0 + 0*315 + 1*99225` |
| 202702 | 202 | 13 | 2 | `202 + 13*315 + 2*99225` |

To verify: `202 + 13*315 + 2*99225 = 202 + 4095 + 198450 = 202747` &mdash; wait, let's recalculate. `nx*ny = 315*315 = 99225`. For index 202702: `z = 202702 // 99225 = 2`, remainder `= 202702 - 2*99225 = 4252`, `y = 4252 // 315 = 13`, `x = 4252 % 315 = 157`. So voxel (157, 13, 2).

### 4.6 Converting 1D Indices to Real-World Coordinates

Grid coordinates map to real-world coordinates via voxel centers:

```python
real_x = x_min + (x_grid + 0.5) * voxel_size
real_y = y_min + (y_grid + 0.5) * voxel_size
real_z = z_min + (z_grid + 0.5) * voxel_size
```

Or equivalently, precompute all voxel centers using `get_centered_grid()` (see [Section 5](#5-voxel-center-grid)) and index into the result:

```python
voxel_centers, grid_size = get_centered_grid(domain_size, voxel_size)
real_coords = voxel_centers[voxel_indices_0based]  # shape (N, 3)
```

---

## 5. Voxel Center Grid

The function `get_centered_grid()` in `core/voxelization_helper_methods.py:10-51` produces a `(N, 3)` array of all voxel center positions in real-world coordinates.

### 5.1 Construction

```python
# Grid dimensions
nx = round((x_max - x_min) / dx)
ny = round((y_max - y_min) / dx)
nz = round((z_max - z_min) / dx)

# Voxel centers along each axis (offset by half a voxel from the boundary)
x = np.linspace(x_min + dx/2, x_max - dx/2, nx)
y = np.linspace(y_min + dx/2, y_max - dx/2, ny)
z = np.linspace(z_min + dx/2, z_max - dx/2, nz)

# 3D meshgrid with ij (matrix) indexing
zzz, yyy, xxx = np.meshgrid(z, y, x, indexing="ij")

# Flatten into (N, 3) in the same order as the 1D index convention
voxel_centers = np.column_stack((xxx.ravel(), yyy.ravel(), zzz.ravel()))
```

### 5.2 Indexing Consistency

The `meshgrid(..., indexing="ij")` with `(z, y, x)` as inputs and then extracting `(xxx, yyy, zzz)` produces the same traversal order as the 1D index formula: X varies fastest, then Y, then Z. This means `voxel_centers[i]` gives the real-world `(x, y, z)` of the voxel with 1D index `i`.

---

## 6. Common Processing Operations

### 6.1 Filtering Entities by Spatial Criteria

To remove entities where any voxel exceeds a threshold on a given axis (e.g. remove particles with any voxel above 600um in Z):

```python
# Given:
#   entities: dict[str, list[list[int]]]  (raw JSON ranges, 1-based)
#   domain_size: [x_min, x_max, y_min, y_max, z_min, z_max]
#   voxel_size: float
#   z_threshold: float (e.g. 600.0)

x_min, x_max, y_min, y_max, z_min, z_max = domain_size
nx = int(round((x_max - x_min) / voxel_size))
ny = int(round((y_max - y_min) / voxel_size))

filtered_entities = {}
for entity_id, ranges in entities.items():
    # Expand ranges to 0-based indices
    indices = []
    for start, end in ranges:
        indices.extend(range(start - 1, end))  # 1-based inclusive -> 0-based
    indices = np.array(indices, dtype=np.int32)

    # Convert to 3D and check Z
    z_grid = indices // (nx * ny)
    z_real = z_min + (z_grid + 0.5) * voxel_size

    if np.all(z_real <= z_threshold):
        filtered_entities[entity_id] = ranges  # Keep original ranges

# filtered_entities now contains only entities fully below the Z threshold
```

### 6.2 Removing Specific Voxels from an Entity

To clip voxels outside a bounding box while keeping the entity:

```python
indices = expand_ranges_to_indices(ranges)  # 0-based

x = indices % nx
y = (indices // nx) % ny
z = indices // (nx * ny)

# Convert to real-world coordinates
real_x = x_min + (x + 0.5) * voxel_size
real_y = y_min + (y + 0.5) * voxel_size
real_z = z_min + (z + 0.5) * voxel_size

# Keep only voxels within bounds
mask = (real_z <= z_max_threshold) & (real_x >= x_min_threshold)
clipped_indices = indices[mask]

# Re-encode as ranges (0-based) and convert back to 1-based for JSON
ranges_0based = convert_indices_to_ranges(clipped_indices)
ranges_1based = [[s + 1, e + 1] for s, e in ranges_0based]
```

### 6.3 Reading and Writing JSON Files

**Reading:**

```python
import json

with open(filepath, 'r') as f:
    data = json.load(f)

# Handle optional list wrapper
if isinstance(data, list):
    data = data[0]

voxel_size = data["voxel_size"]
domain_size = data["domain_size"]
entity_key = "pores" if "pores" in data else "bead_data"
entities = data[entity_key]
```

**Writing:**

```python
import json

output_data = [{
    "voxel_size": voxel_size,
    "domain_size": domain_size,
    "voxel_count": nx * ny * nz,
    entity_key: filtered_entities,
    # Preserve any metadata
    "pores_metadata": data.get("pores_metadata", {})
}]

with open(output_path, 'w') as f:
    json.dump(output_data, f, separators=(",", ":"))  # Compact encoding
```

### 6.4 Computing Entity Statistics from Voxel Data

```python
def get_entity_bounds(indices_0based, nx, ny, voxel_size, domain_size):
    """Get the real-world bounding box of an entity from its 0-based 1D indices."""
    x_min_d, _, y_min_d, _, z_min_d, _ = domain_size

    x = indices_0based % nx
    y = (indices_0based // nx) % ny
    z = indices_0based // (nx * ny)

    real_x = x_min_d + (x + 0.5) * voxel_size
    real_y = y_min_d + (y + 0.5) * voxel_size
    real_z = z_min_d + (z + 0.5) * voxel_size

    return {
        "x_range": (real_x.min(), real_x.max()),
        "y_range": (real_y.min(), real_y.max()),
        "z_range": (real_z.min(), real_z.max()),
        "voxel_count": len(indices_0based),
        "volume": len(indices_0based) * (voxel_size ** 3),
    }
```

---

## 7. Coordinate System Conventions

### 7.1 Microscopy Coordinate System (Input Data)

- **X, Y**: Image plane axes
- **Z**: Optical stacking axis (depth through the sample)
- Z-up convention (Z increases with depth into the stack)
- Units are typically micrometers

### 7.2 glTF / Visualization (Output Meshes)

- glTF 2.0 standard uses Y-up, right-handed coordinates
- The mesh generation pipeline may swap Y and Z axes for conformance
- The `unite_meshes` workflow provides a `flip_yz` option for explicit axis transformation

### 7.3 Real-World vs Grid Space

| Space | X range | Y range | Z range | Units |
|---|---|---|---|---|
| Real-world | `[x_min, x_max]` | `[y_min, y_max]` | `[z_min, z_max]` | micrometers |
| Grid | `[0, nx-1]` | `[0, ny-1]` | `[0, nz-1]` | voxel indices |

**Conversions:**
```python
# Grid -> Real-world (center of voxel)
real = domain_min + (grid_index + 0.5) * voxel_size

# Real-world -> Grid
grid_index = int((real_coord - domain_min) / voxel_size)
```

---

## 8. Quick Reference: Key Formulas

```
Grid dimensions from domain:
    nx = round((x_max - x_min) / voxel_size)
    ny = round((y_max - y_min) / voxel_size)
    nz = round((z_max - z_min) / voxel_size)

3D -> 1D index:
    index = x + y * nx + z * (nx * ny)

1D -> 3D index:
    x = index % nx
    y = (index // nx) % ny
    z = index // (nx * ny)

Grid -> Real-world coordinate:
    real_x = x_min + (x + 0.5) * voxel_size

JSON range (1-based) -> Python indices (0-based):
    python_start = json_start - 1
    python_end = json_end - 1
    indices = range(python_start, python_end + 1)

Python indices (0-based) -> JSON range (1-based):
    json_start = python_start + 1
    json_end = python_end + 1
```

---

## 9. Source Code Reference

| Concept | File | Key Function(s) |
|---|---|---|
| JSON parsing | `core/file_parsing_methods.py` | `parse_json_file()`, `load_raw_json_data()`, `parse_raw_domain_data()` |
| Domain type detection | `core/file_parsing_methods.py` | `detect_domain_type()` |
| 3D grid construction | `core/voxelization_helper_methods.py` | `get_centered_grid()` |
| 1D &harr; 3D conversion | `core/voxelization_helper_methods.py` | `coords_to_index()` |
| Index to range encoding | `core/voxelization_helper_methods.py` | `convert_indices_to_ranges()` |
| Mesh generation | `core/mesh_generation_methods.py` | `generate_mesh_marching_cubes()` |
| JSON output | `core/ml_data_generation_methods.py` | `save_voxelized_data_as_json()` |
| Slice extraction | `core/voxelization_helper_methods.py` | `extract_slices()` |
| Point cloud generation | `core/ml_data_generation_methods.py` | `voxel_grid_to_pointcloud()` |
