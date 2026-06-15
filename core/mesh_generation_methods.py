import time
import pygltflib
import trimesh
import numpy as np
import os
from scipy.spatial import Delaunay
from scipy.interpolate import griddata
from sklearn.metrics import pairwise_distances
from skimage import color
import json

import time
import numpy as np
import open3d as o3d
from skimage.measure import marching_cubes
from scipy.ndimage import gaussian_filter
import subprocess
from trimesh.visual import ColorVisuals
import trimesh.transformations as tf

def generate_mesh_marching_cubes(domain_entities, domain_entity_metadata, voxel_centers, voxel_size, output_path, config, include_color=True, target_faces=10000, color_shuffle_seed=0):
    """
    Generate a .glb mesh using Marching Cubes from skimage, ensuring each particle is independent and has unique colors.

    Parameters:
        domain_data (dict): Dictionary mapping particle labels to their voxel indices.
        voxel_centers (numpy.ndarray): (N, 3) array of all voxel centers.
        voxel_size (float): Size of each voxel.
        grid_size (tuple): (nx, ny, nz) grid dimensions.
        output_path (str): Path to save the .glb file.

    Returns:
        str: Path to the saved .glb file.
    """
    scene = trimesh.Scene()
    num_domain_entities = len(domain_entities)

    # Generate predefined colors
    # predefined_colors = np.random.rand(num_domain_entities, 3)  # RGB only (no alpha)
    predefined_colors = distinguishable_colors(num_domain_entities, 'w', shuffle_seed=color_shuffle_seed)


    for i, (domain_entity_label, voxel_indices) in enumerate(domain_entities.items()):
        if len(voxel_indices) < 10:  # Need at least some voxels for meshing
            print(f"⚠️ Skipping domain_entity {domain_entity_label}: Too few points for meshing.")
            continue

        try:
            # Get actual 3D coordinates of voxel centers
            # coords = voxel_centers[voxel_indices]  # Shape (M, 3)
            coords = voxel_centers[voxel_indices][:, [0, 2, 1]] # Switching y and z axes

            # Compute the grid's min/max bounds
            x_min, y_min, z_min = coords.min(axis=0)
            x_max, y_max, z_max = coords.max(axis=0)

            # Compute grid dimensions based on voxel_size
            gx = int(np.round((x_max - x_min) / voxel_size)) + 3  # Padding +1 on both sides
            gy = int(np.round((y_max - y_min) / voxel_size)) + 3
            gz = int(np.round((z_max - z_min) / voxel_size)) + 3

            # Initialize the voxel grid with extra padding
            voxel_grid = np.zeros((gx, gy, gz), dtype=np.uint8)

            # Convert voxel centers to grid indices with an offset of +1 for padding
            grid_x = np.round((coords[:, 0] - x_min) / voxel_size).astype(int) + 1
            grid_y = np.round((coords[:, 1] - y_min) / voxel_size).astype(int) + 1
            grid_z = np.round((coords[:, 2] - z_min) / voxel_size).astype(int) + 1

            # Fill the grid
            voxel_grid[grid_x, grid_y, grid_z] = 1  # Mark occupied voxels

            # Apply Gaussian Smoothing to Reduce Jagged Edges
            smoothed_voxel_grid = gaussian_filter(voxel_grid.astype(float), sigma=1.2)

            # Apply Marching Cubes
            verts, faces, normals, _ = marching_cubes(smoothed_voxel_grid, level=0.5)

            # Shift vertices back to real-world coordinates
            verts = (verts - 1) * voxel_size + [x_min, y_min, z_min]  # Adjust for padding

            # Decimate
            verts, faces = simplify_mesh(verts, faces, target_faces=target_faces)

            # Convert to Trimesh
            domain_entity_mesh = trimesh.Trimesh(vertices=verts, faces=faces, vertex_normals=normals)

            # Decimate using o3d 
            # domain_entity_mesh = simplify(domain_entity_mesh, target_faces=target_faces)

            # Ensure normals are outward-facing
            domain_entity_mesh.fix_normals()

            if include_color == True:
                # Assign per-vertex color
                color = predefined_colors[i]
                vertex_colors = np.hstack([np.tile(color, (len(verts), 1)), np.ones((len(verts), 1))])  # RGBA
                # vertex_colors = np.tile(color, (len(verts), 1))
                # domain_entity_mesh.visual.vertex_colors = vertex_colors  # Apply colors
                domain_entity_mesh.visual = ColorVisuals(mesh=domain_entity_mesh, vertex_colors=vertex_colors)

            # domain_entity_mesh.vertices = np.array(domain_entity_mesh.vertices, dtype=np.float16)  # Convert to 16-bit
            # domain_entity_mesh.vertex_normals = np.array(domain_entity_mesh.vertex_normals, dtype=np.float16)  # Convert normals

            # Store metadata
            domain_entity_mesh.metadata["domain_entity_id"] = domain_entity_label
            # domain_entity_mesh.metadata["num_voxels"] = len(voxel_indices)

            # Add mesh to scene **as a separate object**
            scene.add_geometry(domain_entity_mesh, node_name=f"{domain_entity_label}")

        except Exception as e:
            print(f"⚠️ Warning: Skipping domain_entity {domain_entity_label} due to meshing error: {e}")
            continue

    if not scene.geometry:
        raise ValueError("No valid domain_entity meshes were generated.")

    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    if (config.get("save_mesh", True)):
        scene.export(output_path, file_type="glb")
        print(f"🎉 GLB file saved: {output_path}")
        # compressed_output_path = output_path.replace(".glb", "_compressed.glb")
        compressed_output_path = output_path
        compress_glb(output_path, compressed_output_path)
    
    if (config.get("save_metadata", True)):
        save_metadata(scene, domain_entity_metadata, output_path=output_path)

    return output_path

def save_metadata(scene, domain_entity_metadata, output_path):
    # Build domain_entity ID list and ID → index mapping
    ids = []
    id_to_index = {}
    entity_metadata = {}  # New

    # Loop through the scene nodes
    for i, node_name in enumerate(scene.graph.nodes_geometry):
        id = node_name  # Node names are set to domain_entity label (as string)
        id_str = str(id)  # JSON keys must be strings
        ids.append(id_str)
        id_to_index[id_str] = i
        
        # If metadata for this ID exists (only pores will have extra metadata)
        if domain_entity_metadata and id in domain_entity_metadata:
            entity_metadata[id_str] = domain_entity_metadata[id]
        else:
            entity_metadata[id_str] = {}  # Empty metadata for particles

    # Final metadata structure
    metadata = {
        "ids": ids,
        "id_to_index": id_to_index,
        "metadata": entity_metadata
    }

    # Optionally: write to .json file alongside .glb
    metadata_path = output_path.replace(".glb", "_metadata.json")
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, separators=(",", ":"))

    print(f"📄 Metadata saved: {metadata_path}")

def _srgb_to_linear(c):
    """sRGB companding → linear, matching the TS srgbToLinear helper."""
    return np.where(c <= 0.04045, c / 12.92, np.power((c + 0.055) / 1.055, 2.4))

def _rgb_to_lab(rgb):
    """Convert an (N, 3) sRGB array in [0,1] to CIE-LAB (D65).

    Uses the same constants as the TS pore-color-testing.ts implementation so
    that both code-paths produce bit-identical LAB values (within float64
    precision).
    """
    lin = _srgb_to_linear(rgb)

    # sRGB → XYZ (D65) — same matrix as the TS version
    X = 0.4124564 * lin[:, 0] + 0.3575761 * lin[:, 1] + 0.1804375 * lin[:, 2]
    Y = 0.2126729 * lin[:, 0] + 0.7151522 * lin[:, 1] + 0.0721750 * lin[:, 2]
    Z = 0.0193339 * lin[:, 0] + 0.1191920 * lin[:, 1] + 0.9503041 * lin[:, 2]

    # D65 reference white
    Xn, Yn, Zn = 0.95047, 1.0, 1.08883

    delta = 6.0 / 29.0
    delta3 = delta ** 3

    def f(t):
        return np.where(t > delta3, np.cbrt(t), t / (3 * delta * delta) + 4.0 / 29.0)

    fx = f(X / Xn)
    fy = f(Y / Yn)
    fz = f(Z / Zn)

    L = 116.0 * fy - 16.0
    a = 500.0 * (fx - fy)
    b = 200.0 * (fy - fz)

    return np.column_stack([L, a, b])

def _mulberry32(seed):
    """Python port of the TS mulberry32 PRNG so the same seed produces the
    same shuffle permutation in both code-paths."""
    state = seed & 0xFFFFFFFF  # uint32

    def next_val():
        nonlocal state
        state = (state + 0x6D2B79F5) & 0xFFFFFFFF
        t = state
        # Math.imul(t ^ (t >>> 15), t | 1)
        t = ((t ^ (t >> 15)) * (t | 1)) & 0xFFFFFFFF
        # t ^= t + Math.imul(t ^ (t >>> 7), t | 61)
        imul = ((t ^ (t >> 7)) * (t | 61)) & 0xFFFFFFFF
        t = (t ^ ((t + imul) & 0xFFFFFFFF)) & 0xFFFFFFFF
        # ((t ^ (t >>> 14)) >>> 0) / 4294967296
        return ((t ^ (t >> 14)) & 0xFFFFFFFF) / 4294967296.0

    return next_val

def distinguishable_colors(n_colors, bg='w', func=None, n_grid=40, L_min=0, L_max=100, gray_tol=0.05, shuffle_seed=None):
    """
    Generate `n_colors` perceptually distinct colors that are not too light or too dark,
    or too gray.

    This is a direct port of the TS distinguishableColors() in lovamap_gw's
    pore-color-testing.ts so that both systems produce identical palettes.

    Parameters:
    - n_colors: The number of distinct colors to generate.
    - bg: The background color. Default is white ('w'), but can be a tuple (r, g, b).
    - func: Optional function for color conversion (default is None).
    - n_grid: Grid size for the color space. Increasing it gives more options.
    - L_min: Minimum lightness threshold to avoid too dark colors. Default 0 (no lower
      filter); use e.g. 15 to exclude near-black colors.
    - L_max: Maximum lightness threshold to avoid too light colors. Default 100 (no upper
      filter); use e.g. 85 to exclude near-white colors. NOTE: values below ~88 will
      exclude pure green/yellow/cyan from the palette and visibly reduce vibrancy.
    - gray_tol: Threshold below which RGB values are considered too similar (i.e. gray).
    - shuffle_seed: Optional int. If provided, the greedy color list is Fisher-Yates
      shuffled with this seed before being returned, so neighboring entity ids get
      non-consecutive palette entries.  Uses the same mulberry32 PRNG as the TS
      implementation for identical results.

    Returns:
    - A numpy array of size (n_colors, 3) representing RGB colors in the range [0, 1].
    """

    # Set default background color (white or custom)
    if bg == 'w':
        bg_rgb = np.array([[1.0, 1.0, 1.0]])
    elif bg == 'k':
        bg_rgb = np.array([[0.0, 0.0, 0.0]])
    elif isinstance(bg, (tuple, list)) and len(bg) == 3:
        bg_rgb = np.array([bg], dtype=float)
    else:
        raise ValueError("Background color must be 'w', 'k', or a tuple of RGB values.")

    if n_colors <= 0 or n_grid < 2:
        return np.empty((0, 3))

    # Build the RGB candidate grid — iterate R slowest, G middle, B fastest
    # to match the TS nested-loop order exactly (critical for tie-breaking in
    # the greedy selection).
    step = 1.0 / (n_grid - 1)
    rgbs = []
    for i in range(n_grid):
        r = i * step
        for j in range(n_grid):
            g = j * step
            for k in range(n_grid):
                b = k * step

                gray_score = max(abs(r - g), abs(r - b), abs(g - b))
                if gray_score < gray_tol:
                    continue

                rgbs.append((r, g, b))

    rgb = np.array(rgbs, dtype=float)

    # Convert to LAB using the same manual conversion as the TS code
    if func is None:
        lab = _rgb_to_lab(rgb)
        bg_lab = _rgb_to_lab(bg_rgb)
    else:
        lab = func(rgb)
        bg_lab = func(bg_rgb)

    # Filter by lightness (strict inequality, matching TS: <= L_min or >= L_max → skip)
    L = lab[:, 0]
    keep = (L > L_min) & (L < L_max)
    rgb = rgb[keep]
    lab = lab[keep]

    if len(rgb) < n_colors:
        raise ValueError(
            f"Not enough distinct colors available within the specified "
            f"lightness range (L: {L_min}-{L_max})."
        )

    # Seed min_dist from background (squared Euclidean in LAB, matching TS)
    diff = lab - bg_lab  # broadcast (N,3) - (1,3)
    min_dist = np.sum(diff * diff, axis=1)

    # Greedy farthest-point selection
    count = min(n_colors, len(rgb))
    selected = np.empty((count, 3), dtype=float)
    for n in range(count):
        idx = int(np.argmax(min_dist))
        selected[n] = rgb[idx]

        d = lab - lab[idx]
        new_dist = np.sum(d * d, axis=1)
        min_dist = np.minimum(min_dist, new_dist)

    # Fisher-Yates shuffle with mulberry32 PRNG (matches TS shuffleInPlace)
    if shuffle_seed is not None:
        rand = _mulberry32(shuffle_seed)
        arr = list(range(len(selected)))
        for i in range(len(arr) - 1, 0, -1):
            j = int(rand() * (i + 1))
            arr[i], arr[j] = arr[j], arr[i]
        selected = selected[arr]

    return selected


def color_scene_unique(scene: trimesh.Scene, colors=None, alpha: float = 1.0, verbose: bool = True) -> trimesh.Scene:
    """Assign unique colors to each geometry in a trimesh.Scene.

    Args:
        scene: trimesh.Scene containing geometry items (Trimesh objects).
        colors: Optional iterable of RGB colors in [0,1] shape (n_items, 3). If None will generate distinguishable colors.
        alpha: Alpha value in [0,1] to apply to all vertices.
        verbose: Print progress messages.

    Returns:
        The same scene with colored geometries (colors applied per-vertex as RGBA).
    """
    n = len(scene.geometry)
    if n == 0:
        return scene

    if colors is None:
        colors = distinguishable_colors(n, 'w')

    colors = list(colors)
    # Ensure we have at least n colors
    if len(colors) < n:
        # Repeat colors if needed
        times = int(np.ceil(n / len(colors)))
        colors = (colors * times)[:n]

    # Apply colors in scene order
    for idx, (name, geom) in enumerate(list(scene.geometry.items())):
        try:
            c = np.array(colors[idx])
            # per-vertex RGBA
            vertex_colors = np.hstack([np.tile(c, (len(geom.vertices), 1)), np.ones((len(geom.vertices), 1)) * float(alpha)])
            geom.visual = ColorVisuals(mesh=geom, vertex_colors=vertex_colors)
            if verbose:
                print(f"🎨 Applied color to {name}: {c.tolist()} (alpha={alpha})")
        except Exception as e:
            if verbose:
                print(f"⚠️ Failed to color {name}: {e}")
            continue

    return scene


def apply_material_color(mesh: trimesh.Trimesh, color, alpha: float = 1.0, verbose: bool = False) -> trimesh.Trimesh:
    """(Deprecated fast path) Apply a material-based color to a single Trimesh.

    NOTE: Trimesh GLB exporter may not serialize SimpleMaterial consistently across
    environments. Prefer `apply_vertex_color` for reliable export.
    """
    try:
        rgb = list(color)[:3]
        material = trimesh.visual.material.SimpleMaterial(diffuse=rgb)
        mesh.visual = trimesh.visual.TextureVisuals(material=material)
        if verbose:
            print(f"🎨 Applied material color {rgb} to mesh")
    except Exception as e:
        if verbose:
            print(f"⚠️ Failed to apply material color: {e}")
    return mesh


def apply_vertex_color(mesh: trimesh.Trimesh, color, alpha: float = 1.0, verbose: bool = False) -> trimesh.Trimesh:
    """Apply a per-vertex uniform color to a single mesh (reliable export).

    This is the recommended "fast" path: we color each mesh before combining
    so we avoid coloring the whole Scene in a second pass.
    """
    try:
        c = np.array(list(color)[:3])
        vertex_colors = np.hstack([np.tile(c, (len(mesh.vertices), 1)), np.ones((len(mesh.vertices), 1)) * float(alpha)])
        mesh.visual = ColorVisuals(mesh=mesh, vertex_colors=vertex_colors)
        if verbose:
            print(f"🎨 Applied vertex color to mesh (n_pts={len(mesh.vertices)}) color={c.tolist()}")
    except Exception as e:
        if verbose:
            print(f"⚠️ Failed to apply vertex color: {e}")
    return mesh


def is_not_gray(rgb, tol=0.05):
    r, g, b = rgb
    return not (abs(r - g) < tol and abs(r - b) < tol and abs(g - b) < tol)

def simplify_mesh(vertices, faces, target_faces=10000):
    """
    Simplifies a mesh using Open3D's quadric decimation.

    Parameters:
        vertices (np.ndarray): (N, 3) array of mesh vertices.
        faces (np.ndarray): (M, 3) array of mesh faces.
        target_faces (int): Approximate target number of faces.

    Returns:
        tuple: (simplified_vertices, simplified_faces) as NumPy arrays.
    """
    try:
        # Convert to Open3D TriangleMesh
        o3d_mesh = o3d.geometry.TriangleMesh()
        o3d_mesh.vertices = o3d.utility.Vector3dVector(vertices)
        o3d_mesh.triangles = o3d.utility.Vector3iVector(faces)

        # Apply Quadric Decimation
        o3d_mesh = o3d_mesh.simplify_quadric_decimation(target_faces)

        # Extract simplified vertices and faces
        simplified_vertices = np.asarray(o3d_mesh.vertices)
        simplified_faces = np.asarray(o3d_mesh.triangles)

        # print(f"✅ Mesh Simplified: {len(faces)} → {len(simplified_faces)} faces")
        return simplified_vertices, simplified_faces

    except Exception as e:
        # print(f"❌ Mesh simplification failed: {e}")
        return vertices, faces  # Return original if simplification fails

def simplify(mesh, target_faces=8000):
    """
    Simplify a mesh using Open3D's quadric decimation.

    Parameters:
        mesh (trimesh.Trimesh): Input mesh.
        target_faces (int): Approximate target number of faces.

    Returns:
        trimesh.Trimesh: Simplified mesh.
    """
    try:
        # Convert Trimesh to Open3D mesh
        o3d_mesh = o3d.geometry.TriangleMesh()
        o3d_mesh.vertices = o3d.utility.Vector3dVector(mesh.vertices)
        o3d_mesh.triangles = o3d.utility.Vector3iVector(mesh.faces)

        # Apply quadric decimation
        o3d_mesh = o3d_mesh.simplify_quadric_decimation(target_faces)

        # Convert back to Trimesh
        simplified_mesh = trimesh.Trimesh(
            vertices=np.asarray(o3d_mesh.vertices),
            faces=np.asarray(o3d_mesh.triangles)
        )

        # print(f"✅ Simplified mesh: {len(mesh.faces)} → {len(simplified_mesh.faces)} faces")
        return simplified_mesh
    except Exception as e:
        # print(f"❌ Mesh simplification failed: {e}")
        return mesh  # Return original mesh if simplification fails

def compress_glb(input_glb, output_glb, compression_level=10):
    """Compress .glb file using Draco with gltf-pipeline (Node.js).

    Returns:
        bool: True on success, False on failure.
    """
    try:
        subprocess.run(
            ["gltf-pipeline", "-i", input_glb, "-o", output_glb, "--draco.compressMeshes", f"--draco.compressionLevel={compression_level}"],
            check=True
        )
        print(f"✅ Compressed GLB saved: {output_glb}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Compression failed: {e}")
        return False


def generate_mesh_from_spheres(centers, radii, output_path, config, subdivisions=3, color_shuffle_seed=0):
    """
    Generate a .glb mesh directly from sphere centroids and radii using icospheres.

    Parameters:
        centers (numpy.ndarray): (N, 3) array of sphere center coordinates.
        radii (numpy.ndarray): (N,) array of sphere radii.
        output_path (str): Path to save the .glb file.
        config (dict): Configuration dictionary (uses save_mesh, save_metadata keys).
        subdivisions (int): Number of icosphere subdivisions (default 3).
        color_shuffle_seed (int): Seed for color shuffling.

    Returns:
        str: Path to the saved .glb file.
    """
    scene = trimesh.Scene()
    num_spheres = len(radii)

    predefined_colors = distinguishable_colors(num_spheres, 'w', shuffle_seed=color_shuffle_seed)

    for i in range(num_spheres):
        sphere_mesh = trimesh.creation.icosphere(subdivisions=subdivisions, radius=float(radii[i]))
        sphere_mesh.apply_translation(centers[i])

        # Assign per-vertex color
        c = predefined_colors[i]
        vertex_colors = np.hstack([
            np.tile(c, (len(sphere_mesh.vertices), 1)),
            np.ones((len(sphere_mesh.vertices), 1))
        ])
        sphere_mesh.visual = ColorVisuals(mesh=sphere_mesh, vertex_colors=vertex_colors)

        # Use 1-indexed string ID to match marching cubes convention
        label = str(i + 1)
        scene.add_geometry(sphere_mesh, node_name=label)

    if not scene.geometry:
        raise ValueError("No sphere meshes were generated.")

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    if config.get("save_mesh", True):
        scene.export(output_path, file_type="glb")
        print(f"🎉 GLB file saved: {output_path}")
        compress_glb(output_path, output_path)

    if config.get("save_metadata", True):
        save_metadata(scene, None, output_path=output_path)

    return output_path


def generate_glb_file_poisson(particles, voxel_centers, voxel_size, output_path, method="poisson"):
    """
    Generates a .glb file using Poisson or Ball Pivoting surface reconstruction.

    Parameters:
        particles (dict): Dictionary where keys are particle labels, values are 1D voxel indices.
        voxel_centers (numpy.ndarray): Array of all voxel centers.
        voxel_size (float): Voxel size for scaling.
        output_path (str): Path to save the .glb file.
        method (str): "poisson" (default) or "bpa" (Ball Pivoting Algorithm).

    Returns:
        str: Path to the saved .glb file.
    """
    meshing_start_time = time.time()
    scene = trimesh.Scene()
    num_particles = len(particles)
    predefined_colors = np.random.rand(num_particles, 4)  # Precompute colors once (RGBA)

    for i, (particle_label, voxel_indices) in enumerate(particles.items()):
        coords = voxel_centers[voxel_indices]
        if len(coords) < 50:
            print(f"⚠️ Skipping particle {particle_label}: Too few points ({len(coords)}).")
            continue  # Skip if not enough points

        # ✅ Check bounding box aspect ratio (skip extreme cases)
        bbox = np.ptp(coords, axis=0)  # Get size in x, y, z
        aspect_ratios = bbox / max(bbox)
        if any(aspect_ratios < 0.05):  # If one dimension is too small, skip
            print(f"⚠️ Skipping particle {particle_label}: Extreme aspect ratio.")
            continue

        # ✅ Convert to Open3D PointCloud
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(coords)

        # ✅ Remove outliers to prevent Poisson failures
        pcd, inlier_indices = pcd.remove_radius_outlier(nb_points=15, radius=voxel_size * 2)
        if len(inlier_indices) < 50:
            print(f"⚠️ Skipping particle {particle_label}: Too many outliers removed.")
            continue

        # ✅ Improve Normal Estimation using PCA
        pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=voxel_size * 2, max_nn=30))
        pcd.orient_normals_towards_camera_location(np.array([0, 0, 0]))  # Orient normals outward

        try:
            if method == "poisson":
                # 🔥 Poisson: Lower depth to prevent crashes
                mesh, _ = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd, depth=5)
            else:
                # 🔥 Ball Pivoting: Use multiple pivoting radii
                radii = [voxel_size, voxel_size * 1.5, voxel_size * 2]
                mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(pcd, o3d.utility.DoubleVector(radii))
            
            # ✅ Check mesh integrity before adding to scene
            if len(mesh.triangles) == 0:
                print(f"⚠️ Skipping particle {particle_label}: Poisson failed to generate triangles.")
                continue
            
            # Convert to Trimesh for GLB export
            mesh_trimesh = trimesh.Trimesh(vertices=np.asarray(mesh.vertices), faces=np.asarray(mesh.triangles))

            # ✅ Apply smoothing
            mesh_trimesh = trimesh.smoothing.filter_taubin(mesh_trimesh, lamb=0.5, nu=-0.53, iterations=5)

            # Assign color
            color = predefined_colors[i]
            material = trimesh.visual.material.SimpleMaterial(diffuse=color[:3])
            mesh_trimesh.visual = trimesh.visual.TextureVisuals(material=material)

            # Store metadata
            mesh_trimesh.metadata["particle_id"] = particle_label
            mesh_trimesh.metadata["num_voxels"] = len(voxel_indices)

            # Add to scene
            scene.add_geometry(mesh_trimesh, node_name=f"particle_{particle_label}")

        except Exception as e:
            print(f"❌ Error processing particle {particle_label}: {e}. Skipping.")

    if not scene.geometry:
        raise ValueError("No voxel data available to create .glb file.")

    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Export as .glb
    scene.export(output_path)
    print(f"✅ GLB file saved: {output_path}")

    meshing_duration = time.time() - meshing_start_time
    print(f"⏳ Time taken for GLB generation: {meshing_duration:.2f} seconds")

    return output_path

def generate_glb_file_delauney(particles, voxel_centers, voxel_size, output_path):
    """
    Generates a .glb file with separate Delaunay meshes per particle and assigns materials.

    Parameters:
        particles (dict): Dictionary where keys are particle labels, values are 1D voxel indices.
        voxel_centers (numpy.ndarray): Array of all voxel centers.
        voxel_size (float): Voxel size for scaling.
        output_path (str): Path to save the .glb file.

    Returns:
        str: Path to the saved .glb file.
    """

    meshing_start_time = time.time()
    scene = trimesh.Scene()

    num_particles = len(particles)
    predefined_colors = np.random.rand(num_particles, 4)  # Precompute colors once (RGBA)

    for i, (particle_label, voxel_indices) in enumerate(particles.items()):
        coords = voxel_centers[voxel_indices]

        if len(coords) < 4:  # Need at least 4 points for a 3D triangulation
            continue

        # Generate Delaunay triangulation instead of a convex hull
        delaunay = Delaunay(coords)
        particle_mesh = trimesh.Trimesh(vertices=coords, faces=delaunay.simplices)

        # Assign precomputed color as material
        color = predefined_colors[i][:3]  # RGB only
        material = trimesh.visual.material.SimpleMaterial(
            diffuse=color,  
            ambient=[0.1, 0.1, 0.1],  # Darker ambient for more shadow contrast
            specular=[0.9, 0.9, 0.9],  # Strong specular highlights
            shininess=120.0  # Increase shininess for better light contrast
        )

        # Apply material to the mesh
        particle_mesh.visual = trimesh.visual.TextureVisuals(material=material)

        # Embed metadata directly into the mesh
        particle_mesh.metadata["particle_id"] = particle_label
        particle_mesh.metadata["num_voxels"] = len(voxel_indices)

        # Add mesh to scene
        scene.add_geometry(particle_mesh, node_name=f"particle_{particle_label}")

    if not scene.geometry:
        raise ValueError("No voxel data available to create .glb file.")

    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Export as .glb with materials
    scene.export(output_path)
    print(f"✅ GLB file saved: {output_path}")

    meshing_duration = time.time() - meshing_start_time
    print(f"⏳ Time taken for GLB generation: {meshing_duration:.2f} seconds")

    return output_path

def generate_glb_file_convex_hull(particles, voxel_centers, voxel_size, output_path):
    """
    Generates a .glb file with separate convex hull meshes per particle and assigns materials.

    Parameters:
        particles (dict): Dictionary where keys are particle labels, values are 1D voxel indices.
        voxel_centers (numpy.ndarray): Array of all voxel centers.
        voxel_size (float): Voxel size for scaling.
        output_path (str): Path to save the .glb file.

    Returns:
        str: Path to the saved .glb file.
    """

    meshing_start_time = time.time()
    scene = trimesh.Scene()

    num_particles = len(particles)
    predefined_colors = np.random.rand(num_particles, 4)  # Precompute colors once (RGBA)

    materials = []  # Store materials for reuse

    for i, (particle_label, voxel_indices) in enumerate(particles.items()):
        coords = voxel_centers[voxel_indices]

        if len(coords) == 0:
            continue

        # Generate convex hull
        particle_hull = trimesh.Trimesh(vertices=coords).convex_hull

        # Assign precomputed color as material
        color = predefined_colors[i]
        # material = trimesh.visual.material.SimpleMaterial(
        #     diffuse=color[:3],  # Use only RGB, ignore Alpha
        #     ambient=[0.2, 0.2, 0.2],
        #     specular=[0.5, 0.5, 0.5],
        #     shininess=90.0
        # )
        # material = trimesh.visual.material.SimpleMaterial(
        #     diffuse=color[:3],  # Use RGB
        #     ambient=[0.1, 0.1, 0.1],  # Darker ambient for more shadow contrast
        #     specular=[1.0, 1.0, 1.0],  # Strong specular highlights
        #     shininess=120.0  # Increase shininess for better light contrast
        # )
        material = trimesh.visual.material.SimpleMaterial(
            diffuse=color[:3],  
            ambient=[0.1, 0.1, 0.1],  # Darker ambient for more shadow contrast
            specular=[0.9, 0.9, 0.9],  # Strong specular highlights
            shininess=120.0  # Increase shininess for better light contrast
        )
        materials.append(material)

        # Apply material to the mesh
        particle_hull.visual = trimesh.visual.TextureVisuals(material=material)

        # Embed metadata directly into the mesh
        particle_hull.metadata["particle_id"] = particle_label
        particle_hull.metadata["num_voxels"] = len(voxel_indices)

        # Add mesh to scene
        scene.add_geometry(particle_hull, node_name=f"particle_{particle_label}")

    if not scene.geometry:
        raise ValueError("No voxel data available to create .glb file.")

    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Export as .glb with materials
    scene.export(output_path)
    print(f"GLB file saved: {output_path}")

    meshing_duration = time.time() - meshing_start_time
    print(f"Time taken for GLB generation: {meshing_duration:.2f} seconds")

    return output_path
