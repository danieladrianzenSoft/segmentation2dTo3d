import time
import pygltflib
import trimesh
import numpy as np
import os
from scipy.spatial import Delaunay
from scipy.interpolate import griddata
from sklearn.metrics import pairwise_distances
from skimage import color

import time
import numpy as np
import open3d as o3d
from skimage.measure import marching_cubes
from scipy.ndimage import gaussian_filter
import subprocess

def generate_mesh_marching_cubes(particles, voxel_centers, voxel_size, output_path, include_color=True, target_faces=10000):
    """
    Generate a .glb mesh using Marching Cubes from skimage, ensuring each particle is independent and has unique colors.

    Parameters:
        particles (dict): Dictionary mapping particle labels to their voxel indices.
        voxel_centers (numpy.ndarray): (N, 3) array of all voxel centers.
        voxel_size (float): Size of each voxel.
        grid_size (tuple): (nx, ny, nz) grid dimensions.
        output_path (str): Path to save the .glb file.

    Returns:
        str: Path to the saved .glb file.
    """
    scene = trimesh.Scene()
    num_particles = len(particles)

    # Generate predefined colors
    # predefined_colors = np.random.rand(num_particles, 3)  # RGB only (no alpha)
    predefined_colors = distinguishable_colors(num_particles, 'w')

    for i, (particle_label, voxel_indices) in enumerate(particles.items()):
        if len(voxel_indices) < 10:  # Need at least some voxels for meshing
            print(f"⚠️ Skipping particle {particle_label}: Too few points for meshing.")
            continue

        try:
            # Get actual 3D coordinates of voxel centers
            coords = voxel_centers[voxel_indices]  # Shape (M, 3)

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

            # 🔥 **Apply Gaussian Smoothing to Reduce Jagged Edges**
            smoothed_voxel_grid = gaussian_filter(voxel_grid.astype(float), sigma=1.2)

            # Apply Marching Cubes
            verts, faces, normals, _ = marching_cubes(smoothed_voxel_grid, level=0.5)

            # Shift vertices back to real-world coordinates
            verts = (verts - 1) * voxel_size + [x_min, y_min, z_min]  # Adjust for padding

            # Decimate
            verts, faces = simplify_mesh(verts, faces, target_faces=target_faces)

            # Convert to Trimesh
            particle_mesh = trimesh.Trimesh(vertices=verts, faces=faces, vertex_normals=normals)

            # Decimate using o3d 
            # particle_mesh = simplify(particle_mesh, target_faces=target_faces)

            # Ensure normals are outward-facing
            particle_mesh.fix_normals()

            if include_color == True:
                # Assign per-vertex color
                color = predefined_colors[i]
                vertex_colors = np.hstack([np.tile(color, (len(verts), 1)), np.ones((len(verts), 1))])  # RGBA
                # vertex_colors = np.tile(color, (len(verts), 1))
                particle_mesh.visual.vertex_colors = vertex_colors  # Apply colors

            # particle_mesh.vertices = np.array(particle_mesh.vertices, dtype=np.float16)  # Convert to 16-bit
            # particle_mesh.vertex_normals = np.array(particle_mesh.vertex_normals, dtype=np.float16)  # Convert normals

            # Store metadata
            particle_mesh.metadata["particle_id"] = particle_label
            # particle_mesh.metadata["num_voxels"] = len(voxel_indices)

            # Add mesh to scene **as a separate object**
            scene.add_geometry(particle_mesh, node_name=f"{particle_label}")

        except Exception as e:
            print(f"⚠️ Warning: Skipping particle {particle_label} due to meshing error: {e}")
            continue

    if not scene.geometry:
        raise ValueError("No valid particle meshes were generated.")

    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Export scene as .glb
    # scene.export(output_path)
    # z = trimesh.util.compress(e)

    scene.export(output_path, file_type="glb")

    print(f"🎉 GLB file saved: {output_path}")

    # compressed_output_path = output_path.replace(".glb", "_compressed.glb")
    compressed_output_path = output_path

    compress_glb(output_path, compressed_output_path)

    # # **Generate a properly named Draco-compressed file path**
    # output_dir = os.path.dirname(output_path)  # Extract directory
    # output_filename = "draco_" + os.path.basename(output_path)  # Add prefix to filename
    # output_path_compressed = os.path.join(output_dir, output_filename)  # Full path

    # # **Apply Draco compression**
    # apply_draco_compression(output_path, output_path_compressed)

    # print(f"🎉 Draco-compressed GLB file saved: {output_path_compressed}")

    return output_path

def distinguishable_colors(n_colors, bg='w', func=None, n_grid=40, L_min=15, L_max=85):
    """
    Generate `n_colors` perceptually distinct colors that are not too light or too dark.

    Parameters:
    - n_colors: The number of distinct colors to generate.
    - bg: The background color. Default is white ('w'), but can be a tuple (r, g, b).
    - func: Optional function for color conversion (default is None).
    - n_grid: Grid size for the color space. Increasing it gives more options.
    - L_min: Minimum lightness threshold to avoid too dark colors (default 20).
    - L_max: Maximum lightness threshold to avoid too light colors (default 80).

    Returns:
    - A numpy array of size (n_colors, 3) representing RGB colors in the range [0, 1].
    """
    
    # Set default background color (white or custom)
    if bg == 'w':
        bg_rgb = np.array([1, 1, 1])  # white background
    elif bg == 'b':
        bg_rgb = np.array([0, 0, 0])  # black background
    elif isinstance(bg, tuple) and len(bg) == 3:
        bg_rgb = np.array(bg)  # custom background
    else:
        raise ValueError("Background color must be 'w' or a tuple of RGB values.")
    
    # Generate a large grid of RGB colors in the [0, 1] range
    x = np.linspace(0, 1, n_grid)
    R, G, B = np.meshgrid(x, x, x)
    rgb = np.vstack([R.flatten(), G.flatten(), B.flatten()]).T
    
    # Normalize to [0, 1] if RGB is in [0, 255] range
    rgb_normalized = rgb  # Normalize to [0, 1]
    # rgb_normalized = rgb

    # Convert RGB to LAB color space for perceptual distinctness
    if func is None:
        lab = color.rgb2lab(rgb_normalized)  # Convert RGB to LAB using default function
        bg_lab = color.rgb2lab(bg_rgb.reshape(1, 1, 3)).reshape(1, 3)  # Normalize background
    else:
        lab = func(rgb_normalized)
        bg_lab = func(bg_rgb)
    
    # Filter out colors that are too light or too dark based on the L value
    L = lab[:, 0]  # L value represents lightness
    mask = (L > L_min) & (L < L_max)
    lab_filtered = lab[mask]
    rgb_filtered = rgb_normalized[mask]
    
    if rgb_filtered.shape[0] < n_colors:
        raise ValueError(f"Not enough distinct colors available within the specified lightness range (L: {L_min}-{L_max}).")
    
    # Initialize color set
    selected_colors = []

    # Greedy selection of the most distinct colors from the grid
    for _ in range(n_colors):
        # Calculate pairwise distances from all colors in the lab space
        if selected_colors:
            selected_lab = color.rgb2lab(np.array(selected_colors))  # Normalize to [0, 1]
            distances = pairwise_distances(lab_filtered, selected_lab)  # distance from previously chosen colors
        else:
            # First iteration: distance from background
            distances = pairwise_distances(lab_filtered, bg_lab)

        # Find the farthest color
        min_dist = distances.min(axis=1)
        farthest_color_index = np.argmax(min_dist)
        
        # Add the farthest color to the selected set
        selected_colors.append(rgb_filtered[farthest_color_index])

    # Convert to [0, 1] range (if your system expects it) or [0, 255] as needed
    selected_colors_rgb_float = np.array(selected_colors).clip(0, 1)  # Ensure the values are within the [0, 1] range

    return selected_colors_rgb_float  # Return in [0, 1] range if needed for further processing

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
    """Compress .glb file using Draco with gltf-pipeline (Node.js)."""
    try:
        subprocess.run(
            ["gltf-pipeline", "-i", input_glb, "-o", output_glb, "--draco.compressMeshes", f"--draco.compressionLevel={compression_level}"],
            check=True
        )
        print(f"✅ Compressed GLB saved: {output_glb}")
    except subprocess.CalledProcessError as e:
        print(f"❌ Compression failed: {e}")


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
