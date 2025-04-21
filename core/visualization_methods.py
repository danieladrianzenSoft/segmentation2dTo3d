import json
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import pyvista as pv
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, to_hex
import time
from typing import Literal
from pathlib import Path
from plyfile import PlyData
import os
import matplotlib.pyplot as plt
# from mpl_toolkits.mplot3d import Axes3D
import cv2
from core.grid_optimization_methods import extract_surface_voxels

def plot_spheres_plotly(centers, radii):
    """
    Create an interactive 3D plot of spheres using Plotly.

    Parameters:
        centers (numpy.ndarray): Array of x, y, z coordinates of sphere centers.
        radii (numpy.ndarray): Array of sphere radii.
    """
    fig = go.Figure()

    for (x, y, z), r in zip(centers, radii):
        # Create sphere data
        u = np.linspace(0, 2 * np.pi, 30)
        v = np.linspace(0, np.pi, 15)
        xs = r * np.outer(np.cos(u), np.sin(v)) + x
        ys = r * np.outer(np.sin(u), np.sin(v)) + y
        zs = r * np.outer(np.ones_like(u), np.cos(v)) + z

        # Add the sphere to the plot
        fig.add_trace(
            go.Surface(
                x=xs,
                y=ys,
                z=zs,
                opacity=1,
                showscale=False,
                colorscale="Blues"
            )
        )

    # Configure layout
    fig.update_layout(
        scene=dict(
            xaxis_title='X',
            yaxis_title='Y',
            zaxis_title='Z',
            aspectmode='data',  # Ensures equal scaling for all axes
        ),
        title='3D Visualization of Granular Scaffold',
    )

    fig.show()

def plot_voxelized_domain(
    domain_data,
    voxel_centers,
    grid_size,
    voxel_size,
    domain_size,
    config,
    cmap,
    label="Particle",
):
    surface_only = config.get("surface_only", False)
    slices = config.get("slices", None)
    slice_coordinates = config.get("slice_coordinates", None)
    axis = config.get("axis", 'z')
    show_legend = config.get("show_legend", False)

    if surface_only:
        print(f"Using surface-only downsampling for {label.lower()}s...")
        domain_data = extract_surface_voxels(domain_data, grid_size)

    total_voxels = sum(len(v) for v in domain_data.values())
    print(f"✅ Plotting {len(domain_data)} {label.lower()}s with {total_voxels} total voxels.")

    # print(len(domain_data))
    # cmap = create_colormap(domain_data, make_unique=False),
    # print(len(cmap))

    plot_with_pyvista_polydata(
        domain_data=domain_data,
        voxel_centers=voxel_centers,
        voxel_size=voxel_size,
        domain_size=domain_size,
        grid_size=grid_size,
        slices=slices,
        slice_positions=slice_coordinates,
        axis=axis,
        cmap=cmap,
        show_legend=show_legend
    )

def plot_particles(
        particles, 
        voxel_centers,
        grid_size,
        voxel_size, 
        domain_size, 
        surface_only=True,
        slices=None, 
        slice_coordinates=None, 
        axis='z', 
        plot_method: Literal['px', 'go_iterative', 'go_batch', 'pv_polydata'] = 'pv_polydata',
        cmap=None,
        show_legend=False
    ):
    """
    Plot particles in 3D using Plotly with batch plotting.

    Parameters:
        particles (dict): Dictionary mapping particle labels to their voxel indices.
        voxel_size (float): The size of each voxel.
        domain_size (tuple): Shape of the 3D grid in real space (x, y, z dimensions).
        downsample_factor (float): Fraction of voxels to retain for each particle (0 < factor <= 1).
        surface_only (bool): If True, only plot surface voxels of particles.
        max_particles (int or None): Number of particles to plot. If None, plot all particles.
        plot_method (str): Method to use for plotting ('px', 'go_iterative', 'go_batch', 'pv_polydata', 'pv_multiblock').
    """

    print(f"Domain size (real space): {domain_size}")
    print(f"Domain size (grid space): {grid_size}")
    # print(f"Voxel centers shape: {voxel_centers.shape}")
    print(f"Sample voxel centers:\n{voxel_centers[:500]}")

    print(f"Plotting {len(particles)} particle(s).")
    total_particle_voxels = sum(len(voxels) for voxels in particles.values())
    print(f"Total number of particle voxels: {total_particle_voxels}")

    if surface_only:
        print("Using surface-only downsampling...")
        particles = extract_surface_voxels(particles, grid_size)

    # Calculate total number of surface voxels
    total_surface_voxels = sum(len(voxels) for voxels in particles.values())
    print(f"Total number of particle surface voxels: {total_surface_voxels}")

    # Call the appropriate plotting method
    if plot_method == 'px':
        plot_with_plotly_express(particles, voxel_centers, voxel_size, domain_size)
    elif plot_method == 'go_iterative':
        plot_with_plotly_iterative(particles, voxel_centers, voxel_size, domain_size)
    elif plot_method == 'go_batch':
        plot_with_plotly_batch(particles, voxel_centers, voxel_size, domain_size)
    elif plot_method == 'pv_polydata':
        plot_with_pyvista_polydata(particles, voxel_centers, voxel_size, domain_size, grid_size, slices, slice_coordinates, axis, cmap=cmap, show_legend=show_legend)
    else:
        print(f"Plotting method {plot_method} is not valid.")

def plot_with_plotly_express(particles, voxel_centers, voxel_size, domain_size):
    # Start timing for data preparation
    prep_start_time = time.time()

    unique_colors = plt.cm.tab20(np.linspace(0, 1, len(particles)))  # Tab20 colormap
    unique_colors = [tuple(c[:3]) for c in unique_colors]  # Convert RGBA to RGB

    all_coords = []
    for i, (particle_label, voxel_indices) in enumerate(particles.items()):
        coords = voxel_centers[voxel_indices]
        for coord in coords:
            all_coords.append({
                'x': coord[0],
                'y': coord[1],
                'z': coord[2],
                'Particle': f"Particle {particle_label}",
                'Color': unique_colors[i]  # Use the unique color for this particle
            })

    # Convert the data to a DataFrame
    df = pd.DataFrame(all_coords)

    # End timing for data preparation
    prep_end_time = time.time()
    prep_duration = prep_end_time - prep_start_time
    print(f"Time taken for method-specific data preparation: {prep_duration:.2f} seconds")
    print(f"Plotting {len(all_coords)} voxels")
    
    # Start timing for plotting
    plot_start_time = time.time()

    # Use plotly.express to create a scatter plot
    fig = px.scatter_3d(
        df,
        x='x',
        y='y',
        z='z',
        color='Particle',  # Color by particle label
        title="3D Particle Visualization (Plotly Express)",
    )

    # Customize marker size and axes
    fig.update_traces(marker=dict(size=voxel_size))
    fig.update_layout(
        scene=dict(
            xaxis=dict(title="X", range=[0, domain_size[0]]),
            yaxis=dict(title="Y", range=[0, domain_size[1]]),
            zaxis=dict(title="Z", range=[0, domain_size[2]]),
        )
    )

    # Show the plot
    fig.show()

    plot_end_time = time.time()
    plot_duration = plot_end_time - plot_start_time
    print(f"Time taken for plotting: {plot_duration:.2f} seconds")

def plot_with_plotly_iterative(particles, voxel_centers, voxel_size, domain_size):
    # Start timing for data preparation
    prep_start_time = time.time()

    unique_colors = plt.cm.tab20(np.linspace(0, 1, len(particles)))  # Tab20 colormap
    unique_colors = ["rgb({}, {}, {})".format(int(c[0] * 255), int(c[1] * 255), int(c[2] * 255)) for c in unique_colors]

    fig = go.Figure()
    num_voxels_to_plot = 0
    for i, (particle_label, voxel_indices) in enumerate(particles.items()):
        # coords = indices_to_coordinates(voxel_indices, voxel_size, grid_size)
        coords = voxel_centers[voxel_indices]
        fig.add_trace(go.Scatter3d(
            x=coords[:, 0],
            y=coords[:, 1],
            z=coords[:, 2],
            mode='markers',
            marker=dict(size=voxel_size, color=unique_colors[i]),
            name=f"Particle {particle_label}",
            showlegend=False 
        ))
        num_voxels_to_plot += len(coords)

    # End timing for data preparation
    prep_end_time = time.time()
    prep_duration = prep_end_time - prep_start_time
    print(f"Time taken for method-specific data preparation: {prep_duration:.2f} seconds")
    print(f"Plotting {num_voxels_to_plot} voxels")

    # Start timing for plotting
    plot_start_time = time.time()

    # Configure layout
    fig.update_layout(
        scene=dict(
            xaxis=dict(title="X", range=[0, domain_size[0]]),
            yaxis=dict(title="Y", range=[0, domain_size[1]]),
            zaxis=dict(title="Z", range=[0, domain_size[2]]),
        ),
        title="3D Particle Visualization (Batch)",
    )

    # Show plot
    fig.show()

    plot_end_time = time.time()
    plot_duration = plot_end_time - plot_start_time
    print(f"Time taken for plotting: {plot_duration:.2f} seconds")

def plot_with_plotly_batch(particles, voxel_centers, voxel_size, domain_size, batch_size=100000):
    """
    Plot particles in 3D using Plotly with batch processing.

    Parameters:
        particles (dict): Dictionary mapping particle labels to their voxel indices.
        voxel_centers (numpy.ndarray): Array of voxel center coordinates.
        voxel_size (float): The size of each voxel.
        domain_size (tuple): Shape of the 3D grid in real space (x, y, z dimensions).
        batch_size (int): Number of voxels per batch.
    """

    # Start timing for data preparation
    prep_start_time = time.time()

    unique_colors = plt.cm.tab20(np.linspace(0, 1, len(particles)))  # Tab20 colormap
    unique_colors = [tuple(c[:3]) for c in unique_colors]  # Convert RGBA to RGB

    all_coords = []
    all_colors = []

    for i, (particle_label, voxel_indices) in enumerate(particles.items()):
        coords = voxel_centers[voxel_indices]
        all_coords.append(coords)
        all_colors.extend([unique_colors[particle_label]] * len(coords))  # Use the color for the particle

    # Combine all coordinates into a single array
    all_coords = np.vstack(all_coords)
    print(f"Total voxels to plot: {all_coords.shape[0]}")

    # End timing for data preparation
    prep_end_time = time.time()
    prep_duration = prep_end_time - prep_start_time
    print(f"Time taken for data preparation: {prep_duration:.2f} seconds")
    print(f"Plotting {len(all_coords)} voxels")

    # Start timing for plotting
    plot_start_time = time.time()

    # Validate coordinates before plotting
    # validate_voxel_coordinates(all_coords, domain_size)

    # Batch processing
    fig = go.Figure()
    num_batches = int(np.ceil(len(all_coords) / batch_size))
    for batch_idx in range(num_batches):
        start = batch_idx * batch_size
        end = min((batch_idx + 1) * batch_size, len(all_coords))

        batch_coords = all_coords[start:end]
        batch_colors = np.arange(start, end)  # Assign unique values for batch coloring

        # Add each batch as a separate trace
        fig.add_trace(go.Scatter3d(
            x=batch_coords[:, 0],
            y=batch_coords[:, 1],
            z=batch_coords[:, 2],
            mode='markers',
            marker=dict(
                size=voxel_size,
                color=batch_colors,
                colorscale='Viridis',
            ),
            showlegend=False  # Optionally disable legend for batch plotting
        ))

    # Configure layout
    fig.update_layout(
        scene=dict(
            xaxis=dict(title="X", range=[0, domain_size[0]]),
            yaxis=dict(title="Y", range=[0, domain_size[1]]),
            zaxis=dict(title="Z", range=[0, domain_size[2]]),
        ),
        title="3D Particle Visualization (Batch)"
    )

    # Show plot
    fig.show()

    plot_end_time = time.time()
    plot_duration = plot_end_time - plot_start_time
    print(f"Time taken for plotting: {plot_duration:.2f} seconds")

def plot_with_pyvista_polydata(domain_data, voxel_centers, voxel_size, domain_size, grid_size, slices=None, slice_positions=None, axis='z', cmap=None, show_legend=False):
    if axis not in {'x', 'y', 'z'}:
        raise ValueError(f"Invalid axis '{axis}'. Must be one of 'x', 'y', or 'z'.")
    
    # Flag to show or hide slices
    show_slices = True

    # Start timing for data preparation
    prep_start_time = time.time()
    
    # Use provided or default colormap
    if cmap is None:
        cmap = create_colormap(domain_data, make_unique=False)  # Default fluorescent green
    
    all_coords = []
    all_colors = []
    
    # Ensure numerical sorting of particle labels
    # label_to_index = {str(label): idx + 1 for idx, label in enumerate(sorted(map(int, particles.keys())))}
    label_to_index = {str(label): idx for idx, label in enumerate(sorted(domain_data.keys(), key=str))}

    for i, (entity_label, voxel_indices) in enumerate(domain_data.items()):
        coords = voxel_centers[voxel_indices]
        all_coords.append(coords)

        # Map particle_label to exact color in colormap
        particle_index = label_to_index[str(entity_label)]  # Use the label-to-index mapping
        color = cmap.colors[particle_index]  # Fetch color directly from colormap
        repeated_colors = np.tile(color, (len(coords), 1))  # Shape: (len(coords), 3)
        all_colors.append(repeated_colors)

        # Map particle_label to color via cmap
        # particle_index = int(particle_label)  # Ensure particle_label is treated as an index
        # color = cmap(particle_index) if particle_index < len(cmap.colors) else (0, 0, 0)  # Default to black if out of bounds
        # all_colors.extend([color] * len(coords))

    if not all_coords:
        raise ValueError("No particle coordinates found.")

    # Combine all coordinates and scalars into arrays
    all_coords = np.vstack(all_coords)
    all_colors = np.vstack(all_colors)

    # Validate all_colors shape
    if all_colors.ndim != 2 or all_colors.shape[1] not in [3, 4]:
        raise ValueError(f"Invalid all_colors shape: {all_colors.shape}")

    # Create a PolyData object for all voxels
    combined_mesh = pv.PolyData(all_coords)
    cube = pv.Cube(x_length=voxel_size, y_length=voxel_size, z_length=voxel_size)

    # Use glyph to replicate cubes for all voxel centers
    combined_mesh = combined_mesh.glyph(geom=cube, scale=False, orient=False)

    # Assign colors to the glyph
    combined_mesh.cell_data["color"] = np.repeat(all_colors, cube.n_cells, axis=0)
    
    # End timing for data preparation
    prep_end_time = time.time()
    prep_duration = prep_end_time - prep_start_time
    print(f"Time taken for method-specific data preparation: {prep_duration:.2f} seconds")

    # Start timing for plotting
    plot_start_time = time.time()

    # Determine axis indices
    axis_indices = {'x': 0, 'y': 1, 'z': 2}
    axis_index = axis_indices[axis]
    other_axes = [i for i in range(3) if i != axis_index]  # Non-chosen axes

    plotter = pv.Plotter()
    # plotter.add_mesh(
    #     combined_mesh, 
    #     scalars=None, 
    #     rgb=True,
    #     show_edges=False, 
    #     show_scalar_bar=False)
    
    plotter.add_mesh(
        combined_mesh, 
        scalars='color', 
        rgb=True,
        show_edges=False, 
        show_scalar_bar=False)
    
    if show_legend:
        legend_entries = [
            (f"{entity_label}", cmap.colors[label_to_index[entity_label]])
            for entity_label in entity_label.keys()
        ]
        plotter.add_legend(legend_entries, bcolor='white', size=(0.2, 0.2))
    
    if slices is not None and slice_positions is not None and show_slices == True:
        # axis_index = {'x': 0, 'y': 1, 'z': 2}[axis]
        for pos in slice_positions:
            plane = pv.Plane(
                center=(
                    pos if axis_index == 0 else domain_size[0] / 2,
                    pos if axis_index == 1 else domain_size[1] / 2,
                    pos if axis_index == 2 else domain_size[2] / 2,
                ),
                direction=(
                    1 if axis == 'x' else 0,
                    1 if axis == 'y' else 0,
                    1 if axis == 'z' else 0,
                ),
                # i_size=domain_size[(axis_index + 1) % 3],
                # j_size=domain_size[(axis_index + 2) % 3],
                i_size=domain_size[other_axes[1]],  # Full span along the first orthogonal axis
                j_size=domain_size[other_axes[0]], 
            )
            plotter.add_mesh(plane, color='white', opacity=0.4, show_edges=False)

    # Add axes for orientation
    plotter.show_bounds(grid=True, location="outer", xtitle="X", ytitle="Y", ztitle="Z")
    plotter.add_axes(interactive=True)  # Interactive axes for orientation

    plot_end_time = time.time()
    plot_duration = plot_end_time - plot_start_time
    print(f"Time taken for plotting: {plot_duration:.2f} seconds")

    plotter.show(interactive=True)

def plot_slices_as_images(slices, slice_coordinates, 
                          voxel_size, grid_size, axis='z', 
                          cmap=None, debug_mode=False):
    """
    Plot slices as 2D images with axes scaled to real-world coordinates.

    Parameters:
        slices (list): List of 2D numpy arrays representing slices.
        slice_coordinates (list): List of slice midpoints (real-world coordinates) along the chosen axis.
        voxel_size (float): Size of each voxel in real-world units.
        grid_size (tuple): Shape of the 3D grid (nx, ny, nz).
        axis (str): Axis along which the slices were taken ('x', 'y', 'z').
        generate_images (bool): If True, saves each slice as a .tiff file with no axes or labels.
        output_dir (str): Directory to save slice images.
        label_file_name (str): Identifier for the original file (e.g., .dat or .json).
    """
    
    # Use provided unique colors
    if cmap is None or debug_mode==False:
        cmap = ListedColormap(['black', '#00FF00'])  # Default fluorescent green

    num_slices = len(slices)
    if num_slices == 0: return

    fig, axes = plt.subplots(1, num_slices, figsize=(5 * num_slices, 5), constrained_layout=True)

    if num_slices == 1:
        axes = [axes]

    # Axis labels for the two axes orthogonal to the slicing axis
    axis_labels = {'x': ('y', 'z'), 'y': ('x', 'z'), 'z': ('x', 'y')}
    other_axes = axis_labels[axis]

    for i, (slice_data, z_coord) in enumerate(zip(slices, slice_coordinates)):
        ax = axes[i]

        # Create the real-world grid extent for the current slice
        if axis == 'z':  # Slicing along z-axis
            extent = [0, grid_size[0] * voxel_size, 0, grid_size[1] * voxel_size]  # x vs. y
        elif axis == 'y':  # Slicing along y-axis
            extent = [0, grid_size[0] * voxel_size, 0, grid_size[2] * voxel_size]  # x vs. z
        elif axis == 'x':  # Slicing along x-axis
            extent = [0, grid_size[1] * voxel_size, 0, grid_size[2] * voxel_size]  # y vs. z

        # Plot the slice using the real-world extent
        im = ax.imshow(
            slice_data,
            cmap=cmap,
            origin='lower',
            interpolation='nearest',
            vmin=0, 
            vmax=len(cmap.colors) - 1
        )

        # Set axis labels and title
        ax.set_title(f"Slice @ {z_coord:.2f} (real-world units)")
        ax.set_xlabel(f"{other_axes[0]} (real-world units)")
        ax.set_ylabel(f"{other_axes[1]} (real-world units)")

        # Ensure correct aspect ratio
        # ax.set_aspect('equal')

        # Remove spines and ticks for a clean look
        ax.spines[:].set_visible(False)
        ax.tick_params(colors='black')

    plt.tight_layout()
    plt.show(block=False)

def create_colormap(particles, make_unique=True):
    """
    Create a colormap for particles.

    Parameters:
        particles (dict): A dictionary where keys are particle IDs.
        make_unique (bool): If True, assign unique colors to each particle. 
                            If False, assign fluorescent green to all particles.

    Returns:
        ListedColormap: A colormap with either unique or uniform colors.
    """
    particle_ids = sorted(map(str, particles.keys()))  # Ensure all particle IDs are strings

    if make_unique:
        # Generate consistent unique colors for all particles
        unique_colors = {
            pid: plt.cm.tab20(i / len(particle_ids))[:3]  # Normalize RGB to [0, 1] and ignore alpha
            for i, pid in enumerate(particle_ids)
        }
        # Create a ListedColormap using unique colors
        cmap = ListedColormap(
            [(0, 0, 0)] + [unique_colors[pid] for pid in particle_ids]  # Black + unique colors
        )
    else:
        # Create a uniform fluorescent green colormap
        cmap = ListedColormap([(0, 0, 0), (0.5, 1.0, 0.0)])  # Black and fluorescent green

    return cmap

def plot_pointcloud_with_slices(base_file_name, label_dir, slice_dir, metadata_path, num_slices=20, axis='z'):
    """
    Visualizes a 3D point cloud (from `.ply` file) and overlays 2D slices (from `.tiff` files).
    
    Parameters:
        base_file_name (str): Base name for files.
        label_dir (str): Directory containing the `.ply` point cloud.
        slice_dir (str): Directory containing the TIFF slice images.
        metadata_path (str): Path to the metadata JSON file.
        num_slices (int): Number of slices to visualize.
        axis (str): Axis along which slices were taken ('x', 'y', 'z').

    Returns:
        None
    """

    metadata_file = Path(metadata_path)

    if not metadata_file.exists():
        print(f"❌ Metadata file not found: {metadata_path}")
        return

    # 🔹 Load slice metadata
    with open(metadata_file, "r") as f:
        metadata = json.load(f)

    dataset_metadata = next((s for s in metadata["scaffolds"] if s["filename"] == base_file_name), None)
    if not dataset_metadata:
            print(f"❌ No metadata found for {base_file_name}.")
            return
    
    slices_info = dataset_metadata["slices"]
    
    # 🔹 Auto-detect point cloud format (.ply or .npy)
    ply_path = os.path.join(label_dir, f"{base_file_name}_label.ply")
    npy_path = os.path.join(label_dir, f"{base_file_name}_label.npy")

    if os.path.exists(ply_path):
        print(f"✅ Loading .PLY file: {ply_path}")
        ply_data = PlyData.read(ply_path)
        vertex = ply_data['vertex']
        x_points = np.array(vertex['x'])
        y_points = np.array(vertex['y'])
        z_points = np.array(vertex['z'])

    elif os.path.exists(npy_path):
        print(f"✅ Loading .NPY file: {npy_path}")
        pointcloud = np.load(npy_path)  # Shape: (N, 4) → (x, y, z, label)
        x_points, y_points, z_points = pointcloud[:, 0], pointcloud[:, 1], pointcloud[:, 2]

    else:
        print(f"❌ No valid point cloud file found for {base_file_name}. Expected `.ply` or `.npy`.")
        return

    print(f"✅ Loaded point cloud (Points: {len(x_points)})")

    # 🔹 Create PyVista plotter
    plotter = pv.Plotter()

    # 🔹 Create and add point cloud mesh
    pointcloud_mesh = pv.PolyData(np.column_stack((x_points, y_points, z_points)))
    plotter.add_mesh(pointcloud_mesh, color="blue", point_size=2, opacity=0.1, label="Point Cloud")

    # 🔹 Select evenly spaced slices for visualization
    total_slices = len(slices_info)
    selected_slices = [slices_info[i] for i in np.linspace(0, total_slices - 1, num_slices, dtype=int)]

    # 🔹 Overlay slices in 3D (based on metadata.json)
    for slice_info in selected_slices:
        slice_path = os.path.join(slice_dir, slice_info["filename"])
        grid_position = slice_info["grid_position"]  # Stored in grid space

        slice_file = Path(slice_path)

        if not slice_file.exists():
            print(f"⚠️ Missing slice file: {slice_path}")
            continue

        # 🔹 Load the 2D slice image
        slice_img = cv2.imread(str(slice_file), cv2.IMREAD_GRAYSCALE)
        if slice_img is None:
            print(f"❌ Error loading slice: {slice_path}")
            continue

        # 🔹 Fix orientation (flip Y-axis)
        # slice_img = cv2.flip(slice_img, 0)

        print(f"✅ Loaded slice: {slice_path} (Shape: {slice_img.shape})")

        # 🔹 Convert 2D slice into 3D grid positions
        height, width = slice_img.shape
        y_indices, x_indices = np.nonzero(slice_img)  # Find nonzero pixels (particles)

        # Check if slice contains any points before proceeding
        if len(x_indices) == 0 or len(y_indices) == 0:
            print(f"⚠️ Slice {slice_path} contains no nonzero pixels (empty mesh). Skipping.")
            continue

        # Assign fixed slice coordinate along the given axis
        grid_x = x_indices  # Already in grid space
        grid_y = y_indices  # Already in grid space
        grid_z = np.full_like(x_indices, grid_position, dtype=int)

        if axis == 'x':
            slice_voxels = np.column_stack((grid_z, grid_x, grid_y))  # Projected along X
        elif axis == 'y':
            slice_voxels = np.column_stack((grid_x, grid_z, grid_y))  # Projected along Y
        else:  # axis == 'z'
            slice_voxels = np.column_stack((grid_x, grid_y, grid_z))  # Projected along Z

        # 🔹 Create PyVista mesh for the slice projection
        slice_mesh = pv.PolyData(slice_voxels)

        # Check if the mesh is empty
        if slice_mesh.n_points == 0:
            print(f"⚠️ Slice {slice_path} produced an empty mesh. Skipping visualization.")
            continue  # Skip adding empty slices

        plotter.add_mesh(slice_mesh, color="red", point_size=5, opacity=1.0, label=f"Slice {grid_position}")

    # 🔹 Configure the plotter
    plotter.add_axes(interactive=True)
    plotter.show_bounds(grid=True, location="outer", xtitle="X", ytitle="Y", ztitle="Z")
    plotter.show()

def visualize_voxel_grid(npz_file, axis=2, num_slices=5, slice_coordinates=None, voxel_size=2.0, cmap="jet"):
    """
    Visualizes slices of a 3D voxel grid from a saved .npz label file, allowing for specific slice coordinates.

    Parameters:
        npz_file (str): Path to the .npz file containing the voxel grid.
        slice_axis (str): The axis along which to slice (0=X, 1=Y, 2=Z).
        num_slices (int): Number of slices to visualize (ignored if `slice_midpoints` is provided).
        slice_midpoints (list or None): List of real-world slice positions (optional).
        voxel_size (float): Size of each voxel in real-world units.
    """
    # Load the .npz file
    data = np.load(npz_file)
    voxel_grid = data["voxel_grid"]  # (nx, ny, nz) 3D array
    grid_size = data["grid_size"]  # (nx, ny, nz)

    # print(f"Voxel Grid Shape: {voxel_grid.shape}")
    # print(f"Unique Labels in Grid: {np.unique(voxel_grid)}")  # Check particle labels

    if axis == 'z':  # Slicing along z-axis
        slice_axis = 2
    elif axis == 'y':  # Slicing along y-axis
        slice_axis = 1
    elif axis == 'x':  # Slicing along x-axis
        slice_axis = 0

    # Convert real-world slice positions into grid indices
    if slice_coordinates is not None:
        slice_indices = np.array((np.array(slice_coordinates) / voxel_size).astype(int))
        slice_indices = np.clip(slice_indices, 0, grid_size[slice_axis] - 1)  # Ensure within bounds
    else:
        # Evenly spaced slices if no specific midpoints provided
        slice_indices = np.linspace(0, grid_size[slice_axis] - 1, num_slices, dtype=int)

    # Create figure
    fig, axes = plt.subplots(1, len(slice_indices), figsize=(15, 5))

    # Plot slices
    for i, (idx, real_coord) in enumerate(zip(slice_indices, slice_coordinates if slice_coordinates is not None else slice_indices * voxel_size)):
        if slice_axis == 0:
            slice_data = voxel_grid[idx, :, :]  # X-axis slices
        elif slice_axis == 1:
            slice_data = voxel_grid[:, idx, :]  # Y-axis slices
        else:
            slice_data = voxel_grid[:, :, idx]  # Z-axis slices (default)

        ax = axes[i]
        ax.imshow(slice_data, cmap=cmap)
        # ax.set_title(f"Slice {idx} (Real: {idx * voxel_size:.2f})")
        ax.set_title(f"Slice {idx} (Real: {real_coord:.2f})")  # Show exact real coordinate
        ax.axis("off")

    plt.tight_layout()
    plt.show(block=False)

def plot_padded_slice_with_pointcloud(slice_path, pointcloud_path, voxel_size, slice_index, base_file_name, axis='z'):
    """
    Plots a single padded 2D slice overlayed with the 3D point cloud.

    Parameters:
        slice_path (str): Path to the padded TIFF image.
        pointcloud_path (str): Path to the PLY point cloud file.
        voxel_size (float): Size of each voxel in real-world units.
        slice_index (int): Index of the slice to visualize in 3D.
        axis (str): Axis along which the slices were taken ('x', 'y', 'z').
        
    Returns:
        None
    """

    # slice file path
    slice_path = os.path.join(slice_path, f"{base_file_name}_slice_{slice_index:03d}.tiff")
    label_path = os.path.join(pointcloud_path, f"{base_file_name}_label.ply")

    # 🔹 Load the padded slice
    slice_img = cv2.imread(slice_path, cv2.IMREAD_GRAYSCALE)
    if slice_img is None:
        print(f"❌ Error: Could not load slice at {slice_path}")
        return
    
    print(f"✅ Loaded padded slice: {slice_path} (Shape: {slice_img.shape})")

    # 🔹 Load the point cloud
    ply_data = PlyData.read(label_path)
    vertex = ply_data['vertex']
    
    x_points = np.array(vertex['x'])
    y_points = np.array(vertex['y'])
    z_points = np.array(vertex['z'])

    print(f"✅ Loaded point cloud: {label_path} (Points: {len(x_points)})")

    # 🔹 Convert slice to 3D coordinates
    height, width = slice_img.shape
    y_indices, x_indices = np.nonzero(slice_img)  # Find nonzero pixels (particles)
    
    # Convert 2D indices to 3D world coordinates
    slice_voxels = []
    for x, y in zip(x_indices, y_indices):
        real_x = x * voxel_size
        real_y = y * voxel_size
        real_z = slice_index * voxel_size  # Position along slicing axis
        slice_voxels.append((real_x, real_y, real_z))

    slice_voxels = np.array(slice_voxels)
    
    print(f"✅ Extracted {len(slice_voxels)} 3D points from the padded slice.")

    # 🔹 Plot in 3D
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection='3d')
    
    # Plot point cloud
    ax.scatter(x_points, y_points, z_points, c='b', marker='o', alpha=0.2, label="Point Cloud (3D)")

    # Plot slice voxels
    if slice_voxels.size > 0:
        ax.scatter(slice_voxels[:, 0], slice_voxels[:, 1], slice_voxels[:, 2], 
                   c='r', marker='s', alpha=1.0, label="Padded Slice (Projected)")

    # Labels and legend
    ax.set_xlabel('X (Real-World Units)')
    ax.set_ylabel('Y (Real-World Units)')
    ax.set_zlabel('Z (Slice Position)')
    ax.set_title(f'3D Overlay of Padded Slice #{slice_index} and Point Cloud')
    ax.legend()
    plt.show()

    print("✅ Visualization complete!")