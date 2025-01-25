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

from grid_optimization_methods import extract_surface_voxels

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
        plot_with_pyvista_polydata(particles, voxel_centers, voxel_size, domain_size, slices, slice_coordinates, axis, cmap=cmap, show_legend=show_legend)
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

def plot_with_pyvista_polydata(particles, voxel_centers, voxel_size, domain_size, slices=None, slice_positions=None, axis='z', cmap=None, show_legend=False):
    # Start timing for data preparation
    prep_start_time = time.time()
    
    # Use provided or default colormap
    if cmap is None:
        cmap = create_colormap(particles, make_unique=False)  # Default fluorescent green
    
    all_coords = []
    all_colors = []
    
    # Ensure numerical sorting of particle labels
    label_to_index = {str(label): idx + 1 for idx, label in enumerate(sorted(map(int, particles.keys())))}

    for i, (particle_label, voxel_indices) in enumerate(particles.items()):
        coords = voxel_centers[voxel_indices]
        all_coords.append(coords)

        # Map particle_label to exact color in colormap
        particle_index = label_to_index[particle_label]  # Use the label-to-index mapping
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
            (f"Particle {particle_label}", cmap(int(particle_label)))
            for particle_label in particles.keys()
        ]
        plotter.add_legend(legend_entries, bcolor='white', size=(0.2, 0.2))
    
    if slices is not None and slice_positions is not None:
        axis_index = {'x': 0, 'y': 1, 'z': 2}[axis]
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
                i_size=domain_size[(axis_index + 1) % 3],
                j_size=domain_size[(axis_index + 2) % 3],
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
                          cmap=None):
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
    if cmap is None:
        cmap = ListedColormap(['black', '#00FF00'])  # Default fluorescent green

    num_slices = len(slices)
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
            extent=extent,
            interpolation='nearest',
            vmin=0, 
            vmax=len(cmap.colors) - 1
        )

        # Set axis labels and title
        ax.set_title(f"Slice @ {z_coord:.2f} (real-world units)")
        ax.set_xlabel(f"{other_axes[0]} (real-world units)")
        ax.set_ylabel(f"{other_axes[1]} (real-world units)")

        # Ensure correct aspect ratio
        ax.set_aspect('equal')

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