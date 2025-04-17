import matplotlib.pyplot as plt
import pandas as pd

def summarize_slices(slices, slice_positions, grid_size, voxel_size, particles, axis='z'):
    """
    Summarize slice data with information about particle intersections.

    Parameters:
        slices (list): List of 2D numpy arrays representing slices.
        slice_positions (list): List of slice positions along the chosen axis in real-world coordinates.
        grid_size (tuple): Shape of the 3D grid (nx, ny, nz).
        voxel_size (float): Size of each voxel in real-world units.
        particles (dict): Dictionary mapping particle labels to voxel indices.
        axis (str): Axis along which the slices were taken ('x', 'y', 'z').

    Returns:
        pd.DataFrame: DataFrame summarizing the slices.
    """
    # Map axis to its grid size index
    axis_index = {'x': 0, 'y': 1, 'z': 2}[axis]
    grid_positions = [(pos / voxel_size) for pos in slice_positions]

    summary_data = []
    for i, (slice_data, real_pos, grid_pos) in enumerate(zip(slices, slice_positions, grid_positions)):
        # Identify unique particles in the slice
        unique_particles = set(slice_data.flatten()) - {0}  # Exclude background
        num_particles = len(unique_particles)

        # Add data for the current slice
        summary_data.append({
            'Slice': i + 1,
            'Real-World Position': real_pos,
            'Grid Position': grid_pos,
            'Number of Particles': num_particles,
            'Particle IDs': ", ".join(map(str, sorted(unique_particles)))
        })

    # Convert to DataFrame for easier display
    df_summary = pd.DataFrame(summary_data)

    print(df_summary)
    
    return df_summary
