import numpy as np

def filter_incomplete_beads(bead_data, grid, threshold, axis=2):
    filtered_data = {}
    max_z = np.max(grid[:, axis])

    for bead_id, ranges in bead_data.items():
        indices = []
        for r in ranges:
            indices.extend(range(r[0], r[1] + 1))

        coords = grid[indices]
        max_z_hits = np.sum(np.isclose(coords[:, axis], max_z))

        if max_z_hits < threshold:
            filtered_data[bead_id] = ranges

    return filtered_data

