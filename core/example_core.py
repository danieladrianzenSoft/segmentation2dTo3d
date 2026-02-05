"""Example core helper functions.

Keep core functions pure and document inputs/outputs. This file demonstrates load/process/save separation.
"""
import numpy as np


def load_and_process(path: str, threshold: float = 0.5) -> np.ndarray:
    """A minimal processing function used by the runner template.

    Args:
        path: path-like input identifier (string). In real code, replace with real I/O.
        threshold: numeric parameter affecting processing.

    Returns:
        numpy.ndarray: a small deterministic array for tests and examples.
    """
    # Minimal deterministic behavior for examples/tests
    data = np.array([0.1, 0.6, 0.3, 0.8])
    mask = (data > threshold).astype(int)
    return mask


def save_result(path, arr: np.ndarray):
    """Save result array to a compressed numpy file.

    Kept simple for demonstration and tests.
    """
    import numpy as _np
    _np.savez_compressed(str(path), result=arr)
