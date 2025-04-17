import os
import numpy as np
import tifffile as tiff
from skimage.measure import label
from skimage.morphology import closing
from skimage.filters import threshold_otsu, gaussian
import scipy.ndimage
# from skimage.transform import resize
# from visualization_methods import plot_particles
from scipy.ndimage import zoom

def load_tiff_slices(tiff_files):
    """
    Loads a series of 2D TIFF images from a list of file paths and stacks them into a 3D numpy array.

    Parameters:
        tiff_files (list of str): Paths to the TIFF slice images.

    Returns:
        numpy.ndarray: 3D numpy array representing the volume.
    """
    slices = [tiff.imread(f) for f in sorted(tiff_files)]  # Load images and sort to ensure correct order
    volume = np.stack(slices, axis=-1)  # Stack along the z-axis
    return volume

def gray_to_binary(volume, threshold=None, blur=None):
    """
    Converts a grayscale 3D volume to binary using Otsu's method or a specified threshold.
    
    Parameters:
        volume (numpy.ndarray): 3D grayscale volume.
        threshold (float, optional): Threshold value. If None, Otsu's method is used.
        blur (float, optional): Sigma value for Gaussian blur.
    
    Returns:
        numpy.ndarray: Binary 3D array.
    """
    if threshold is None:
        threshold = threshold_otsu(volume)
    
    if blur is not None:
        volume = gaussian(volume, sigma=blur)
    
    binary_volume = volume > threshold
    return binary_volume.astype(np.uint8)

def remove_disconnected_regions(labelled_volume):
    """
    Remove disconnected regions from labeled particles, keeping only the largest connected component per particle.
    
    Parameters:
        labelled_volume (numpy.ndarray): Labeled 3D array.
    
    Returns:
        numpy.ndarray: Cleaned labeled 3D array.
    """
    cleaned_volume = np.zeros_like(labelled_volume)
    unique_labels = np.unique(labelled_volume)
    
    for label_id in unique_labels[1:]:  # Skip background (label 0)
        mask = labelled_volume == label_id
        connected_components, num_features = scipy.ndimage.label(mask)
        
        if num_features > 1:
            component_sizes = np.bincount(connected_components.ravel())[1:]
            largest_component = np.argmax(component_sizes) + 1
            cleaned_volume[connected_components == largest_component] = label_id
        else:
            cleaned_volume[mask] = label_id
    
    return cleaned_volume

def segment_particles(binary_volume, min_voxels=100):
    """
    Segments individual particles using 3D connected component labeling.
    
    Parameters:
        binary_volume (numpy.ndarray): 3D binary array where particles are 1 and background is 0.
        min_voxels (int, optional): Minimum voxel count for a valid particle (default=100).
    
    Returns:
        numpy.ndarray: 3D array with labeled particles.
    """
    # Apply morphological closing to clean noise
    structure = np.ones((3, 3, 3), dtype=np.uint8)
    closed_volume = closing(binary_volume, structure)
    
    # Label connected components
    labeled_volume = label(closed_volume)
    print(f"Segmented {labeled_volume.max()} regions. Filtering small regions...")
    
    # Remove small particles
    label_sizes = np.bincount(labeled_volume.ravel())
    large_labels = np.where(label_sizes >= min_voxels)[0]
    
    filtered_volume = np.isin(labeled_volume, large_labels) * labeled_volume
    
    # Relabel sequentially
    unique_labels = np.unique(filtered_volume)
    relabeled_volume = np.zeros_like(labeled_volume)
    for i, label_id in enumerate(unique_labels[1:], start=1):  # Skip 0 (background)
        relabeled_volume[filtered_volume == label_id] = i
    
    # Remove disconnected regions
    cleaned_volume = remove_disconnected_regions(relabeled_volume)
    
    print(f"Final segmentation contains {cleaned_volume.max()} particles.")
    return cleaned_volume

from skimage.transform import resize

def resample_3d_labels(label_volume, voxel_sizes_segmented, voxel_sizes_truth):
    """
    Resamples a labeled 3D volume to match a given voxel spacing using nearest-neighbor interpolation.

    Parameters:
        label_volume (numpy.ndarray): The labeled 3D volume.
        voxel_sizes_segmented (tuple): The voxel size (dx, dy, dz) for the segmented slices.
        voxel_sizes_truth (tuple): The voxel size (dx, dy, dz) for the ground truth.

    Returns:
        numpy.ndarray: Resampled labeled 3D volume with the new resolution.
    """
    dx_reconstructed, dy_reconstructed, dz_reconstructed = voxel_sizes_segmented  # Reconstructed voxel spacing
    dx_truth, dy_truth, dz_truth = voxel_sizes_truth  # Target voxel spacing (ground truth)

    # Compute separate scale factors for X, Y, and Z
    scale_factors = (
        dx_reconstructed / dx_truth,  # Scale factor in X
        dy_reconstructed / dy_truth,  # Scale factor in Y
        dz_reconstructed / dz_truth   # Scale factor in Z (important!)
    )

    # Apply zoom-based resampling (true interpolation)
    resampled_labels = zoom(label_volume, zoom=scale_factors, order=0)  # Nearest-neighbor to preserve labels

    print(f"Resampled label volume from shape {label_volume.shape} to {resampled_labels.shape} using scale factors {scale_factors}")

    return resampled_labels.astype(label_volume.dtype)
