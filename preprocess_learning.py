import json
import os
import numpy as np
import cv2

metadata_cache = None

def load_metadata(metadata_json_path):
    """Loads metadata.json only if it's not already cached."""
    global metadata_cache
    if metadata_cache is None:
        with open(metadata_json_path, "r") as f:
            metadata_cache = json.load(f)
        print("Metadata loaded and cached.")
    return metadata_cache

def get_max_grid_size(metadata_json_path, labels_dir):
    """
    Finds the largest grid dimension in the dataset to determine padding size.

    :param metadata_json_path: Path to metadata.json.
    :param labels_dir: Directory containing .npz ground truth files.
    :return: Maximum dimension to use for cubic padding.
    """
    metadata = load_metadata(metadata_json_path)
    max_dim = 0

    for file_entry in metadata["files"]:
        label_path = os.path.join(labels_dir, os.path.basename(file_entry["filename_label"]))
        
        try:
            npz_data = np.load(label_path)
            grid_size = tuple(npz_data["grid_size"])
            max_dim = max(max_dim, max(grid_size))  # Track largest dimension
        except Exception as e:
            print(f"⚠️ Error loading {label_path}: {e}")

    print(f"📏 Maximum grid size for cubic padding: {max_dim}")
    return max_dim

def pad_slice(img, max_grid_size):
    """
    Pads a 2D image to max_grid_size while keeping it centered.

    :param img: 2D numpy array of the slice.
    :param max_grid_size: Target size (height, width).
    :return: Padded image.
    """
    pad_x = (max_grid_size - img.shape[0]) // 2
    pad_y = (max_grid_size - img.shape[1]) // 2

    img_padded = np.pad(img, ((pad_x, max_grid_size - img.shape[0] - pad_x),
                              (pad_y, max_grid_size - img.shape[1] - pad_y)),
                        mode='constant', constant_values=0)
    return img_padded

def pad_voxel_grid(voxel_grid, grid_size, max_grid_size, slicing_axis):
    """
    Pads a 3D voxel grid to max_grid_size while keeping the slicing axis unchanged.
    
    Steps:
    1. Rotate grid BACK to original (nx, ny, nz) frame.
    2. Compute padding and apply it.
    3. Rotate grid FORWARD again to match `.npz` expected orientation.

    :param voxel_grid: The rotated 3D voxel grid from `.npz`.
    :param grid_size: The original (nx, ny, nz) grid size before rotation.
    :param max_grid_size: The target cubic size for padding.
    :param slicing_axis: The axis along which slices were taken ('x', 'y', or 'z').
    :return: Padded voxel grid in the correct rotated orientation.
    """

    # 🔹 Step 1: Rotate Grid Back to Original Orientation
    if slicing_axis == "y":
        voxel_grid = np.rot90(voxel_grid, k=-1, axes=(0, 2))  # Reverse Y rotation
    elif slicing_axis == "x":
        voxel_grid = np.rot90(voxel_grid, k=-1, axes=(1, 2))  # Reverse X rotation
    elif slicing_axis == "z":
        voxel_grid = np.rot90(voxel_grid, k=-1, axes=(0, 1))  # Reverse Z rotation

    # 🔹 Step 2: Compute Padding (Ensuring No Padding on the Slicing Axis)
    nx, ny, nz = grid_size  # Original shape
    pad_x = ((max_grid_size - nx) // 2, max_grid_size - nx - (max_grid_size - nx) // 2)
    pad_y = ((max_grid_size - ny) // 2, max_grid_size - ny - (max_grid_size - ny) // 2)
    pad_z = ((max_grid_size - nz) // 2, max_grid_size - nz - (max_grid_size - nz) // 2)

    # 🔹 DO NOT pad along slicing axis
    if slicing_axis == "x":
        pad_x = (0, 0)
    elif slicing_axis == "y":
        pad_y = (0, 0)
    elif slicing_axis == "z":
        pad_z = (0, 0)

    # Apply padding
    voxel_grid_padded = np.pad(voxel_grid, (pad_x, pad_y, pad_z), mode='constant', constant_values=0)

    # 🔹 Step 3: Rotate Grid Forward Again to Match `.npz` Orientation
    if slicing_axis == "y":
        voxel_grid_padded = np.rot90(voxel_grid_padded, k=1, axes=(0, 2))  # Reapply Y rotation
    elif slicing_axis == "x":
        voxel_grid_padded = np.rot90(voxel_grid_padded, k=1, axes=(1, 2))  # Reapply X rotation
    elif slicing_axis == "z":
        voxel_grid_padded = np.rot90(voxel_grid_padded, k=1, axes=(0, 1))  # Reapply Z rotation

    # print(f"✅ Applied Padding (X, Y, Z): {pad_x}, {pad_y}, {pad_z}")
    # print(f"✅ Final Grid Size (after reapplying rotation): {voxel_grid_padded.shape}")

    return voxel_grid_padded

def extract_relevant_slices(voxel_grid, slice_indices, slicing_axis):
    """
    Extracts the corresponding slices from the voxel grid based on metadata.
    Ensures the final shape matches (H, W, D) by transposing correctly.

    :param voxel_grid: The full 3D voxel grid.
    :param slice_indices: List of slice indices extracted from metadata.
    :param slicing_axis: The axis along which slices were taken ('x', 'y', or 'z').
    :return: Extracted 3D voxel grid with shape (H, W, D).
    """
    if slicing_axis == "x":
        extracted = voxel_grid[slice_indices, :, :]  # Shape: (D, H, W)
        extracted = np.transpose(extracted, (1, 2, 0))  # ✅ Shape: (H, W, D)

    elif slicing_axis == "y":
        extracted = voxel_grid[:, slice_indices, :]  # Shape: (H, D, W)
        extracted = np.transpose(extracted, (0, 2, 1))  # ✅ Shape: (H, W, D)

    elif slicing_axis == "z":
        extracted = voxel_grid[:, :, slice_indices]  # ✅ Already Shape: (H, W, D)

    return extracted  # Final shape always (H, W, D)

def preprocess_labels(labels_dir, metadata_path, output_dir, max_grid_size):
    """
    Preprocesses all 3D ground truth labels by extracting relevant slices, rotating, and padding.

    :param labels_dir: Directory containing original `.npz` label files.
    :param metadata_path: Path to metadata.json containing slice information.
    :param output_dir: Directory to save preprocessed `.npz` files.
    :param max_grid_size: The cubic grid size to pad everything to.
    """
    os.makedirs(output_dir, exist_ok=True)

    # Load metadata to extract correct slices
    metadata = load_metadata(metadata_path)

    for file_entry in metadata["files"]:
        filename = os.path.basename(file_entry["filename_label"])
        label_path = os.path.join(labels_dir, filename)

        if not os.path.exists(label_path):
            print(f"Skipping missing file: {label_path}")
            continue

        # Load voxel grid
        data = np.load(label_path)
        voxel_grid = data["voxel_grid"]
        grid_size = tuple(data["grid_size"])
        slicing_axis = file_entry["axis"]
        slice_set = file_entry["slices"]
        voxel_size = file_entry["voxel_size"]

        # Step 1: Pad the extracted slices
        voxel_grid_padded = pad_voxel_grid(voxel_grid, grid_size, max_grid_size, slicing_axis)

        # Step 2: Extract only the relevant slices
        slice_indices = [s["grid_position"][slicing_axis] for s in slice_set]
        voxel_grid_extracted = extract_relevant_slices(voxel_grid_padded, slice_indices, slicing_axis)

        # Single Sanity Check: Ensure final shape matches expectations
        expected_shape = (max_grid_size, max_grid_size, len(slice_indices))
        if voxel_grid_extracted.shape != expected_shape:
            print(f"⚠️ Warning: {filename} has incorrect final shape!")
            print(f"   Expected: {expected_shape}, Got: {voxel_grid_extracted.shape}")
            raise ValueError(
                f"🚨 ERROR: {filename} has incorrect final shape!\n"
                f"   Expected: {expected_shape}, Got: {voxel_grid_extracted.shape}"
            )

        # Step 3: Save the preprocessed label
        output_path = os.path.join(output_dir, filename)
        np.savez_compressed(output_path, voxel_grid=voxel_grid_extracted, voxel_size=voxel_size, grid_size=grid_size)
        print(f"Saved preprocessed label: {output_path}")

def preprocess_slices(slices_dir, output_dir, max_grid_size):
    """
    Preprocesses all 2D slices by padding them to max_grid_size.

    :param slices_dir: Directory containing original 2D slice `.tiff` images.
    :param output_dir: Directory to save preprocessed `.tiff` images.
    :param max_grid_size: The final padded size (Height, Width).
    """
    os.makedirs(output_dir, exist_ok=True)

    for filename in os.listdir(slices_dir):
        if filename.endswith(".tiff"):
            slice_path = os.path.join(slices_dir, filename)
            img = cv2.imread(slice_path, cv2.IMREAD_GRAYSCALE)

            # Pad the image
            padded_img = pad_slice(img, max_grid_size)

            # Single Sanity Check: Ensure final shape matches expectations
            expected_shape = (max_grid_size, max_grid_size)
            if padded_img.shape != expected_shape:
              print(f"⚠️ Warning: {slice_path} has incorrect final shape!")
              print(f"   Expected: {expected_shape}, Got: {padded_img.shape}")
              raise ValueError(
                f"🚨 ERROR: {filename} has incorrect final shape!\n"
                f"   Expected: {expected_shape}, Got: {padded_img.shape}"
            )

            # Save the preprocessed image
            output_path = os.path.join(output_dir, filename)
            cv2.imwrite(output_path, padded_img)
            print(f"Saved preprocessed slice: {output_path}")

def preprocess_dataset(metadata_path, labels_dir, slices_dir, output_labels_dir, output_slices_dir, max_grid_size):
    """
    Wrapper function to preprocess both labels and slices.

    :param labels_dir: Directory of raw 3D ground truth labels.
    :param slices_dir: Directory of raw 2D slices.
    :param output_labels_dir: Directory to save preprocessed labels.
    :param output_slices_dir: Directory to save preprocessed slices.
    :param max_grid_size: The target cubic size.
    """
    print("\n🚀 Starting Preprocessing...")
    
    preprocess_labels(labels_dir, metadata_path, output_labels_dir, max_grid_size)
    preprocess_slices(slices_dir, output_slices_dir, max_grid_size)

    print("\n✅ Preprocessing Completed!")

def main():
    # Define directories
    raw_labels_dir = "/Users/mimc/Documents/MIMC/MaterialsAI/DanielAdrianzen/segmentation_2D3D/labels"
    raw_slices_dir = "/Users/mimc/Documents/MIMC/MaterialsAI/DanielAdrianzen/segmentation_2D3D/slices"
    preprocessed_labels_dir = "/Users/mimc/Documents/MIMC/MaterialsAI/DanielAdrianzen/segmentation_2D3D/preprocessed_labels_512x512"
    preprocessed_slices_dir = "/Users/mimc/Documents/MIMC/MaterialsAI/DanielAdrianzen/segmentation_2D3D/preprocessed_slices_512x512"
    metadata_path = "/Users/mimc/Documents/MIMC/MaterialsAI/DanielAdrianzen/segmentation_2D3D/metadata.json"
    max_grid_size = 512

    # Preprocess everything
    preprocess_dataset(metadata_path, raw_labels_dir, raw_slices_dir, preprocessed_labels_dir, preprocessed_slices_dir, max_grid_size)

if __name__ == "__main__":
    main()