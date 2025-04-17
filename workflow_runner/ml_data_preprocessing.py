from core.ml_data_preprocessing_methods import preprocess_dataset

def run(config):
    preprocess_dataset(
        metadata_path=config["metadata_path"],
        labels_dir=config["raw_labels_dir"],
        slices_dir=config["raw_slices_dir"],
        output_labels_dir=config["preprocessed_labels_dir"],
        output_slices_dir=config["preprocessed_slices_dir"],
        max_grid_size=config["max_grid_size"]
    )