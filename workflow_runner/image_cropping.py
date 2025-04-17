from core.image_cropping_methods import crop_image

def run(config):
    crop_image(
        input_path=config["input_path"],
        output_path=config["output_path"],
        left=config.get("left", "0%"),
        right=config.get("right", "0%"),
        top=config.get("top", "0%"),
        bottom=config.get("bottom", "0%")
    )