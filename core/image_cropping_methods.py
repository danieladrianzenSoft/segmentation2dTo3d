from PIL import Image
import os

def parse_crop_value(value, total_length):
    """
    Converts a string like '10%' or '50px' into a pixel value.
    Defaults to percentage if no suffix is given.
    """
    if isinstance(value, str):
        value = value.strip().lower()
        if value.endswith("px"):
            return int(value[:-2])
        elif value.endswith("%"):
            return int(total_length * float(value[:-1]) / 100)
        else:
            # Assume it's a percentage
            return int(total_length * float(value) / 100)
    elif isinstance(value, (int, float)):
        # Assume it's raw pixels
        return int(value)
    else:
        raise ValueError(f"Invalid crop value: {value}")

def crop_image(input_path, output_path, left='0%', right='0%', top='0%', bottom='0%'):
    """
    Crops an image with given margins from each side.
    Margins can be in percentage ('10%') or pixels ('50px').
    """
    img = Image.open(input_path)
    width, height = img.size

    # Convert all crop values to pixels
    left_px = parse_crop_value(left, width)
    right_px = parse_crop_value(right, width)
    top_px = parse_crop_value(top, height)
    bottom_px = parse_crop_value(bottom, height)

    # Define the crop box (left, top, right, bottom)
    crop_box = (
        left_px,
        top_px,
        width - right_px,
        height - bottom_px
    )

    # Safety check
    if crop_box[0] >= crop_box[2] or crop_box[1] >= crop_box[3]:
        raise ValueError("Invalid crop dimensions — image would be inverted or empty.")

    cropped = img.crop(crop_box)
    cropped.save(output_path)
    print(f"Cropped image saved as '{output_path}'")
    