from PIL import Image, ImageChops, ImageEnhance
import os
import uuid

def generate_ela_image(image: Image.Image, quality: int = 85, resize_factor: float = 0.5) -> Image.Image:
    """
    Generate an Error Level Analysis (ELA) image from the given input image.

    :param image: PIL Image object
    :param quality: JPEG compression quality for simulating loss (default: 85)
    :param resize_factor: Factor to resize image (0.5 means 50% smaller)
    :return: ELA Image object
    """

    # Resize the image if requested
    if 0 < resize_factor < 1:
        new_width = int(image.width * resize_factor)
        new_height = int(image.height * resize_factor)
        image = image.resize((new_width, new_height), Image.LANCZOS)

    # Save image to a temporary compressed JPEG file
    temp_filename = f"temp_{uuid.uuid4().hex}.jpg"
    image.convert("RGB").save(temp_filename, "JPEG", quality=quality)

    # Load compressed image and calculate the difference
    compressed = Image.open(temp_filename)
    ela_image = ImageChops.difference(image.convert("RGB"), compressed)

    # Determine max difference to scale brightness
    extrema = ela_image.getextrema()
    max_diff = max([channel[1] for channel in extrema])
    scale = 255.0 / max_diff if max_diff != 0 else 1.0

    # Apply brightness enhancement
    ela_image = ImageEnhance.Brightness(ela_image).enhance(scale)

    # Cleanup
    os.remove(temp_filename)

    return ela_image
