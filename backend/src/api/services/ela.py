from PIL import Image, ImageChops, ImageEnhance
import os
import uuid

def generate_ela_image(image: Image.Image, quality=90) -> Image.Image:
    temp_filename = f"temp_{uuid.uuid4().hex}.jpg"
    image.convert("RGB").save(temp_filename, "JPEG", quality=quality)

    compressed = Image.open(temp_filename)
    ela_image = ImageChops.difference(image.convert("RGB"), compressed)

    extrema = ela_image.getextrema()
    max_diff = max([ex[1] for ex in extrema])
    scale = 255.0 / max_diff if max_diff != 0 else 1.0

    ela_image = ImageEnhance.Brightness(ela_image).enhance(scale)

    os.remove(temp_filename)
    return ela_image
