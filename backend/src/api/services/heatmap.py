import numpy as np
import cv2
from PIL import Image

def generate_forgery_heatmap(original_image: Image.Image, prediction_mask: np.ndarray) -> Image.Image:
    """
    Generate a heatmap from a forgery prediction mask and overlay it on the original image.
    :param original_image: PIL.Image of the input image
    :param prediction_mask: 2D NumPy array with values between 0 and 1
    :return: PIL.Image with heatmap overlay
    """
    # Convert PIL to RGB NumPy array
    original_np = np.array(original_image.convert("RGB"))
    h, w = original_np.shape[:2]

    # Resize prediction mask to image size
    heatmap = cv2.resize(prediction_mask, (w, h))
    heatmap = np.uint8(255 * heatmap)  # Scale to 0-255

    # Apply color map
    heatmap_color = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

    # Blend with original image
    overlay = cv2.addWeighted(original_np, 0.6, heatmap_color, 0.4, 0)

    return Image.fromarray(overlay)
