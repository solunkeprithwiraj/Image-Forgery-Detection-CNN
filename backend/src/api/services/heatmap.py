import numpy as np
import cv2
from PIL import Image

def generate_forgery_heatmap(original_image: Image.Image, prediction_mask: np.ndarray, 
                            alpha: float = 0.6, colormap: int = cv2.COLORMAP_JET,
                            threshold: float = 0.5) -> Image.Image:
    """
    Generate a heatmap from a forgery prediction mask and overlay it on the original image.
    
    :param original_image: PIL.Image of the input image
    :param prediction_mask: 2D NumPy array with values between 0 and 1
    :param alpha: Transparency of the heatmap overlay (0-1)
    :param colormap: OpenCV colormap to use (e.g., cv2.COLORMAP_JET, cv2.COLORMAP_VIRIDIS)
    :param threshold: Threshold for highlighting potential forgery regions (0-1)
    :return: PIL.Image with heatmap overlay
    """
    # Convert PIL to RGB NumPy array
    original_np = np.array(original_image.convert("RGB"))
    h, w = original_np.shape[:2]

    # Resize prediction mask to image size
    heatmap = cv2.resize(prediction_mask, (w, h), interpolation=cv2.INTER_LINEAR)
    
    # Apply threshold to highlight potential forgery regions
    # Values below threshold will be less prominent
    heatmap_highlighted = np.copy(heatmap)
    heatmap_highlighted[heatmap < threshold] *= 0.3  # Reduce intensity for low-confidence areas
    
    # Scale to 0-255
    heatmap_uint8 = np.uint8(255 * heatmap_highlighted)
    
    # Apply color map
    heatmap_color = cv2.applyColorMap(heatmap_uint8, colormap)
    
    # Create a mask for high-confidence regions
    high_confidence_mask = (heatmap >= threshold).astype(np.float32)
    high_confidence_mask = cv2.resize(high_confidence_mask, (w, h))
    high_confidence_mask = np.expand_dims(high_confidence_mask, axis=2)
    
    # Blend with original image with variable alpha based on confidence
    beta = 1.0 - alpha
    overlay = cv2.addWeighted(original_np, beta, heatmap_color, alpha, 0)
    
    # Add border to high-confidence regions
    high_confidence_regions = (heatmap >= threshold + 0.2)
    if np.any(high_confidence_regions):
        high_confidence_regions = cv2.resize((high_confidence_regions).astype(np.uint8), (w, h))
        contours, _ = cv2.findContours(high_confidence_regions, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(overlay, contours, -1, (255, 255, 255), 2)

    return Image.fromarray(overlay)
