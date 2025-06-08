import cv2
import numpy as np
from PIL import Image
import io
import time
from src.api.utils.logger import logger

def resize_image(image, max_size=1200, min_size=None):
    """
    Resize image if any dimension is larger than max_size while preserving aspect ratio
    
    Args:
        image (PIL.Image): Input image
        max_size (int): Maximum size for any dimension
        min_size (int): Minimum size for smaller dimension (optional)
        
    Returns:
        PIL.Image: Resized image
    """
    if not image:
        return None
        
    width, height = image.size
    
    # Check if resize needed for max_size
    needs_resize = False
    if max_size and (width > max_size or height > max_size):
        needs_resize = True
        if width > height:
            new_width = max_size
            new_height = int(height * (max_size / width))
        else:
            new_height = max_size
            new_width = int(width * (max_size / height))
    else:
        new_width, new_height = width, height
    
    # Check if resize needed for min_size
    if min_size and min(new_width, new_height) < min_size:
        needs_resize = True
        if new_width < new_height:
            scale = min_size / new_width
            new_width = min_size
            new_height = int(new_height * scale)
        else:
            scale = min_size / new_height
            new_height = min_size
            new_width = int(new_width * scale)
    
    # Perform resize if needed
    if needs_resize:
        return image.resize((new_width, new_height), Image.LANCZOS)
    return image

def pil_to_cv2(pil_image):
    """
    Convert PIL Image to OpenCV format (numpy array)
    
    Args:
        pil_image (PIL.Image): PIL Image
        
    Returns:
        numpy.ndarray: OpenCV image (BGR format)
    """
    # Convert PIL image to numpy array (RGB)
    rgb_image = np.array(pil_image)
    
    # Convert RGB to BGR (OpenCV format)
    if len(rgb_image.shape) == 3 and rgb_image.shape[2] == 3:
        bgr_image = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR)
        return bgr_image
    elif len(rgb_image.shape) == 3 and rgb_image.shape[2] == 4:
        # Handle RGBA images
        bgr_image = cv2.cvtColor(rgb_image, cv2.COLOR_RGBA2BGR)
        return bgr_image
    else:
        # Grayscale image
        return rgb_image

def cv2_to_pil(cv2_image):
    """
    Convert OpenCV image to PIL Image
    
    Args:
        cv2_image (numpy.ndarray): OpenCV image (BGR format)
        
    Returns:
        PIL.Image: PIL Image (RGB format)
    """
    # Convert BGR to RGB (PIL format)
    if len(cv2_image.shape) == 3 and cv2_image.shape[2] == 3:
        rgb_image = cv2.cvtColor(cv2_image, cv2.COLOR_BGR2RGB)
        return Image.fromarray(rgb_image)
    else:
        # Grayscale image
        return Image.fromarray(cv2_image)

def preprocess_image(image, target_size=None, normalize=False, grayscale=False):
    """
    Preprocess image for model input
    
    Args:
        image (PIL.Image): Input image
        target_size (tuple): Target size as (width, height) or None to preserve size
        normalize (bool): Whether to normalize pixel values to [0,1]
        grayscale (bool): Whether to convert to grayscale
        
    Returns:
        numpy.ndarray: Preprocessed image
    """
    # Resize if target_size is specified
    if target_size:
        image = image.resize(target_size, Image.LANCZOS)
    
    # Convert to grayscale if requested
    if grayscale:
        image = image.convert('L')
        img_array = np.array(image)
        if normalize:
            img_array = img_array / 255.0
    else:
        # Convert to RGB and then to numpy array
        image = image.convert('RGB')
        img_array = np.array(image)
        if normalize:
            img_array = img_array / 255.0
    
    return img_array

def get_image_bytes(image, format='PNG'):
    """
    Convert PIL Image to bytes
    
    Args:
        image (PIL.Image): PIL Image
        format (str): Image format (e.g., 'PNG', 'JPEG')
        
    Returns:
        bytes: Image bytes
    """
    img_byte_arr = io.BytesIO()
    image.save(img_byte_arr, format=format)
    return img_byte_arr.getvalue()

def load_image_from_bytes(image_bytes):
    """
    Load PIL Image from bytes
    
    Args:
        image_bytes (bytes): Image bytes
        
    Returns:
        PIL.Image: PIL Image
    """
    return Image.open(io.BytesIO(image_bytes))

def create_overlay_image(base_image, overlay_image, opacity=0.5):
    """
    Create an overlay of two images with specified opacity
    
    Args:
        base_image (PIL.Image): Base image
        overlay_image (PIL.Image): Overlay image
        opacity (float): Opacity of overlay (0.0 to 1.0)
        
    Returns:
        PIL.Image: Combined image
    """
    # Ensure both images are the same size
    if base_image.size != overlay_image.size:
        overlay_image = overlay_image.resize(base_image.size, Image.LANCZOS)
    
    # Convert to RGBA if needed
    if base_image.mode != 'RGBA':
        base_image = base_image.convert('RGBA')
    if overlay_image.mode != 'RGBA':
        overlay_image = overlay_image.convert('RGBA')
    
    # Create a new image blending the two
    return Image.blend(base_image, overlay_image, opacity)

def add_colorbar_to_image(image, colormap='jet', height=30, vertical=False):
    """
    Add a colorbar to an image
    
    Args:
        image (PIL.Image): Input image
        colormap (str): Colormap name (e.g., 'jet', 'viridis')
        height (int): Height of the colorbar
        vertical (bool): Whether to add a vertical colorbar
        
    Returns:
        PIL.Image: Image with colorbar
    """
    # Get a numpy array from PIL Image
    img_np = np.array(image)
    
    # Create colorbar gradient
    if vertical:
        width = height
        gradient = np.linspace(0, 255, image.height).astype(np.uint8)
        gradient = np.tile(gradient[:, np.newaxis], (1, width))
        colorbar = cv2.applyColorMap(gradient, getattr(cv2, f'COLORMAP_{colormap.upper()}'))
        
        # Create new image with colorbar on the right
        new_width = image.width + width
        new_img = np.zeros((image.height, new_width, 3), dtype=np.uint8)
        new_img[:, :image.width] = img_np
        new_img[:, image.width:] = colorbar
    else:
        gradient = np.linspace(0, 255, image.width).astype(np.uint8)
        gradient = np.tile(gradient[np.newaxis, :], (height, 1))
        colorbar = cv2.applyColorMap(gradient, getattr(cv2, f'COLORMAP_{colormap.upper()}'))
        
        # Create new image with colorbar at the bottom
        new_height = image.height + height
        new_img = np.zeros((new_height, image.width, 3), dtype=np.uint8)
        new_img[:image.height] = img_np
        new_img[image.height:] = colorbar
    
    # Add text labels
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.5
    font_color = (255, 255, 255)
    thickness = 1
    
    if vertical:
        # Add "High" at the top
        cv2.putText(new_img, "High", (image.width + 5, 20), font, font_scale, font_color, thickness)
        # Add "Low" at the bottom
        cv2.putText(new_img, "Low", (image.width + 5, image.height - 10), font, font_scale, font_color, thickness)
    else:
        # Add "Low" on the left
        cv2.putText(new_img, "Low", (10, image.height + height - 10), font, font_scale, font_color, thickness)
        # Add "High" on the right
        cv2.putText(new_img, "High", (image.width - 40, image.height + height - 10), font, font_scale, font_color, thickness)
    
    return Image.fromarray(new_img) 