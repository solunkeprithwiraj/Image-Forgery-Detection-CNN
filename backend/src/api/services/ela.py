from PIL import Image, ImageChops, ImageEnhance, ImageDraw, ImageFilter
import os
import uuid
import numpy as np
import cv2
from typing import Tuple, Optional, List

def generate_ela_image(image: Image.Image, quality: int = 85, resize_factor: float = 0.5, 
                      enhance_contrast: bool = True, 
                      colorize: bool = False) -> Image.Image:
    """
    Generate an Enhanced Error Level Analysis (ELA) image from the given input image.

    :param image: PIL Image object
    :param quality: JPEG compression quality for simulating loss (default: 85)
    :param resize_factor: Factor to resize image (0.5 means 50% smaller)
    :param enhance_contrast: Whether to enhance contrast in the ELA image
    :param colorize: Whether to apply color mapping for better visualization
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
    
    # Optional contrast enhancement for better visualization
    if enhance_contrast:
        ela_image = ImageEnhance.Contrast(ela_image).enhance(1.5)
        
    # Apply colorization for better visualization if requested
    if colorize:
        # Convert to numpy array for OpenCV processing
        ela_np = np.array(ela_image)
        
        # Apply a viridis-like colormap which is good for ELA visualization
        # Convert to grayscale first
        ela_gray = cv2.cvtColor(ela_np, cv2.COLOR_RGB2GRAY)
        ela_color = cv2.applyColorMap(ela_gray, cv2.COLORMAP_VIRIDIS)
        
        # Convert back to PIL
        ela_image = Image.fromarray(ela_color)

    # Cleanup
    os.remove(temp_filename)

    return ela_image

def generate_ela_enhanced(image: Image.Image, quality: int = 85, 
                         highlight_threshold: int = 15) -> Image.Image:
    """
    Generate an enhanced ELA image with suspicious areas highlighted.
    
    :param image: PIL Image object
    :param quality: JPEG compression quality (default: 85)
    :param highlight_threshold: Threshold for highlighting suspicious areas (0-255)
    :return: Enhanced ELA image with suspicious areas highlighted
    """
    # Generate basic ELA image
    ela_image = generate_ela_image(image, quality, resize_factor=1.0, enhance_contrast=True)
    
    # Convert to numpy array
    ela_np = np.array(ela_image)
    
    # Convert to grayscale for processing
    if len(ela_np.shape) == 3:
        ela_gray = cv2.cvtColor(ela_np, cv2.COLOR_RGB2GRAY)
    else:
        ela_gray = ela_np
    
    # Find potentially manipulated regions (high error areas)
    _, binary = cv2.threshold(ela_gray, highlight_threshold, 255, cv2.THRESH_BINARY)
    
    # Apply morphological operations to clean up the mask
    kernel = np.ones((3, 3), np.uint8)
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=2)
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=1)
    
    # Find contours of suspicious regions
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Convert to colorized ELA for better visualization
    ela_color = cv2.applyColorMap(ela_gray, cv2.COLORMAP_JET)
    
    # Draw contours around suspicious regions
    significant_regions = []
    
    for contour in contours:
        area = cv2.contourArea(contour)
        
        # Skip very small regions
        if area < 100:
            continue
            
        # Draw contour
        cv2.drawContours(ela_color, [contour], -1, (255, 255, 255), 2)
        
        # Add to significant regions
        x, y, w, h = cv2.boundingRect(contour)
        significant_regions.append((x, y, w, h, area))
    
    # Add text annotation
    if significant_regions:
        cv2.putText(ela_color, f"Found {len(significant_regions)} suspicious regions", 
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    
    return Image.fromarray(ela_color)

def generate_ela_comparison(image: Image.Image, qualities: List[int] = [50, 75, 85, 95]) -> Image.Image:
    """
    Generate a comparison of ELA images at different JPEG qualities.
    
    :param image: PIL Image object
    :param qualities: List of JPEG qualities to compare
    :return: A grid image showing ELA at different qualities
    """
    # Resize input image to make the grid manageable
    max_size = 800
    orig_width, orig_height = image.size
    
    scale_factor = 0.5
    if max(orig_width, orig_height) > max_size:
        scale_factor = max_size / max(orig_width, orig_height)
    
    width = int(orig_width * scale_factor)
    height = int(orig_height * scale_factor)
    
    resized_image = image.resize((width, height), Image.LANCZOS)
    
    # Generate ELA images at different qualities
    ela_images = []
    
    for quality in qualities:
        ela = generate_ela_image(resized_image, quality=quality, resize_factor=1.0, 
                              enhance_contrast=True, colorize=True)
        
        # Add quality label
        ela_np = np.array(ela)
        cv2.putText(ela_np, f"Quality: {quality}", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        ela_images.append(Image.fromarray(ela_np))
    
    # Create a 2x2 grid
    grid_cols = 2
    grid_rows = (len(ela_images) + 1) // 2  # +1 for original image
    
    grid_width = width * grid_cols
    grid_height = height * grid_rows
    
    grid_image = Image.new('RGB', (grid_width, grid_height))
    
    # Add the original image with label
    resized_np = np.array(resized_image)
    cv2.putText(resized_np, "Original Image", (10, 30), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    grid_image.paste(Image.fromarray(resized_np), (0, 0))
    
    # Add ELA images
    for i, ela_img in enumerate(ela_images):
        row = (i + 1) // grid_cols
        col = (i + 1) % grid_cols
        grid_image.paste(ela_img, (col * width, row * height))
    
    return grid_image

def generate_ela_zoom(image: Image.Image, quality: int = 85, 
                    highlight_threshold: int = 15) -> Optional[Image.Image]:
    """
    Generate a zoomed view of the most suspicious region in the ELA image.
    
    :param image: PIL Image object
    :param quality: JPEG compression quality
    :param highlight_threshold: Threshold for highlighting suspicious areas
    :return: Zoomed detail view of most suspicious region, or None if no significant region found
    """
    # Generate basic ELA image at full resolution
    ela_image = generate_ela_image(image, quality, resize_factor=1.0, enhance_contrast=True)
    
    # Convert to numpy array
    ela_np = np.array(ela_image)
    
    # Convert to grayscale for processing
    if len(ela_np.shape) == 3:
        ela_gray = cv2.cvtColor(ela_np, cv2.COLOR_RGB2GRAY)
    else:
        ela_gray = ela_np
    
    # Find potentially manipulated regions (high error areas)
    _, binary = cv2.threshold(ela_gray, highlight_threshold, 255, cv2.THRESH_BINARY)
    
    # Apply morphological operations to clean up the mask
    kernel = np.ones((3, 3), np.uint8)
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=2)
    
    # Find contours of suspicious regions
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        return None
    
    # Find the largest contour
    largest_contour = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(largest_contour)
    
    # Ensure minimum size for zoom region
    min_size = 100
    if w < min_size:
        center_x = x + w // 2
        x = max(0, center_x - min_size // 2)
        w = min_size
    if h < min_size:
        center_y = y + h // 2
        y = max(0, center_y - min_size // 2)
        h = min_size
    
    # Add padding for context (20% on each side)
    padding_x = int(w * 0.2)
    padding_y = int(h * 0.2)
    
    # Calculate region with padding, ensuring we stay within image bounds
    orig_width, orig_height = ela_image.size
    x1 = max(0, x - padding_x)
    y1 = max(0, y - padding_y)
    x2 = min(orig_width, x + w + padding_x)
    y2 = min(orig_height, y + h + padding_y)
    
    # Extract the region of interest from both original and ELA
    original_roi = image.crop((x1, y1, x2, y2))
    ela_roi = ela_image.crop((x1, y1, x2, y2))
    
    # Create a side-by-side comparison
    roi_width, roi_height = original_roi.size
    comparison = Image.new('RGB', (roi_width * 2, roi_height))
    
    # Add original and ELA side by side
    comparison.paste(original_roi, (0, 0))
    
    # Colorize the ELA zoom for better visualization
    ela_roi_np = np.array(ela_roi)
    ela_roi_color = cv2.applyColorMap(cv2.cvtColor(ela_roi_np, cv2.COLOR_RGB2GRAY), cv2.COLORMAP_JET)
    comparison.paste(Image.fromarray(ela_roi_color), (roi_width, 0))
    
    # Add labels and border
    draw = ImageDraw.Draw(comparison)
    draw.rectangle([(0, 0), (roi_width * 2 - 1, roi_height - 1)], outline=(255, 255, 255), width=2)
    draw.rectangle([(0, 0), (roi_width - 1, roi_height - 1)], outline=(255, 255, 255), width=1)
    draw.rectangle([(roi_width, 0), (roi_width * 2 - 1, roi_height - 1)], outline=(255, 255, 255), width=1)
    
    # Convert to numpy for text
    comparison_np = np.array(comparison)
    cv2.putText(comparison_np, "Original Detail", (10, 25), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
    cv2.putText(comparison_np, "ELA Detail", (roi_width + 10, 25), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
    
    return Image.fromarray(comparison_np)
