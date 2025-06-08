import cv2
import numpy as np
from PIL import Image
import io
import time
from scipy.fftpack import dct, idct
from scipy.signal import convolve2d
from src.api.utils.logger import logger

def resize_if_needed(image, max_size=1200):
    """Resize image if any dimension is larger than max_size while preserving aspect ratio"""
    width, height = image.size
    if width > max_size or height > max_size:
        if width > height:
            new_width = max_size
            new_height = int(height * (max_size / width))
        else:
            new_height = max_size
            new_width = int(width * (max_size / height))
        return image.resize((new_width, new_height), Image.LANCZOS)
    return image

def detect_pixel_forgery(image, window_size=64, stride=32, threshold=0.55):
    """
    Detect image forgery based on pixel analysis of JPEG blocking artifacts
    
    Args:
        image (PIL.Image): Input image
        window_size (int): Size of analysis window
        stride (int): Step size for sliding window
        threshold (float): Threshold for artifact inconsistency detection
        
    Returns:
        Tuple[PIL.Image, float, list]: Visualization image, confidence score, and detected regions
    """
    start_time = time.time()
    logger.info("Starting pixel-based forgery detection")
    
    # Resize image if needed
    image = resize_if_needed(image)
    
    # Convert PIL image to OpenCV format
    img_np = np.array(image)
    
    # Convert to grayscale if the image is color
    if len(img_np.shape) == 3:
        gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
    else:
        gray = img_np.copy()
    
    height, width = gray.shape
    
    # Create a visualization image and heat map
    vis_img = img_np.copy()
    heat_map = np.zeros_like(gray, dtype=np.float32)
    
    # Cross-difference filter to detect blocking artifacts
    cross_diff_filter = np.array([
        [0, 0, 0, 0, 0],
        [0, -1, 2, -1, 0],
        [0, 2, -4, 2, 0],
        [0, -1, 2, -1, 0],
        [0, 0, 0, 0, 0]
    ])
    
    # Detect JPEG blocking artifacts
    block_map = detect_blocking_artifacts(gray)
    
    # Slide a window across the image
    detected_regions = []
    
    for y in range(0, height - window_size, stride):
        for x in range(0, width - window_size, stride):
            # Extract window
            window = block_map[y:y+window_size, x:x+window_size]
            
            # Compute inconsistency score within the window
            score = analyze_window(window)
            
            # If score exceeds threshold, mark as potential forgery
            if score > threshold:
                # Record the region
                detected_regions.append({
                    "x": int(x),
                    "y": int(y),
                    "width": int(window_size),
                    "height": int(window_size),
                    "score": float(score)
                })
                
                # Add to heat map
                heat_map[y:y+window_size, x:x+window_size] = np.maximum(
                    heat_map[y:y+window_size, x:x+window_size], 
                    score
                )
    
    # Normalize heat map
    if np.max(heat_map) > 0:
        heat_map = heat_map / np.max(heat_map)
    
    # Generate visualization
    for region in detected_regions:
        x, y, w, h = region["x"], region["y"], region["width"], region["height"]
        score = region["score"]
        
        # Color intensity based on score
        intensity = int(score * 255)
        color = (0, 0, intensity) if len(img_np.shape) == 3 else intensity
        
        # Draw rectangle on visualization image
        if len(img_np.shape) == 3:
            cv2.rectangle(vis_img, (x, y), (x + w, y + h), (0, 0, 255), 2)
        else:
            cv2.rectangle(vis_img, (x, y), (x + w, y + h), 255, 2)
    
    # Apply color map to heat map for visualization
    if len(img_np.shape) == 3:
        heat_colored = cv2.applyColorMap((heat_map * 255).astype(np.uint8), cv2.COLORMAP_JET)
        alpha = 0.5
        vis_with_heat = cv2.addWeighted(vis_img, 1 - alpha, heat_colored, alpha, 0)
    else:
        heat_colored = (heat_map * 255).astype(np.uint8)
        vis_with_heat = cv2.addWeighted(vis_img, 0.5, heat_colored, 0.5, 0)
    
    # Calculate overall confidence based on detected regions
    confidence = 0.0
    if detected_regions:
        # Weight by area coverage and scores
        total_area = height * width
        weighted_scores = sum(r["width"] * r["height"] * r["score"] for r in detected_regions)
        covered_area = sum(r["width"] * r["height"] for r in detected_regions)
        
        # Combine area coverage and average score
        coverage_factor = min(1.0, covered_area / total_area * 4)  # Amplify small detections
        avg_score = weighted_scores / covered_area if covered_area > 0 else 0
        
        confidence = min(0.95, 0.3 + coverage_factor * 0.3 + avg_score * 0.4)
    
    elapsed_time = time.time() - start_time
    logger.info(f"Pixel-based forgery detection completed in {elapsed_time:.2f} seconds with confidence {confidence:.2f}")
    
    return Image.fromarray(vis_with_heat), confidence, detected_regions

def detect_blocking_artifacts(gray_img):
    """
    Detect JPEG blocking artifacts in the image
    
    Args:
        gray_img (numpy.ndarray): Grayscale image
        
    Returns:
        numpy.ndarray: Map of blocking artifacts
    """
    # Create kernels to detect horizontal and vertical edges at block boundaries
    h_kernel = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    v_kernel = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])
    
    # Apply kernels
    h_edges = np.abs(convolve2d(gray_img, h_kernel, mode='same', boundary='symm'))
    v_edges = np.abs(convolve2d(gray_img, v_kernel, mode='same', boundary='symm'))
    
    # Calculate edge strength
    edge_strength = h_edges + v_edges
    
    # Create a blocking artifact map
    artifact_map = np.zeros_like(gray_img, dtype=np.float32)
    
    # JPEG uses 8x8 blocks for compression
    block_size = 8
    height, width = gray_img.shape
    
    # Iterate through the image and detect blocking artifacts
    for y in range(0, height - block_size, block_size):
        for x in range(0, width - block_size, block_size):
            # Check horizontal boundary
            if x + block_size < width:
                boundary_h = edge_strength[y:y+block_size, x+block_size-1:x+block_size+1]
                artifact_map[y:y+block_size, x+block_size-1:x+block_size+1] = np.maximum(
                    artifact_map[y:y+block_size, x+block_size-1:x+block_size+1],
                    np.mean(boundary_h)
                )
            
            # Check vertical boundary
            if y + block_size < height:
                boundary_v = edge_strength[y+block_size-1:y+block_size+1, x:x+block_size]
                artifact_map[y+block_size-1:y+block_size+1, x:x+block_size] = np.maximum(
                    artifact_map[y+block_size-1:y+block_size+1, x:x+block_size],
                    np.mean(boundary_v)
                )
    
    return artifact_map

def analyze_window(window):
    """
    Analyze a window for inconsistency in blocking artifacts
    
    Args:
        window (numpy.ndarray): Window from the blocking artifact map
        
    Returns:
        float: Inconsistency score
    """
    # Get window dimensions
    h, w = window.shape
    
    # Calculate statistics
    mean_val = np.mean(window)
    std_val = np.std(window)
    
    # Look for unusual patterns in the blocking artifacts
    # Higher std indicates inconsistency
    if mean_val > 0:
        return min(1.0, std_val / mean_val)
    else:
        return 0.0

def detect_jpeg_grid(image):
    """
    Detect JPEG grid inconsistencies for forgery detection
    This is a simplified version based on the paper "Local JPEG Grid Detector via Blocking Artifacts"
    
    Args:
        image (PIL.Image): Input image
        
    Returns:
        Tuple[PIL.Image, float, list]: Visualization image, confidence score, and detected regions
    """
    # Use the more generic pixel-based detection method
    return detect_pixel_forgery(image) 