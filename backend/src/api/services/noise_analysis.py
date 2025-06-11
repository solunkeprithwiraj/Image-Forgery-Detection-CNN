import numpy as np
import cv2
from PIL import Image
import io
from scipy.signal import wiener
from scipy.stats import entropy
from skimage.feature import local_binary_pattern
from src.api.utils.logger import logger

def analyze_noise_patterns(image_data):
    """
    Analyze noise patterns in an image to detect forgery
    
    This technique looks for inconsistencies in noise patterns which often 
    indicate manipulation, similar to the approach used by FotoForensics.
    
    :param image_data: PIL Image, file path, bytes or file-like object
    :return: Tuple of (result_image, confidence, highlighted_regions)
    """
    logger.info("Analyzing noise patterns in image")
    
    # Open the image if needed
    if isinstance(image_data, str):
        # It's a file path
        img = np.array(Image.open(image_data).convert('RGB'))
    elif isinstance(image_data, bytes):
        # It's raw bytes
        img = np.array(Image.open(io.BytesIO(image_data)).convert('RGB'))
    elif hasattr(image_data, 'read') and hasattr(image_data, 'seek'):
        # It's a file-like object
        pos = image_data.tell()
        img = np.array(Image.open(image_data).convert('RGB'))
        image_data.seek(pos)
    elif isinstance(image_data, Image.Image):
        # It's already a PIL Image
        img = np.array(image_data.convert('RGB'))
    else:
        # Try to use it directly as a numpy array
        img = np.array(image_data)
    
    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    
    # Create a result image for visualization
    result_img = img.copy()
    
    # Extract noise using the noise residual method
    noise_residual = extract_noise_residual(gray)
    
    # Normalize for better visualization
    noise_normalized = cv2.normalize(noise_residual, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    
    # Calculate local binary patterns for texture inconsistency detection
    radius = 1
    n_points = 8 * radius
    lbp = local_binary_pattern(gray, n_points, radius, method='uniform')
    
    # Create a grid of blocks to analyze
    block_size = 32
    h, w = gray.shape
    blocks_h = h // block_size
    blocks_w = w // block_size
    
    # For each block, calculate noise statistics
    noise_stats = np.zeros((blocks_h, blocks_w, 3))  # 3 features: std, entropy, lbp_entropy
    
    for i in range(blocks_h):
        for j in range(blocks_w):
            # Extract block
            y1, y2 = i * block_size, (i + 1) * block_size
            x1, x2 = j * block_size, (j + 1) * block_size
            
            block_noise = noise_residual[y1:y2, x1:x2]
            block_lbp = lbp[y1:y2, x1:x2]
            
            # Calculate statistics
            noise_stats[i, j, 0] = np.std(block_noise)
            noise_stats[i, j, 1] = entropy(block_noise.flatten())
            
            # Calculate LBP histogram for texture analysis
            hist, _ = np.histogram(block_lbp.ravel(), bins=np.arange(0, n_points + 3), density=True)
            noise_stats[i, j, 2] = entropy(hist)
    
    # Detect inconsistencies by finding outliers
    # Flatten to analyze all blocks together
    noise_stats_flat = noise_stats.reshape(-1, 3)
    
    # Calculate Z-scores for each feature
    z_scores = np.abs((noise_stats_flat - np.mean(noise_stats_flat, axis=0)) / (np.std(noise_stats_flat, axis=0) + 1e-10))
    
    # Mark blocks with high Z-scores as suspicious
    # Higher threshold = fewer suspicious blocks
    threshold = 2.0  # Z-score threshold (2.0 = ~5% of blocks flagged as outliers)
    suspicious_blocks = np.any(z_scores > threshold, axis=1).reshape(blocks_h, blocks_w)
    
    # Create a heatmap of suspicious regions
    heatmap = np.zeros((h, w), dtype=np.uint8)
    highlighted_regions = []
    
    for i in range(blocks_h):
        for j in range(blocks_w):
            if suspicious_blocks[i, j]:
                y1, y2 = i * block_size, (i + 1) * block_size
                x1, x2 = j * block_size, (j + 1) * block_size
                
                # Add to heatmap
                heatmap[y1:y2, x1:x2] = 255
                
                # Add rectangular region
                highlighted_regions.append({
                    'x': int(x1),
                    'y': int(y1),
                    'width': int(x2 - x1),
                    'height': int(y2 - y1),
                    'score': float(np.max(z_scores.reshape(blocks_h, blocks_w, 3)[i, j]))
                })
                
                # Mark in the result image
                cv2.rectangle(result_img, (x1, y1), (x2, y2), (0, 0, 255), 2)
    
    # Apply a colormap to the noise for visualization
    noise_colored = cv2.applyColorMap(noise_normalized, cv2.COLORMAP_JET)
    
    # Create the final visualization by blending
    alpha = 0.4
    blended = cv2.addWeighted(img, 1 - alpha, noise_colored, alpha, 0)
    
    # Calculate confidence score (higher score = more likely to be tampered)
    if len(highlighted_regions) > 0:
        # Calculate weighted score based on the number of suspicious blocks and their z-scores
        total_blocks = blocks_h * blocks_w
        suspicious_count = np.sum(suspicious_blocks)
        suspicious_ratio = suspicious_count / total_blocks
        
        # Get maximum z-score as an indication of how anomalous the outliers are
        max_zscore = np.max(z_scores) if len(z_scores) > 0 else 0
        
        # Combined confidence score (0-1 range)
        # Weight the ratio more if we have strong outliers
        if max_zscore > 3.0:  # Very strong outliers
            confidence = min(0.5 + suspicious_ratio + (max_zscore - 3) * 0.1, 1.0)
        else:
            confidence = min(0.3 + suspicious_ratio * 0.7 + max_zscore * 0.1, 1.0)
    else:
        confidence = 0.1  # Low confidence if no suspicious regions
    
    # Add a heatmap overlay
    overlay = result_img.copy()
    for i, region in enumerate(highlighted_regions):
        x, y, w, h = region['x'], region['y'], region['width'], region['height']
        score = region['score']
        
        # Color based on score (green to red)
        color_r = min(int(score * 255), 255)
        color_g = min(int((3.0 - score) * 80), 255) if score < 3.0 else 0
        color_b = 0  # Blue component is zero
        cv2.rectangle(overlay, (x, y), (x + w, y + h), (color_b, color_g, color_r), -1)
    
    # Add the overlay to the result image
    cv2.addWeighted(overlay, 0.3, result_img, 0.7, 0, result_img)
    
    # Convert back to PIL Image
    result_pil = Image.fromarray(result_img)
    
    logger.info(f"Noise analysis complete with {len(highlighted_regions)} suspicious regions, confidence: {confidence:.2f}")
    return result_pil, confidence, highlighted_regions

def extract_noise_residual(gray_img):
    """
    Extract noise residual from an image using Wiener filtering
    
    :param gray_img: Grayscale image as numpy array
    :return: Noise residual as numpy array
    """
    # Apply Wiener filter to remove noise (this creates a smoothed version)
    filtered_img = wiener(gray_img, (5, 5))
    
    # Normalize to same range as input
    filtered_img = filtered_img.astype(np.float32)
    
    # Calculate noise residual (original - filtered)
    noise = gray_img.astype(np.float32) - filtered_img
    
    return noise

def detect_noise_forgery(image_data):
    """
    Detect forgery using noise analysis
    
    :param image_data: PIL Image, file path, bytes or file-like object
    :return: Tuple of (result_image, confidence, is_tampered)
    """
    result_img, confidence, regions = analyze_noise_patterns(image_data)
    
    # Determine if the image is tampered based on confidence
    is_tampered = confidence > 0.5
    
    return result_img, confidence, is_tampered, regions 