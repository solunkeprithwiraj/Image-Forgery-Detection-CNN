import cv2
import numpy as np
from PIL import Image
from skimage.feature import local_binary_pattern
from skimage.filters import gabor
from skimage.measure import shannon_entropy
from scipy.stats import kurtosis, skew
import time
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

def detect_inpainting_texture(image, block_size=16, threshold=0.65):
    """
    Detect inpainting forgery using texture analysis with Local Binary Patterns
    
    Args:
        image (PIL.Image): Input image
        block_size (int): Size of the analysis blocks
        threshold (float): Threshold for anomaly detection
        
    Returns:
        Tuple[PIL.Image, float]: Visualization image and confidence score
    """
    start_time = time.time()
    logger.info("Starting inpainting detection with texture analysis")
    
    # Resize image if needed
    image = resize_if_needed(image)
    
    # Convert PIL image to numpy array
    img_np = np.array(image)
    
    # Convert to RGB if it's in RGBA format
    if img_np.shape[2] == 4:
        img_np = cv2.cvtColor(img_np, cv2.COLOR_RGBA2RGB)
    
    # Convert to grayscale
    gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
    
    # Calculate LBP
    radius = 1
    n_points = 8 * radius
    lbp = local_binary_pattern(gray, n_points, radius, method='uniform')
    
    # Get image dimensions
    height, width = gray.shape
    
    # Create a heatmap for visualization
    heatmap = np.zeros((height, width), dtype=np.float32)
    
    # Analyze blocks and compute LBP histograms
    rows = height // block_size
    cols = width // block_size
    
    # Store block statistics
    block_stats = []
    
    # Process each block
    for i in range(rows):
        for j in range(cols):
            # Extract block
            y = i * block_size
            x = j * block_size
            block_lbp = lbp[y:y+block_size, x:x+block_size]
            
            # Calculate histogram of LBP values
            hist, _ = np.histogram(block_lbp, bins=n_points + 2, range=(0, n_points + 2), density=True)
            
            # Calculate entropy of the histogram
            entropy = shannon_entropy(hist)
            
            # Calculate other statistical measures
            kurt = kurtosis(hist, axis=0, fisher=True)
            skewness = skew(hist, axis=0)
            
            # Store block statistics
            block_stats.append({
                'pos': (x, y),
                'entropy': entropy,
                'kurtosis': kurt,
                'skewness': skewness
            })
            
            # Fill the heatmap with entropy values
            heatmap[y:y+block_size, x:x+block_size] = entropy
    
    # Calculate global statistics
    all_entropies = np.array([b['entropy'] for b in block_stats])
    mean_entropy = np.mean(all_entropies)
    std_entropy = np.std(all_entropies)
    
    # Create a mask for suspicious blocks
    suspicious_mask = np.zeros((height, width), dtype=np.uint8)
    suspicious_blocks = []
    
    # Identify suspicious blocks (anomalies)
    for block in block_stats:
        x, y = block['pos']
        entropy = block['entropy']
        
        # Calculate Z-score
        z_score = abs(entropy - mean_entropy) / (std_entropy + 1e-10)
        
        # Blocks with significantly different entropy are suspicious
        if z_score > threshold:
            suspicious_mask[y:y+block_size, x:x+block_size] = 255
            suspicious_blocks.append((x, y, block_size, block_size))
    
    # Apply morphological operations to clean up the mask
    kernel = np.ones((5, 5), np.uint8)
    suspicious_mask = cv2.morphologyEx(suspicious_mask, cv2.MORPH_CLOSE, kernel)
    suspicious_mask = cv2.morphologyEx(suspicious_mask, cv2.MORPH_OPEN, kernel)
    
    # Find contours in the mask
    contours, _ = cv2.findContours(suspicious_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Create visualization image
    vis_img = img_np.copy()
    
    # Draw contours of suspicious regions
    cv2.drawContours(vis_img, contours, -1, (0, 0, 255), 2)
    
    # Draw rectangles around suspicious blocks
    for x, y, w, h in suspicious_blocks:
        cv2.rectangle(vis_img, (x, y), (x + w, y + h), (255, 0, 0), 1)
    
    # Calculate confidence based on number and size of suspicious regions
    total_blocks = rows * cols
    num_suspicious = len(suspicious_blocks)
    
    if total_blocks == 0:
        confidence = 0.0
    else:
        confidence_ratio = min(0.95, num_suspicious / total_blocks * 5)
        
        # If too many blocks are suspicious, it might be a false positive
        if confidence_ratio > 0.5:
            confidence = max(0.1, 1.0 - confidence_ratio)
        else:
            confidence = confidence_ratio
    
    elapsed_time = time.time() - start_time
    logger.info(f"Inpainting texture detection completed in {elapsed_time:.2f} seconds with confidence {confidence:.2f}")
    
    return Image.fromarray(vis_img), confidence

def detect_inpainting_noise(image, window_size=16, threshold=0.7):
    """
    Detect inpainting forgery by analyzing local noise variance
    
    Args:
        image (PIL.Image): Input image
        window_size (int): Size of the analysis window
        threshold (float): Threshold for anomaly detection
        
    Returns:
        Tuple[PIL.Image, float]: Visualization image and confidence score
    """
    start_time = time.time()
    logger.info("Starting inpainting detection with noise analysis")
    
    # Resize image if needed
    image = resize_if_needed(image)
    
    # Convert PIL image to numpy array
    img_np = np.array(image)
    
    # Convert to RGB if it's in RGBA format
    if img_np.shape[2] == 4:
        img_np = cv2.cvtColor(img_np, cv2.COLOR_RGBA2RGB)
    
    # Convert to grayscale
    gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
    
    # Apply Gaussian blur to reduce noise
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # Extract noise by subtracting blurred image from original
    noise = cv2.absdiff(gray, blurred)
    
    # Get image dimensions
    height, width = gray.shape
    
    # Create a heatmap for visualization
    heatmap = np.zeros((height, width), dtype=np.float32)
    
    # Analyze windows and compute local noise variance
    rows = height // window_size
    cols = width // window_size
    
    # Store window statistics
    window_stats = []
    
    # Process each window
    for i in range(rows):
        for j in range(cols):
            # Extract window
            y = i * window_size
            x = j * window_size
            window_noise = noise[y:y+window_size, x:x+window_size]
            
            # Calculate variance of noise in the window
            variance = np.var(window_noise)
            
            # Store window statistics
            window_stats.append({
                'pos': (x, y),
                'variance': variance
            })
            
            # Fill the heatmap with variance values
            heatmap[y:y+window_size, x:x+window_size] = variance
    
    # Calculate global statistics
    all_variances = np.array([w['variance'] for w in window_stats])
    mean_variance = np.mean(all_variances)
    std_variance = np.std(all_variances)
    
    # Create a mask for suspicious windows
    suspicious_mask = np.zeros((height, width), dtype=np.uint8)
    suspicious_windows = []
    
    # Identify suspicious windows (anomalies)
    for window in window_stats:
        x, y = window['pos']
        variance = window['variance']
        
        # Calculate Z-score
        z_score = abs(variance - mean_variance) / (std_variance + 1e-10)
        
        # Windows with significantly different variance are suspicious
        if z_score > threshold:
            suspicious_mask[y:y+window_size, x:x+window_size] = 255
            suspicious_windows.append((x, y, window_size, window_size))
    
    # Apply morphological operations to clean up the mask
    kernel = np.ones((5, 5), np.uint8)
    suspicious_mask = cv2.morphologyEx(suspicious_mask, cv2.MORPH_CLOSE, kernel)
    suspicious_mask = cv2.morphologyEx(suspicious_mask, cv2.MORPH_OPEN, kernel)
    
    # Find contours in the mask
    contours, _ = cv2.findContours(suspicious_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Create visualization image
    vis_img = img_np.copy()
    
    # Draw contours of suspicious regions
    cv2.drawContours(vis_img, contours, -1, (0, 255, 0), 2)
    
    # Draw rectangles around suspicious windows
    for x, y, w, h in suspicious_windows:
        cv2.rectangle(vis_img, (x, y), (x + w, y + h), (255, 0, 0), 1)
    
    # Calculate confidence based on number and size of suspicious regions
    total_windows = rows * cols
    num_suspicious = len(suspicious_windows)
    
    if total_windows == 0:
        confidence = 0.0
    else:
        confidence_ratio = min(0.95, num_suspicious / total_windows * 5)
        
        # If too many windows are suspicious, it might be a false positive
        if confidence_ratio > 0.5:
            confidence = max(0.1, 1.0 - confidence_ratio)
        else:
            confidence = confidence_ratio
    
    elapsed_time = time.time() - start_time
    logger.info(f"Inpainting noise detection completed in {elapsed_time:.2f} seconds with confidence {confidence:.2f}")
    
    return Image.fromarray(vis_img), confidence

def detect_inpainting_combined(image, texture_weight=0.6, noise_weight=0.4):
    """
    Detect inpainting forgery using a combination of texture and noise analysis
    
    Args:
        image (PIL.Image): Input image
        texture_weight (float): Weight for texture analysis result
        noise_weight (float): Weight for noise analysis result
        
    Returns:
        Tuple[PIL.Image, float]: Visualization image and confidence score
    """
    start_time = time.time()
    logger.info("Starting combined inpainting detection")
    
    # Resize image if needed
    image = resize_if_needed(image)
    
    # Perform texture analysis
    texture_img, texture_confidence = detect_inpainting_texture(image)
    
    # Perform noise analysis
    noise_img, noise_confidence = detect_inpainting_noise(image)
    
    # Combine confidences using weighted average
    combined_confidence = texture_confidence * texture_weight + noise_confidence * noise_weight
    
    # Choose the visualization with higher confidence
    if texture_confidence > noise_confidence:
        result_img = texture_img
    else:
        result_img = noise_img
    
    elapsed_time = time.time() - start_time
    logger.info(f"Combined inpainting detection completed in {elapsed_time:.2f} seconds with confidence {combined_confidence:.2f}")
    
    return result_img, combined_confidence 