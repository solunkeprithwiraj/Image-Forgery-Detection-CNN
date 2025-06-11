import numpy as np
import cv2
from PIL import Image
import io
import math
from scipy.fftpack import dct, idct
from src.api.utils.logger import logger
import base64

def detect_frequency_artifacts(image_data):
    """
    Detect frequency domain artifacts that indicate manipulation
    
    This technique analyzes DCT coefficients in JPEG images to find
    inconsistencies that suggest tampering, similar to techniques used
    by professional forensic tools.
    
    :param image_data: PIL Image, file path, bytes or file-like object
    :return: Tuple of (is_tampered, confidence, result_img_bytes, highlighted_regions)
    """
    logger.info("Analyzing frequency domain artifacts")
    
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
    
    # Get image dimensions
    h, w = gray.shape
    
    # Make dimensions divisible by 8 (JPEG block size)
    h_blocks = h // 8
    w_blocks = w // 8
    
    # Crop to multiple of 8
    gray = gray[:h_blocks*8, :w_blocks*8]
    
    # Analyze JPEG blocks
    block_scores = np.zeros((h_blocks, w_blocks))
    
    # Apply DCT transform to each 8x8 block
    for i in range(h_blocks):
        for j in range(w_blocks):
            # Extract 8x8 block
            block = gray[i*8:(i+1)*8, j*8:(j+1)*8].astype(float)
            
            # Apply DCT-II
            dct_block = dct(dct(block.T, norm='ortho').T, norm='ortho')
            
            # Calculate statistics on DCT coefficients
            # We're looking for abnormalities in the distribution
            
            # 1. Quantization artifact detection
            # JPEG quantizes DCT coefficients, which leaves specific patterns
            block_scores[i, j] = analyze_dct_block(dct_block)
    
    # Normalize scores
    if np.max(block_scores) > np.min(block_scores):
        block_scores = (block_scores - np.min(block_scores)) / (np.max(block_scores) - np.min(block_scores))
    
    # Calculate global statistics
    mean_score = np.mean(block_scores)
    std_score = np.std(block_scores)
    
    # Detect anomalies (blocks with scores significantly different from others)
    threshold = mean_score + 2 * std_score
    suspicious_blocks = block_scores > threshold
    
    # Create visualization and gather results
    highlighted_regions = []
    
    # Create heatmap for visualization
    heatmap = np.zeros((h, w, 3), dtype=np.uint8)
    
    for i in range(h_blocks):
        for j in range(w_blocks):
            score = block_scores[i, j]
            
            # Colorize block based on score (blue-green-red)
            # Higher score = more suspicious
            r = min(int(score * 255 * 2), 255)
            g = min(int((1 - score) * 255 * 2), 255) if score < 0.5 else 0
            b = 0
            
            # Fill block in heatmap
            heatmap[i*8:(i+1)*8, j*8:(j+1)*8] = (b, g, r)
            
            # Mark suspicious blocks
            if suspicious_blocks[i, j]:
                y1, y2 = i * 8, (i + 1) * 8
                x1, x2 = j * 8, (j + 1) * 8
                
                # Add rectangular region to the list
                highlighted_regions.append({
                    'x': int(x1),
                    'y': int(y1),
                    'width': int(x2 - x1),
                    'height': int(y2 - y1),
                    'score': float(score)
                })
                
                # Draw rectangle on result image
                cv2.rectangle(result_img, (x1, y1), (x2, y2), (0, 0, 255), 1)
    
    # Blend heatmap with original image
    alpha = 0.4
    blended = cv2.addWeighted(img[:h_blocks*8, :w_blocks*8], 1-alpha, heatmap, alpha, 0)
    
    # Calculate confidence based on suspicious blocks
    if np.sum(suspicious_blocks) > 0:
        # Calculate confidence based on number and intensity of suspicious blocks
        suspicious_count = np.sum(suspicious_blocks)
        total_blocks = h_blocks * w_blocks
        suspicious_ratio = suspicious_count / total_blocks
        
        # Average score of suspicious blocks
        suspicious_scores = block_scores[suspicious_blocks]
        avg_suspicious_score = np.mean(suspicious_scores) if len(suspicious_scores) > 0 else 0
        
        # Weighted confidence score
        confidence = min(0.3 + suspicious_ratio * 0.4 + avg_suspicious_score * 0.5, 1.0)
    else:
        confidence = 0.1  # Low confidence if no suspicious blocks
    
    # Determine if the image is likely tampered
    is_tampered = confidence > 0.5
    
    # Create result image with suspicious regions highlighted
    for region in highlighted_regions:
        x, y, w, h = region['x'], region['y'], region['width'], region['height']
        score = region['score']
        
        # Color based on score (higher score = more red)
        color_intensity = int(score * 255)
        cv2.rectangle(blended, (x, y), (x + w, y + h), (0, 255 - color_intensity, color_intensity), 2)
    
    # Convert result image to bytes for returning
    result_pil = Image.fromarray(blended)
    result_buffer = io.BytesIO()
    result_pil.save(result_buffer, format="PNG")
    result_buffer.seek(0)
    result_bytes = result_buffer.getvalue()
    
    # Convert to base64 for API response
    result_base64 = base64.b64encode(result_bytes).decode('utf-8')
    
    logger.info(f"Frequency analysis complete with {len(highlighted_regions)} suspicious regions, confidence: {confidence:.2f}")
    return is_tampered, confidence, result_base64, highlighted_regions

def analyze_dct_block(dct_block):
    """
    Analyze a DCT block for signs of manipulation
    
    :param dct_block: 8x8 DCT coefficient block
    :return: Suspiciousness score (0-1)
    """
    # Extract the AC coefficients (DC is at [0,0])
    ac_coeffs = dct_block.copy()
    ac_coeffs[0, 0] = 0
    
    # Several tests for suspicious patterns
    
    # 1. Zero coefficient ratio - many zeros can indicate double compression
    zero_ratio = np.sum(np.abs(ac_coeffs) < 0.01) / 63  # 63 = total AC coeffs
    
    # 2. Check for unusual high-frequency components
    # High frequency is bottom-right region of DCT block
    high_freq = ac_coeffs[4:, 4:]
    high_freq_energy = np.sum(np.abs(high_freq))
    
    # 3. Check for abrupt changes in coefficient distributions
    # This can reveal doctored images
    coeff_histogram, _ = np.histogram(ac_coeffs, bins=10, range=(-1, 1))
    coeff_histogram = coeff_histogram / np.sum(coeff_histogram)
    
    # Calculate histogram smoothness - less smooth suggests tampering
    hist_diff = np.abs(np.diff(coeff_histogram))
    hist_smoothness = 1.0 - np.mean(hist_diff)
    
    # 4. Compare coefficient ratios against expected JPEG distributions
    # Natural JPEG compression has specific patterns
    low_freq = ac_coeffs[:4, :4]
    low_freq_energy = np.sum(np.abs(low_freq))
    
    # Calculate energy ratio
    if low_freq_energy > 0:
        energy_ratio = high_freq_energy / low_freq_energy
    else:
        energy_ratio = 0
    
    # Combine metrics into a single score
    # Weights determined empirically
    score = (
        0.3 * (1 - zero_ratio) +  # Fewer zeros = higher score
        0.3 * min(energy_ratio * 5, 1.0) +  # Higher energy ratio = higher score
        0.4 * (1 - hist_smoothness)  # Less smooth histogram = higher score
    )
    
    return score 