import cv2
import numpy as np
from PIL import Image
import io
import time
import os
from tempfile import NamedTemporaryFile
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

def detect_double_jpeg_histogram(image):
    """
    Detect double JPEG compression by analyzing DCT coefficient histograms
    
    Args:
        image (PIL.Image): Input image
        
    Returns:
        Tuple[PIL.Image, float, dict]: Visualization image, confidence score, and analysis details
    """
    start_time = time.time()
    logger.info("Starting double JPEG compression detection using histogram analysis")
    
    # Resize image if needed
    image = resize_if_needed(image)
    
    # Convert to grayscale
    gray_img = image.convert('L')
    
    # Convert PIL image to OpenCV format
    img_np = np.array(gray_img)
    
    # Calculate the DCT transform
    dct = cv2.dct(np.float32(img_np))
    
    # Take the absolute values
    dct_abs = np.abs(dct)
    
    # Calculate histograms for some DCT coefficients
    histograms = {}
    bin_edges = {}
    
    # Check specific DCT coefficients that are most affected by JPEG compression
    # These are typically the low-frequency coefficients
    coef_indices = [(1, 2), (2, 1), (3, 0), (0, 3)]
    
    for i, j in coef_indices:
        # Extract coefficient values
        coef = dct_abs[i::8, j::8].flatten()
        
        # Create histogram
        hist, bins = np.histogram(coef, bins=50, range=(0, 100))
        
        histograms[(i, j)] = hist
        bin_edges[(i, j)] = bins
    
    # Calculate histogram characteristics that indicate double compression
    # In particular, we look for periodic patterns and peaks
    periodicity_scores = {}
    peak_ratios = {}
    
    for idx, hist in histograms.items():
        # Calculate autocorrelation to detect periodicity
        autocorr = np.correlate(hist, hist, mode='full')
        autocorr = autocorr[len(hist)-1:]
        autocorr = autocorr / autocorr[0]  # Normalize
        
        # Calculate periodicity score (sum of autocorrelation peaks)
        periodicity_scores[idx] = np.sum(autocorr[1:20]) / 19
        
        # Calculate peak ratio (ratio of highest peak to average value)
        if np.mean(hist) > 0:
            peak_ratios[idx] = np.max(hist) / np.mean(hist)
        else:
            peak_ratios[idx] = 1.0
    
    # Calculate overall confidence score
    avg_periodicity = np.mean(list(periodicity_scores.values()))
    avg_peak_ratio = np.mean(list(peak_ratios.values()))
    
    # Higher periodicity and peak ratio indicate double compression
    # Scale to 0-1 range
    periodicity_factor = min(1.0, avg_periodicity / 0.4)
    peak_factor = min(1.0, (avg_peak_ratio - 2) / 8)
    
    # Combined confidence score
    confidence = 0.7 * periodicity_factor + 0.3 * peak_factor
    confidence = min(0.95, max(0.05, confidence))
    
    # Create visualization image - we'll use a histogram plot
    hist_img = np.ones((400, 600, 3), dtype=np.uint8) * 255
    
    # Draw histograms for each coefficient
    colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0)]
    
    max_hist_val = 0
    for hist in histograms.values():
        max_val = np.max(hist)
        if max_val > max_hist_val:
            max_hist_val = max_val
    
    # Draw each histogram
    for i, (idx, hist) in enumerate(histograms.items()):
        color = colors[i % len(colors)]
        bins = bin_edges[idx]
        
        # Scale histogram for visualization
        scaled_hist = hist * 300 / max_hist_val if max_hist_val > 0 else hist
        
        # Draw histogram bars
        for j in range(len(hist)):
            height = int(scaled_hist[j])
            x1 = 50 + j * 10
            y1 = 350
            x2 = x1 + 8
            y2 = y1 - height
            cv2.rectangle(hist_img, (x1, y1), (x2, y2), color, -1)
    
    # Add detection result text
    if confidence > 0.6:
        cv2.putText(hist_img, "Double JPEG Detected", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 
                    1, (0, 0, 255), 2, cv2.LINE_AA)
    else:
        cv2.putText(hist_img, "Single JPEG Likely", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 
                    1, (0, 255, 0), 2, cv2.LINE_AA)
    
    cv2.putText(hist_img, f"Confidence: {confidence:.2f}", (50, 80), cv2.FONT_HERSHEY_SIMPLEX, 
                0.7, (0, 0, 0), 1, cv2.LINE_AA)
    
    # Add legend
    for i, idx in enumerate(histograms.keys()):
        color = colors[i % len(colors)]
        text = f"DCT({idx[0]},{idx[1]})"
        cv2.putText(hist_img, text, (400, 50 + i * 30), cv2.FONT_HERSHEY_SIMPLEX, 
                    0.6, color, 1, cv2.LINE_AA)
    
    # Prepare result details
    details = {
        "periodicity_scores": {f"{i},{j}": float(periodicity_scores[(i, j)]) for i, j in coef_indices},
        "peak_ratios": {f"{i},{j}": float(peak_ratios[(i, j)]) for i, j in coef_indices},
        "avg_periodicity": float(avg_periodicity),
        "avg_peak_ratio": float(avg_peak_ratio)
    }
    
    elapsed_time = time.time() - start_time
    logger.info(f"Double JPEG detection completed in {elapsed_time:.2f} seconds with confidence {confidence:.2f}")
    
    return Image.fromarray(hist_img), confidence, details

def detect_double_jpeg_ela(image, quality=90):
    """
    Detect double JPEG compression using Error Level Analysis (ELA)
    This utilizes the existing ELA functionality from the project
    
    Args:
        image (PIL.Image): Input image
        quality (int): JPEG quality for ELA
        
    Returns:
        Tuple[PIL.Image, float, dict]: ELA image, confidence score, and analysis details
    """
    start_time = time.time()
    logger.info("Starting double JPEG detection using ELA")
    
    # Resize image if needed
    image = resize_if_needed(image)
    
    # Import the ELA function from the existing module
    from src.api.services.ela import generate_ela_image
    
    # Generate ELA image
    ela_image = generate_ela_image(image, quality=quality, enhance_contrast=True)
    
    # Convert to numpy array for analysis
    ela_np = np.array(ela_image)
    
    # Convert to grayscale if needed
    if len(ela_np.shape) == 3 and ela_np.shape[2] == 3:
        ela_gray = cv2.cvtColor(ela_np, cv2.COLOR_RGB2GRAY)
    else:
        ela_gray = ela_np
    
    # Calculate statistics for the ELA image
    mean_ela = np.mean(ela_gray)
    std_ela = np.std(ela_gray)
    max_ela = np.max(ela_gray)
    
    # Calculate the ratio of pixels with high error (potential tampering)
    high_error_pixels = np.sum(ela_gray > mean_ela + 2 * std_ela)
    total_pixels = ela_gray.size
    high_error_ratio = high_error_pixels / total_pixels if total_pixels > 0 else 0
    
    # Calculate histogram of ELA values
    hist, bins = np.histogram(ela_gray.flatten(), bins=50)
    
    # Check for peaks in the histogram (characteristic of double compression)
    peak_count = 0
    for i in range(1, len(hist) - 1):
        if hist[i] > hist[i-1] and hist[i] > hist[i+1] and hist[i] > np.mean(hist):
            peak_count += 1
    
    # Calculate confidence based on ELA statistics
    # High standard deviation and multiple peaks indicate possible double compression
    if std_ela < 5:  # Very low variance usually means single compression
        std_factor = 0.1
    elif std_ela > 30:  # Very high variance can mean noise or complex image
        std_factor = 0.5
    else:
        std_factor = min(1.0, std_ela / 20)
    
    peak_factor = min(1.0, peak_count / 3)
    
    # Combined confidence score
    confidence = 0.6 * std_factor + 0.4 * peak_factor
    confidence = min(0.95, max(0.05, confidence))
    
    # Double compression often shows as consistent error levels
    # If high_error_ratio is too high, it might be a complex image rather than tampering
    if high_error_ratio > 0.4:
        confidence *= 0.5
    
    # Prepare result details
    details = {
        "mean_ela": float(mean_ela),
        "std_ela": float(std_ela),
        "max_ela": float(max_ela),
        "high_error_ratio": float(high_error_ratio),
        "peak_count": peak_count
    }
    
    elapsed_time = time.time() - start_time
    logger.info(f"Double JPEG ELA detection completed in {elapsed_time:.2f} seconds with confidence {confidence:.2f}")
    
    return ela_image, confidence, details

def detect_double_jpeg(image):
    """
    Detect double JPEG compression using multiple methods
    
    Args:
        image (PIL.Image): Input image
        
    Returns:
        Tuple[PIL.Image, float, dict]: Visualization image, confidence score, and analysis details
    """
    start_time = time.time()
    logger.info("Starting combined double JPEG compression detection")
    
    # Resize image if needed
    image = resize_if_needed(image)
    
    # Run both detection methods
    _, hist_confidence, hist_details = detect_double_jpeg_histogram(image)
    ela_image, ela_confidence, ela_details = detect_double_jpeg_ela(image)
    
    # Combine confidences with more weight to histogram analysis
    combined_confidence = hist_confidence * 0.7 + ela_confidence * 0.3
    
    # Combine details
    details = {
        "histogram_analysis": {
            "confidence": float(hist_confidence),
            **hist_details
        },
        "ela_analysis": {
            "confidence": float(ela_confidence),
            **ela_details
        },
        "combined_confidence": float(combined_confidence)
    }
    
    elapsed_time = time.time() - start_time
    logger.info(f"Combined double JPEG detection completed in {elapsed_time:.2f} seconds with confidence {combined_confidence:.2f}")
    
    # Return the ELA image as visualization
    return ela_image, combined_confidence, details 