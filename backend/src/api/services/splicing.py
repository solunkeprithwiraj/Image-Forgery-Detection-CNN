import cv2
import numpy as np
from PIL import Image
import io
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

def detect_splicing_edge_inconsistencies(image, threshold1=50, threshold2=150, dilate_iterations=2):
    """
    Detect splicing forgery by analyzing edge inconsistencies
    
    Args:
        image (PIL.Image): Input image
        threshold1 (int): First threshold for Canny edge detector
        threshold2 (int): Second threshold for Canny edge detector
        dilate_iterations (int): Number of dilation iterations
        
    Returns:
        Tuple[PIL.Image, float]: Visualization image and confidence score
    """
    start_time = time.time()
    logger.info("Starting splicing detection with edge inconsistency analysis")
    
    # Resize image if needed
    image = resize_if_needed(image)
    
    # Convert PIL image to OpenCV format
    img_np = np.array(image)
    
    # Convert to RGB if it's in RGBA format
    if img_np.shape[2] == 4:
        img_np = cv2.cvtColor(img_np, cv2.COLOR_RGBA2RGB)
    
    # Convert to grayscale
    gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
    
    # Apply Gaussian blur to reduce noise
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # Detect edges using Canny edge detector
    edges = cv2.Canny(blurred, threshold1, threshold2)
    
    # Dilate edges to connect broken edges
    kernel = np.ones((3, 3), np.uint8)
    dilated_edges = cv2.dilate(edges, kernel, iterations=dilate_iterations)
    
    # Find contours in the edge image
    contours, _ = cv2.findContours(dilated_edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Create a visualization image
    vis_img = img_np.copy()
    
    # Draw all contours
    cv2.drawContours(vis_img, contours, -1, (0, 255, 0), 2)
    
    # Filter contours by area
    min_area = 100  # Minimum area threshold
    suspicious_contours = []
    
    for contour in contours:
        area = cv2.contourArea(contour)
        if area > min_area:
            # Check if contour is regular or irregular using approx polygon
            perimeter = cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, 0.04 * perimeter, True)
            
            # Regular shapes have fewer vertices
            if len(approx) > 4 and len(approx) < 10:
                suspicious_contours.append(contour)
    
    # Draw suspicious contours in red
    cv2.drawContours(vis_img, suspicious_contours, -1, (255, 0, 0), 3)
    
    # Calculate confidence based on number of suspicious contours
    # and ratio to total number of significant contours
    significant_contours = [c for c in contours if cv2.contourArea(c) > min_area]
    
    if len(significant_contours) == 0:
        confidence = 0.0
    else:
        ratio = len(suspicious_contours) / len(significant_contours)
        confidence = min(0.95, ratio * 0.8)
    
    elapsed_time = time.time() - start_time
    logger.info(f"Splicing edge detection completed in {elapsed_time:.2f} seconds with confidence {confidence:.2f}")
    
    return Image.fromarray(vis_img), confidence

def detect_splicing_lighting_inconsistency(image):
    """
    Detect splicing forgery by analyzing lighting inconsistencies
    
    Args:
        image (PIL.Image): Input image
    
    Returns:
        Tuple[PIL.Image, float]: Visualization image and confidence score
    """
    start_time = time.time()
    logger.info("Starting splicing detection with lighting inconsistency analysis")
    
    # Resize image if needed
    image = resize_if_needed(image)
    
    # Convert PIL image to OpenCV format
    img_np = np.array(image)
    
    # Convert to RGB if it's in RGBA format
    if img_np.shape[2] == 4:
        img_np = cv2.cvtColor(img_np, cv2.COLOR_RGBA2RGB)
    
    # Convert to LAB color space
    lab = cv2.cvtColor(img_np, cv2.COLOR_RGB2LAB)
    
    # Split the LAB image into L, a, b channels
    l_channel, a_channel, b_channel = cv2.split(lab)
    
    # Apply CLAHE to L channel
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    cl = clahe.apply(l_channel)
    
    # Merge the enhanced L channel with the original a and b channels
    enhanced_lab = cv2.merge((cl, a_channel, b_channel))
    
    # Convert the LAB image back to RGB
    enhanced_rgb = cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2RGB)
    
    # Calculate the difference between original and enhanced images
    diff = cv2.absdiff(img_np, enhanced_rgb)
    
    # Convert difference to grayscale
    diff_gray = cv2.cvtColor(diff, cv2.COLOR_RGB2GRAY)
    
    # Apply threshold to get binary image
    _, thresh = cv2.threshold(diff_gray, 30, 255, cv2.THRESH_BINARY)
    
    # Apply morphological operations to remove noise
    kernel = np.ones((5, 5), np.uint8)
    opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
    closing = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, kernel)
    
    # Find contours in the binary image
    contours, _ = cv2.findContours(closing, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Create a visualization image
    vis_img = img_np.copy()
    
    # Filter contours by area
    min_area = 200  # Minimum area threshold
    suspicious_regions = []
    
    for contour in contours:
        area = cv2.contourArea(contour)
        if area > min_area:
            # Draw the contour
            cv2.drawContours(vis_img, [contour], -1, (0, 0, 255), 2)
            
            # Calculate bounding rectangle
            x, y, w, h = cv2.boundingRect(contour)
            
            # Draw bounding rectangle
            cv2.rectangle(vis_img, (x, y), (x + w, y + h), (255, 0, 0), 2)
            
            suspicious_regions.append((x, y, w, h))
    
    # Calculate confidence based on number and size of suspicious regions
    img_area = img_np.shape[0] * img_np.shape[1]
    suspicious_area = sum(w * h for _, _, w, h in suspicious_regions)
    
    # Confidence is higher if the suspicious area is a significant portion of the image
    # but not too large (which might indicate a false positive)
    area_ratio = suspicious_area / img_area
    if area_ratio < 0.01:  # Less than 1% of image area
        confidence = area_ratio * 10  # Low confidence
    elif area_ratio > 0.4:  # More than 40% of image area
        confidence = max(0, 0.8 - (area_ratio - 0.4))  # Decreasing confidence
    else:
        confidence = min(0.95, 0.4 + area_ratio)
    
    elapsed_time = time.time() - start_time
    logger.info(f"Splicing lighting detection completed in {elapsed_time:.2f} seconds with confidence {confidence:.2f}")
    
    return Image.fromarray(vis_img), confidence

def detect_splicing_combined(image):
    """
    Detect splicing forgery using a combination of methods
    
    Args:
        image (PIL.Image): Input image
    
    Returns:
        Tuple[PIL.Image, float]: Visualization image and confidence score
    """
    start_time = time.time()
    logger.info("Starting combined splicing detection")
    
    # Resize image if needed
    image = resize_if_needed(image)
    
    # Run both detection methods
    edge_img, edge_confidence = detect_splicing_edge_inconsistencies(image)
    lighting_img, lighting_confidence = detect_splicing_lighting_inconsistency(image)
    
    # If one method has much higher confidence, use its result
    if abs(edge_confidence - lighting_confidence) > 0.3:
        if edge_confidence > lighting_confidence:
            result_img = edge_img
            confidence = edge_confidence
        else:
            result_img = lighting_img
            confidence = lighting_confidence
    else:
        # Otherwise, combine the results with weighted average
        # Here we give slightly more weight to lighting analysis
        confidence = edge_confidence * 0.4 + lighting_confidence * 0.6
        
        # For visualization, overlay the edge detection result on the lighting result
        # This is a simple way to combine the visualizations
        result_img = lighting_img
    
    elapsed_time = time.time() - start_time
    logger.info(f"Combined splicing detection completed in {elapsed_time:.2f} seconds with confidence {confidence:.2f}")
    
    return result_img, confidence 