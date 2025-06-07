import numpy as np
import cv2
from PIL import Image
from typing import Tuple, Optional, List

def generate_forgery_heatmap(original_image: Image.Image, prediction_mask: np.ndarray, 
                            alpha: float = 0.6, colormap: int = cv2.COLORMAP_JET,
                            threshold: float = 0.5, 
                            enhance_detail: bool = True,
                            adaptive_overlay: bool = True) -> Image.Image:
    """
    Generate an advanced heatmap from a forgery prediction mask and overlay it on the original image.
    
    :param original_image: PIL.Image of the input image
    :param prediction_mask: 2D NumPy array with values between 0 and 1
    :param alpha: Transparency of the heatmap overlay (0-1)
    :param colormap: OpenCV colormap to use (e.g., cv2.COLORMAP_JET, cv2.COLORMAP_VIRIDIS)
    :param threshold: Threshold for highlighting potential forgery regions (0-1)
    :param enhance_detail: Whether to apply detail enhancement to highlight small manipulations
    :param adaptive_overlay: Whether to use adaptive opacity based on confidence
    :return: PIL.Image with heatmap overlay
    """
    # Convert PIL to RGB NumPy array
    original_np = np.array(original_image.convert("RGB"))
    h, w = original_np.shape[:2]

    # Resize prediction mask to image size
    heatmap = cv2.resize(prediction_mask, (w, h), interpolation=cv2.INTER_LINEAR)
    
    # Apply detail enhancement if requested
    if enhance_detail:
        # Use edge-preserving filter to enhance details
        heatmap = cv2.edgePreservingFilter(heatmap.astype(np.float32), 
                                          flags=cv2.RECURS_FILTER, 
                                          sigma_s=0.08, 
                                          sigma_r=0.1)
        
        # Apply sharpening to further enhance details
        kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]], dtype=np.float32)
        heatmap = cv2.filter2D(heatmap, -1, kernel)
        
        # Normalize again to 0-1 range
        heatmap = np.clip(heatmap, 0, 1)
    
    # Apply multi-level thresholding for better visualization
    heatmap_highlighted = np.copy(heatmap).astype(np.float32)  # Convert to float32 to ensure proper multiplication
    
    # Low confidence areas (less than threshold)
    low_mask = (heatmap < threshold)
    heatmap_highlighted[low_mask] = heatmap_highlighted[low_mask] * 0.2  # Use assignment instead of *=
    
    # Medium confidence areas
    medium_mask = (heatmap >= threshold) & (heatmap < threshold + 0.2)
    heatmap_highlighted[medium_mask] = heatmap_highlighted[medium_mask] * 0.7  # Use assignment instead of *=
    
    # High confidence areas remain at full intensity
    
    # Scale to 0-255
    heatmap_uint8 = np.uint8(255 * heatmap_highlighted)
    
    # Apply color map
    heatmap_color = cv2.applyColorMap(heatmap_uint8, colormap)
    
    # Adaptive overlay - vary alpha based on confidence if requested
    if adaptive_overlay:
        # Create an alpha mask that varies with confidence
        alpha_mask = np.ones_like(heatmap) * alpha
        alpha_mask[heatmap >= threshold + 0.2] = alpha * 1.5  # More opaque for high confidence
        alpha_mask[heatmap < threshold] = alpha * 0.5        # More transparent for low confidence
        alpha_mask = np.clip(alpha_mask, 0, 1)  # Ensure alpha stays in valid range
        alpha_mask = np.stack([alpha_mask, alpha_mask, alpha_mask], axis=2)
        
        # Create the overlay using the varying alpha
        overlay = original_np.copy()
        blended = (original_np * (1 - alpha_mask) + heatmap_color * alpha_mask)

# Clip and convert to uint8 to match overlay dtype
        blended_uint8 = np.clip(blended, 0, 255).astype(np.uint8)

        np.copyto(overlay, blended_uint8, where=(alpha_mask > 0))

    else:
        # Standard blending with fixed alpha
        beta = 1.0 - alpha
        overlay = cv2.addWeighted(original_np, beta, heatmap_color, alpha, 0)
    
    # Add border to high-confidence regions with enhanced visualization
    high_confidence_regions = (heatmap >= threshold + 0.2)
    if np.any(high_confidence_regions):
        high_confidence_regions = cv2.resize((high_confidence_regions).astype(np.uint8), (w, h))
        
        # Dilate slightly to create a more visible region
        kernel = np.ones((3, 3), np.uint8)
        high_confidence_regions = cv2.dilate(high_confidence_regions, kernel, iterations=1)
        
        # Find contours with hierarchy to detect nested manipulations
        contours, hierarchy = cv2.findContours(high_confidence_regions, 
                                             cv2.RETR_TREE, 
                                             cv2.CHAIN_APPROX_SIMPLE)
        
        # Draw contours with different styles based on area size
        for i, contour in enumerate(contours):
            area = cv2.contourArea(contour)
            
            if area > 500:  # Large regions
                cv2.drawContours(overlay, [contour], -1, (255, 255, 255), 2)
                
                # Add label showing confidence level at center of large regions
                M = cv2.moments(contour)
                if M["m00"] != 0:
                    cx = int(M["m10"] / M["m00"])
                    cy = int(M["m01"] / M["m00"])
                    
                    # Get average confidence in this region
                    mask = np.zeros((h, w), dtype=np.uint8)
                    cv2.drawContours(mask, [contour], -1, 255, -1)
                    mean_confidence = np.mean(heatmap[mask == 255]) * 100
                    confidence_text = f"{mean_confidence:.1f}%"
                    
                    # Put text with background for better visibility
                    text_size, _ = cv2.getTextSize(confidence_text, 
                                                cv2.FONT_HERSHEY_SIMPLEX, 
                                                0.5, 1)
                    text_w, text_h = text_size
                    
                    # Add text background
                    cv2.rectangle(overlay, 
                                (cx - text_w//2 - 2, cy - text_h//2 - 2),
                                (cx + text_w//2 + 2, cy + text_h//2 + 2),
                                (0, 0, 0), -1)
                    
                    cv2.putText(overlay, confidence_text, 
                              (cx - text_w//2, cy + text_h//2), 
                              cv2.FONT_HERSHEY_SIMPLEX, 0.5, 
                              (255, 255, 255), 1)
            else:  # Small regions
                cv2.drawContours(overlay, [contour], -1, (0, 255, 255), 1)

    return Image.fromarray(overlay)

def generate_multi_colormap_heatmap(original_image: Image.Image, prediction_mask: np.ndarray,
                                   threshold: float = 0.5) -> Image.Image:
    """
    Generate a multi-colormap heatmap for comparison, showing different visualization styles.
    
    :param original_image: Original PIL image
    :param prediction_mask: Prediction mask as numpy array
    :param threshold: Confidence threshold
    :return: A combined image with multiple colormap visualizations
    """
    # Define colormap options
    colormaps = [
        (cv2.COLORMAP_JET, "Jet"), 
        (cv2.COLORMAP_VIRIDIS, "Viridis"),
        (cv2.COLORMAP_INFERNO, "Inferno"), 
        (cv2.COLORMAP_PLASMA, "Plasma")
    ]
    
    # Generate heatmaps with different colormaps
    heatmap_images = []
    
    for colormap, name in colormaps:
        heatmap = generate_forgery_heatmap(
            original_image, 
            prediction_mask,
            colormap=colormap,
            threshold=threshold,
            alpha=0.7
        )
        
        # Add colormap name as text
        heatmap_np = np.array(heatmap)
        cv2.putText(heatmap_np, name, (10, 25), 
                  cv2.FONT_HERSHEY_SIMPLEX, 0.7, 
                  (255, 255, 255), 2)
        
        heatmap_images.append(Image.fromarray(heatmap_np))
    
    # Create a 2x2 grid
    width, height = original_image.width, original_image.height
    grid_image = Image.new('RGB', (width * 2, height * 2))
    
    # Paste the heatmaps into the grid
    grid_image.paste(heatmap_images[0], (0, 0))
    grid_image.paste(heatmap_images[1], (width, 0))
    grid_image.paste(heatmap_images[2], (0, height))
    grid_image.paste(heatmap_images[3], (width, height))
    
    return grid_image

def generate_heatmap_detail_view(original_image: Image.Image, prediction_mask: np.ndarray,
                               threshold: float = 0.5) -> Optional[Image.Image]:
    """
    Generate a zoomed-in view of the most suspicious region for detailed analysis.
    
    :param original_image: Original PIL image
    :param prediction_mask: Prediction mask as numpy array
    :param threshold: Confidence threshold
    :return: A zoomed detail view image, or None if no significant region found
    """
    # Convert to numpy and resize mask to image size
    original_np = np.array(original_image.convert("RGB"))
    h, w = original_np.shape[:2]
    heatmap = cv2.resize(prediction_mask, (w, h), interpolation=cv2.INTER_LINEAR)
    
    # Find regions above threshold
    binary_mask = (heatmap > threshold).astype(np.uint8)
    if not np.any(binary_mask):
        return None  # No significant regions found
    
    # Find contours of suspicious regions
    contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        return None
    
    # Find the largest contour (most suspicious region)
    largest_contour = max(contours, key=cv2.contourArea)
    x, y, w_roi, h_roi = cv2.boundingRect(largest_contour)
    
    # Add padding for context (20% on each side)
    padding_x = int(w_roi * 0.2)
    padding_y = int(h_roi * 0.2)
    
    # Calculate region with padding, ensuring we stay within image bounds
    x1 = max(0, x - padding_x)
    y1 = max(0, y - padding_y)
    x2 = min(w, x + w_roi + padding_x)
    y2 = min(h, y + h_roi + padding_y)
    
    # Extract the region of interest
    roi = original_np[y1:y2, x1:x2]
    
    # Create a heatmap for this region
    roi_mask = heatmap[y1:y2, x1:x2]
    
    # Generate a high-detail heatmap for this region
    detail_heatmap = generate_forgery_heatmap(
        Image.fromarray(roi),
        roi_mask,
        alpha=0.7,
        threshold=threshold,
        enhance_detail=True
    )
    
    # Add a border to highlight this is a zoomed view
    detail_np = np.array(detail_heatmap)
    cv2.rectangle(detail_np, (0, 0), (detail_np.shape[1]-1, detail_np.shape[0]-1), (255, 255, 255), 2)
    
    # Add text label
    cv2.putText(detail_np, "Detail View", (10, 25), 
              cv2.FONT_HERSHEY_SIMPLEX, 0.7, 
              (255, 255, 255), 2)
    
    return Image.fromarray(detail_np)
