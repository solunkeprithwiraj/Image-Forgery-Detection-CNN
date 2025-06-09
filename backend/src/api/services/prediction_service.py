import os
import time
import numpy as np
from src.api.utils.logger import logger
from src.api.services.feature_extractor import get_feature_vector
import cv2
import io
import logging
from PIL import Image, ImageEnhance, ImageDraw, ImageFont
from scipy import ndimage
from sklearn.cluster import DBSCAN
# Cache for prediction results
prediction_cache = {}

def predict_image(image_path_or_data, cnn_model, svm_model):
    """
    Predict whether an image is tampered or authentic
    :param image_path_or_data: Path to the image file or file-like object or bytes
    :param cnn_model: The pre-trained CNN model
    :param svm_model: The pre-trained SVM model
    :returns: Prediction result with confidence and ELA image
    """
    # Handle different types of image inputs
    image_hash = None
    image_path = None
    
    if isinstance(image_path_or_data, str):
        # It's a file path
        image_path = image_path_or_data
        image_hash = f"file:{image_path}"
    elif isinstance(image_path_or_data, bytes):
        # It's raw bytes
        image_hash = f"bytes:{hash(image_path_or_data)}"
    elif hasattr(image_path_or_data, 'read') and hasattr(image_path_or_data, 'seek'):
        # It's a file-like object (buffer)
        # Get the current position
        pos = image_path_or_data.tell()
        # Read the content
        content = image_path_or_data.read()
        # Reset the position
        image_path_or_data.seek(pos)
        # Create a hash of the content
        image_hash = f"buffer:{hash(content)}"
    else:
        raise ValueError(f"Unsupported image input type: {type(image_path_or_data)}")
    
    # Check if we have this prediction in cache
    if image_hash in prediction_cache:
        logger.debug(f"Using cached prediction for hash: {image_hash}")
        return prediction_cache[image_hash]
    
    # Log image source
    if image_path:
        logger.info(f"Predicting image: {os.path.basename(image_path)}")
    else:
        logger.info(f"Predicting image from memory (hash: {image_hash})")
    
    start_time = time.time()
    
    try:
        # Get feature vector
        feature_vector = get_feature_vector(image_path_or_data, cnn_model)
        
        # Open the image to get direct CNN prediction
        if image_path:
            img = Image.open(image_path).convert('RGB')
        elif isinstance(image_path_or_data, bytes):
            img = Image.open(io.BytesIO(image_path_or_data)).convert('RGB')
        else:
            # Save current position
            pos = image_path_or_data.tell()
            # Reset buffer position
            image_path_or_data.seek(0)
            # Open the image
            img = Image.open(image_path_or_data).convert('RGB')
            # Reset buffer position
            image_path_or_data.seek(pos)
        
        # Get CNN model's direct prediction
        try:
            # For CNN models that output directly
            img_resized = img.resize((224, 224))  # Adjust size based on your CNN model's input requirements
            img_array = np.array(img_resized) / 255.0  # Normalize
            img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
            
            # Try to predict directly with the CNN model
            cnn_prediction = cnn_model.predict(img_array)
            
            # Handle different output formats
            if isinstance(cnn_prediction, list):
                cnn_prob_tampered = float(cnn_prediction[0][1] if len(cnn_prediction[0]) > 1 else cnn_prediction[0][0])
            else:
                cnn_prob_tampered = float(cnn_prediction[0][1] if cnn_prediction.shape[1] > 1 else cnn_prediction[0][0])
                
            cnn_class = 1 if cnn_prob_tampered > 0.5 else 0
            logger.debug(f"CNN direct prediction: {cnn_class}, tampered prob: {cnn_prob_tampered}")
            
            # For binary classification where output is just a single value
            if cnn_prediction.shape[1] == 1:
                cnn_prob_tampered = float(cnn_prediction[0][0])
                cnn_class = 1 if cnn_prob_tampered > 0.5 else 0
                # For models that output sigmoid values directly
                logger.debug(f"CNN using sigmoid output: {cnn_class}, prob: {cnn_prob_tampered}")
        except Exception as e:
            logger.debug(f"Could not get direct CNN prediction: {str(e)}. Using features with SVM instead.")
            # If CNN direct prediction fails, use feature extraction with SVM
            cnn_class = None
            cnn_prob_tampered = None
        
        # Get SVM prediction using CNN features
        svm_prediction = svm_model.predict(feature_vector)[0]
        
        # Get SVM prediction probabilities
        svm_probabilities = svm_model.predict_proba(feature_vector)[0]
        svm_prob_tampered = svm_probabilities[1]  # Probability of being tampered
        
        logger.debug(f"SVM prediction: {svm_prediction}, tampered prob: {svm_prob_tampered}")
        
        # Combine predictions if we have both
        if cnn_class is not None:
            # Ensemble prediction - weighted average of CNN and SVM probabilities
            combined_prob_tampered = 0.7 * cnn_prob_tampered + 0.3 * svm_prob_tampered
            final_prediction = 1 if combined_prob_tampered > 0.5 else 0
            confidence = combined_prob_tampered if final_prediction == 1 else (1 - combined_prob_tampered)
            logger.debug(f"Combined prediction: {final_prediction}, confidence: {confidence}")
        else:
            # Use only SVM prediction
            final_prediction = svm_prediction
            confidence = svm_prob_tampered if final_prediction == 1 else (1 - svm_prob_tampered)
        
        # Generate ELA image
        ela_image_bytes = generate_ela_image(image_path_or_data)
        
        # Create filename for the result
        if image_path:
            filename = os.path.basename(image_path)
        else:
            filename = f"image_{image_hash.split(':')[1][:8]}.jpg"
        
        result = {
            "filename": filename,
            "prediction": int(final_prediction),
            "prediction_label": "tampered" if final_prediction == 1 else "authentic",
            "confidence": float(confidence),
            "ela_image": ela_image_bytes,
            "processing_time": time.time() - start_time
        }
        
        # Cache the result
        prediction_cache[image_hash] = result
        
        logger.info(f"Prediction result: {result['prediction_label']} with {result['confidence']:.2f} confidence")
        return result
    except Exception as e:
        logger.error(f"Error predicting image: {str(e)}", exc_info=True)
        raise ValueError(f"Prediction error: {str(e)}")



logger = logging.getLogger(__name__)

def generate_ela_image(image_path_or_data, quality=85, scale=10, 
                                        detect_tampering=True, highlight_regions=True,
                                        return_analysis=False):
    """
    Generate ELA image with automatic tampered region detection
    
    :param image_path_or_data: Path to the image file or file-like object or bytes
    :param quality: JPEG quality for recompression (1-100)
    :param scale: Base scaling factor for ELA
    :param detect_tampering: Whether to detect tampered regions
    :param highlight_regions: Whether to highlight detected regions
    :param return_analysis: Whether to return detailed analysis
    :return: Bytes of the ELA image, optionally with analysis data
    """
    try:
        # Load and process image
        original = _load_image(image_path_or_data)
        recompressed = _recompress_image(original, quality)
        ela_array = _calculate_ela_difference(original, recompressed)
        
        # Adaptive scaling
        adaptive_scale = _calculate_adaptive_scale(ela_array, base_scale=scale)
        ela_scaled = _scale_ela_values(ela_array, adaptive_scale)
        
        analysis_data = {}
        
        if detect_tampering:
            # Detect tampered regions
            tampered_regions = detect_tampered_regions(ela_scaled)
            analysis_data['tampered_regions'] = tampered_regions
            analysis_data['tampering_detected'] = len(tampered_regions) > 0
            
            if highlight_regions and tampered_regions:
                ela_scaled = highlight_tampered_regions(ela_scaled, tampered_regions)
        
        # Convert to bytes
        ela_bytes = _ela_to_bytes(ela_scaled)
        
        if return_analysis:
            analysis_data['ela_stats'] = calculate_ela_statistics(ela_array)
            return ela_bytes, analysis_data
        
        return ela_bytes
    
    except Exception as e:
        logger.error(f"Error generating ELA with tampering detection: {str(e)}", exc_info=True)
        return None


def detect_tampered_regions(ela_image, min_area=100, intensity_threshold=None):
    """
    Detect potentially tampered regions in ELA image
    
    :param ela_image: ELA image array
    :param min_area: Minimum area for a region to be considered significant
    :param intensity_threshold: Threshold for high-intensity pixels (auto if None)
    :return: List of detected regions with bounding boxes and confidence scores
    """
    try:
        # Convert to grayscale for analysis
        if len(ela_image.shape) == 3:
            ela_gray = np.mean(ela_image, axis=2)
        else:
            ela_gray = ela_image
        
        # Auto-calculate threshold if not provided
        if intensity_threshold is None:
            # Use adaptive threshold based on image statistics
            mean_intensity = np.mean(ela_gray)
            std_intensity = np.std(ela_gray)
            intensity_threshold = mean_intensity + 2 * std_intensity
            intensity_threshold = max(intensity_threshold, 30)  # Minimum threshold
        
        # Create binary mask for high-intensity regions
        high_intensity_mask = ela_gray > intensity_threshold
        
        # Morphological operations to clean up the mask
        kernel = np.ones((3, 3), np.uint8)
        cleaned_mask = cv2.morphologyEx(high_intensity_mask.astype(np.uint8), 
                                       cv2.MORPH_CLOSE, kernel)
        cleaned_mask = cv2.morphologyEx(cleaned_mask, cv2.MORPH_OPEN, kernel)
        
        # Find connected components
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
            cleaned_mask, connectivity=8)
        
        tampered_regions = []
        
        for i in range(1, num_labels):  # Skip background (label 0)
            area = stats[i, cv2.CC_STAT_AREA]
            
            if area >= min_area:
                x = stats[i, cv2.CC_STAT_LEFT]
                y = stats[i, cv2.CC_STAT_TOP]
                w = stats[i, cv2.CC_STAT_WIDTH]
                h = stats[i, cv2.CC_STAT_HEIGHT]
                
                # Calculate confidence score based on multiple factors
                region_mask = (labels == i)
                region_intensities = ela_gray[region_mask]
                
                confidence_score = calculate_tampering_confidence(
                    region_intensities, area, w, h, ela_gray.shape)
                
                tampered_regions.append({
                    'bbox': (x, y, w, h),
                    'area': area,
                    'confidence': confidence_score,
                    'mean_intensity': np.mean(region_intensities),
                    'max_intensity': np.max(region_intensities),
                    'centroid': (int(centroids[i][0]), int(centroids[i][1]))
                })
        
        # Sort by confidence score (highest first)
        tampered_regions.sort(key=lambda x: x['confidence'], reverse=True)
        
        return tampered_regions
    
    except Exception as e:
        logger.error(f"Error detecting tampered regions: {str(e)}")
        return []


def calculate_tampering_confidence(region_intensities, area, width, height, image_shape):
    """Calculate confidence score for a potential tampered region"""
    
    # Factor 1: Intensity statistics (higher mean/max = more suspicious)
    mean_intensity = np.mean(region_intensities)
    max_intensity = np.max(region_intensities)
    intensity_score = min(1.0, (mean_intensity / 255) * 0.6 + (max_intensity / 255) * 0.4)
    
    # Factor 2: Size relative to image (moderate sizes are more suspicious)
    total_pixels = image_shape[0] * image_shape[1]
    size_ratio = area / total_pixels
    # Peak suspicion around 1-10% of image
    if 0.01 <= size_ratio <= 0.10:
        size_score = 1.0
    elif size_ratio < 0.01:
        size_score = size_ratio / 0.01  # Smaller regions less suspicious
    else:
        size_score = max(0.1, 1.0 - (size_ratio - 0.10) / 0.20)  # Very large regions less suspicious
    
    # Factor 3: Shape factor (very elongated shapes might be artifacts)
    aspect_ratio = max(width, height) / min(width, height)
    if aspect_ratio <= 3:
        shape_score = 1.0
    else:
        shape_score = max(0.3, 1.0 - (aspect_ratio - 3) / 10)
    
    # Factor 4: Intensity variance (uniform high intensity more suspicious)
    intensity_var = np.var(region_intensities)
    if intensity_var < 100:  # Low variance = uniform = more suspicious
        variance_score = 1.0
    else:
        variance_score = max(0.5, 1.0 - (intensity_var - 100) / 1000)
    
    # Combine factors with weights
    confidence = (intensity_score * 0.4 + 
                 size_score * 0.25 + 
                 shape_score * 0.15 + 
                 variance_score * 0.20)
    
    return min(1.0, confidence)


def highlight_tampered_regions(ela_image, tampered_regions, min_confidence=0.3):
    """
    Highlight detected tampered regions on the ELA image
    
    :param ela_image: Original ELA image
    :param tampered_regions: List of detected regions
    :param min_confidence: Minimum confidence to highlight a region
    :return: ELA image with highlighted regions
    """
    try:
        # Convert to PIL for drawing
        ela_pil = Image.fromarray(ela_image)
        draw = ImageDraw.Draw(ela_pil)
        
        # Define colors for different confidence levels
        colors = {
            'high': (255, 0, 0),      # Red for high confidence
            'medium': (255, 165, 0),   # Orange for medium confidence
            'low': (255, 255, 0)       # Yellow for low confidence
        }
        
        for region in tampered_regions:
            if region['confidence'] < min_confidence:
                continue
            
            x, y, w, h = region['bbox']
            confidence = region['confidence']
            
            # Choose color based on confidence
            if confidence >= 0.7:
                color = colors['high']
                width = 3
            elif confidence >= 0.5:
                color = colors['medium']
                width = 2
            else:
                color = colors['low']
                width = 2
            
            # Draw bounding box
            draw.rectangle([x, y, x + w, y + h], outline=color, width=width)
            
            # Add confidence label
            label = f"{confidence:.2f}"
            
            # Try to load a font, fall back to default if not available
            try:
                font = ImageFont.truetype("arial.ttf", 12)
            except:
                font = ImageFont.load_default()
            
            # Calculate text size and position
            bbox = draw.textbbox((0, 0), label, font=font)
            text_width = bbox[2] - bbox[0]
            text_height = bbox[3] - bbox[1]
            
            # Position text above the region
            text_x = x
            text_y = max(0, y - text_height - 2)
            
            # Draw background rectangle for text
            draw.rectangle([text_x, text_y, text_x + text_width, text_y + text_height], 
                         fill=(0, 0, 0, 128))
            
            # Draw text
            draw.text((text_x, text_y), label, fill=color, font=font)
            
            # Add center point
            cx, cy = region['centroid']
            draw.ellipse([cx-2, cy-2, cx+2, cy+2], fill=color)
        
        return np.array(ela_pil)
    
    except Exception as e:
        logger.error(f"Error highlighting tampered regions: {str(e)}")
        return ela_image


def generate_tampering_report(image_path_or_data, quality=85):
    """
    Generate a comprehensive tampering analysis report
    
    :param image_path_or_data: Image to analyze
    :param quality: JPEG quality for ELA
    :return: Dictionary with detailed analysis
    """
    try:
        ela_bytes, analysis = generate_ela_with_tampering_detection(
            image_path_or_data, quality=quality, return_analysis=True)
        
        report = {
            'overall_assessment': 'SUSPICIOUS' if analysis['tampering_detected'] else 'CLEAN',
            'confidence_level': 'HIGH' if len(analysis['tampered_regions']) > 2 else 
                              'MEDIUM' if analysis['tampering_detected'] else 'LOW',
            'regions_detected': len(analysis['tampered_regions']),
            'ela_statistics': analysis['ela_stats'],
            'detailed_regions': []
        }
        
        for i, region in enumerate(analysis['tampered_regions'][:5]):  # Top 5 regions
            report['detailed_regions'].append({
                'region_id': i + 1,
                'confidence': region['confidence'],
                'location': f"({region['bbox'][0]}, {region['bbox'][1]})",
                'size': f"{region['bbox'][2]}x{region['bbox'][3]}",
                'area_pixels': region['area'],
                'assessment': 'HIGH SUSPICION' if region['confidence'] > 0.7 else
                            'MODERATE SUSPICION' if region['confidence'] > 0.5 else
                            'LOW SUSPICION'
            })
        
        return report, ela_bytes
    
    except Exception as e:
        logger.error(f"Error generating tampering report: {str(e)}")
        return None, None


# Keep the utility functions from the previous version
def _load_image(image_path_or_data):
    """Load image from various input types"""
    if isinstance(image_path_or_data, str):
        return Image.open(image_path_or_data).convert('RGB')
    elif isinstance(image_path_or_data, bytes):
        return Image.open(io.BytesIO(image_path_or_data)).convert('RGB')
    elif hasattr(image_path_or_data, 'read') and hasattr(image_path_or_data, 'seek'):
        pos = image_path_or_data.tell()
        image_path_or_data.seek(0)
        img = Image.open(image_path_or_data).convert('RGB')
        image_path_or_data.seek(pos)
        return img
    else:
        raise ValueError(f"Unsupported image input type: {type(image_path_or_data)}")


def _recompress_image(original, quality):
    """Recompress image with optimized JPEG settings"""
    temp_buffer = io.BytesIO()
    save_kwargs = {
        'format': 'JPEG',
        'quality': quality,
        'optimize': True,
        'progressive': False,
        'subsampling': 0 if quality >= 95 else -1
    }
    original.save(temp_buffer, **save_kwargs)
    temp_buffer.seek(0)
    return Image.open(temp_buffer).convert('RGB')


def _calculate_ela_difference(original, recompressed):
    """Calculate ELA difference with improved precision"""
    orig_array = np.array(original, dtype=np.float64)
    recomp_array = np.array(recompressed, dtype=np.float64)
    return np.abs(orig_array - recomp_array)


def _calculate_adaptive_scale(ela_array, base_scale=10):
    """Calculate adaptive scaling factor"""
    mean_diff = np.mean(ela_array)
    std_diff = np.std(ela_array)
    max_diff = np.max(ela_array)
    
    if max_diff == 0:
        return base_scale
    
    if std_diff > mean_diff * 0.5:
        adaptive_factor = min(2.0, std_diff / mean_diff)
    else:
        adaptive_factor = max(0.5, mean_diff / (std_diff + 1e-6))
    
    target_max = 255 * 0.8
    auto_scale = target_max / (max_diff + 1e-6)
    final_scale = base_scale * adaptive_factor * min(1.0, auto_scale / base_scale)
    
    return max(1.0, min(50.0, final_scale))


def _scale_ela_values(ela_array, scale):
    """Scale ELA values with improved dynamic range"""
    scaled = ela_array * scale
    p99 = np.percentile(scaled, 99)
    if p99 > 0:
        scaled = np.clip(scaled, 0, p99)
        scaled = (scaled / p99 * 255).astype(np.uint8)
    else:
        scaled = np.clip(scaled, 0, 255).astype(np.uint8)
    return scaled


def _ela_to_bytes(ela_array):
    """Convert ELA array to PNG bytes"""
    ela_pil = Image.fromarray(ela_array.astype(np.uint8))
    output_buffer = io.BytesIO()
    ela_pil.save(output_buffer, format='PNG', optimize=True, compress_level=6)
    output_buffer.seek(0)
    return output_buffer.getvalue()


def calculate_ela_statistics(ela_array):
    """Calculate comprehensive ELA statistics"""
    return {
        'mean_error': float(np.mean(ela_array)),
        'std_error': float(np.std(ela_array)),
        'max_error': float(np.max(ela_array)),
        'median_error': float(np.median(ela_array)),
        'q75_error': float(np.percentile(ela_array, 75)),
        'q95_error': float(np.percentile(ela_array, 95)),
        'high_error_pixels': int(np.sum(ela_array > 20)),
        'total_pixels': int(ela_array.size // 3)
    }


# Example usage functions
def quick_tampering_check(image_path):
    """Quick function to check if an image might be tampered"""
    report, _ = generate_tampering_report(image_path)
    if report:
        print(f"Assessment: {report['overall_assessment']}")
        print(f"Confidence: {report['confidence_level']}")
        print(f"Suspicious regions found: {report['regions_detected']}")
        return report['overall_assessment'] == 'SUSPICIOUS'
    return False 
def get_prediction_mask(image, cnn_model, svm_model, patch_size=64, stride=32):
    """
    Slide a window over the image, run prediction per patch, and return 2D heatmap.
    :param image: PIL.Image object of the input image
    :param cnn_model: The pre-trained CNN model
    :param svm_model: The pre-trained SVM model
    :param patch_size: Size of the sliding window patch
    :param stride: Step size for sliding window
    :returns: 2D numpy array with tampering confidence scores
    """
    image = image.convert("RGB")
    image_np = np.array(image)
    h, w, _ = image_np.shape

    heatmap = np.zeros(((h - patch_size) // stride + 1, (w - patch_size) // stride + 1))
    
    logger.info(f"Generating prediction mask for image of size {w}x{h}")
    start_time = time.time()

    temp_file = None
    try:
        # Create a temporary file to save patches
        import tempfile
        temp_file = tempfile.NamedTemporaryFile(suffix='.jpg', delete=False)
        temp_path = temp_file.name
        temp_file.close()
        
        for i in range(0, h - patch_size + 1, stride):
            for j in range(0, w - patch_size + 1, stride):
                patch = image.crop((j, i, j + patch_size, i + patch_size))
                
                # Save patch to temp file
                patch.save(temp_path)
                
                try:
                    # Get feature vector using the existing function
                    feature = get_feature_vector(temp_path, cnn_model)
                    
                    # Try to get CNN prediction first
                    try:
                        cnn_prediction = cnn_model.predict(np.array([feature[0]]))
                        proba = float(cnn_prediction[0][1])  # Probability of being tampered
                    except:
                        # Fall back to SVM if CNN direct prediction fails
                        proba = svm_model.predict_proba(feature)[0][1]
                        
                except Exception as e:
                    logger.error(f"Error processing patch at ({j},{i}): {str(e)}")
                    proba = 0.0  # fallback in case of error

                heatmap[i // stride, j // stride] = proba
        
        logger.info(f"Heatmap generation completed in {time.time() - start_time:.2f} seconds")
        return heatmap
    
    except Exception as e:
        logger.error(f"Error generating prediction mask: {str(e)}", exc_info=True)
        raise ValueError(f"Heatmap generation error: {str(e)}")
    
    finally:
        # Clean up temp file
        if temp_file and os.path.exists(temp_path):
            os.unlink(temp_path)
            logger.debug(f"Temporary patch file removed: {temp_path}")
