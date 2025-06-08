import os
import time
import numpy as np
from PIL import Image
from src.api.utils.logger import logger
from src.api.services.feature_extractor import get_feature_vector
import cv2
import io

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

def generate_ela_image(image_path_or_data, quality=85, scale=10):
    """
    Generate an Error Level Analysis (ELA) image
    
    :param image_path_or_data: Path to the image file or file-like object or bytes
    :param quality: JPEG quality for recompression
    :param scale: Scaling factor for ELA
    :return: Bytes of the ELA image in PNG format
    """
    try:
        # Open the image
        if isinstance(image_path_or_data, str):
            original = Image.open(image_path_or_data).convert('RGB')
        elif isinstance(image_path_or_data, bytes):
            original = Image.open(io.BytesIO(image_path_or_data)).convert('RGB')
        elif hasattr(image_path_or_data, 'read') and hasattr(image_path_or_data, 'seek'):
            # Save current position
            pos = image_path_or_data.tell()
            # Reset buffer position
            image_path_or_data.seek(0)
            # Open the image
            original = Image.open(image_path_or_data).convert('RGB')
            # Reset buffer position
            image_path_or_data.seek(pos)
        else:
            raise ValueError(f"Unsupported image input type: {type(image_path_or_data)}")
        
        # Save to a temporary buffer with the specified quality
        temp_buffer = io.BytesIO()
        original.save(temp_buffer, format='JPEG', quality=quality)
        temp_buffer.seek(0)
        
        # Load the recompressed image
        recompressed = Image.open(temp_buffer)
        
        # Calculate the difference
        ela_image = np.array(original) - np.array(recompressed)
        
        # Scale the difference to make it more visible
        ela_image = (np.abs(ela_image) * scale).astype(np.uint8)
        
        # Convert to PIL Image
        ela_pil = Image.fromarray(ela_image)
        
        # Convert ELA image to bytes
        output_buffer = io.BytesIO()
        ela_pil.save(output_buffer, format='PNG')
        output_buffer.seek(0)
        
        return output_buffer.getvalue()
    
    except Exception as e:
        logger.error(f"Error generating ELA image: {str(e)}", exc_info=True)
        return None
    
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
