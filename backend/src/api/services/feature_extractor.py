import os
import time
import cv2
import numpy as np
import torch
from torch.autograd import Variable
import torchvision.transforms as transforms
from skimage.util import view_as_windows
from src.api.utils.logger import logger
import io

# Cache for feature vectors to avoid recomputing them
feature_vector_cache = {}

def optimized_get_patch_yi(model, image):
    """
    Optimized version of get_patch_yi that uses fewer patches and faster processing
    :param model: The pre-trained CNN object
    :param image: The image
    :returns: The image's feature representation
    """
    # Use a larger stride to reduce the number of patches
    # Original stride was 1024, we'll use a larger value based on image size
    h, w = image.shape[:2]
    # Adaptive stride based on image size - larger images get larger stride
    stride = max(1024, min(h, w) // 2)  # At least 1024, but can be larger for big images
    
    transform = transforms.Compose([transforms.ToTensor()])
    
    # Extract patches with the window shape and stride
    window_shape = (128, 128, 3)
    
    # Handle images that are too small
    if h < window_shape[0] or w < window_shape[1]:
        # Resize to minimum dimensions if image is too small
        new_h = max(window_shape[0], h)
        new_w = max(window_shape[1], w)
        image = cv2.resize(image, (new_w, new_h))
        h, w = image.shape[:2]
    
    # Extract windows
    try:
        windows = view_as_windows(image, window_shape, step=stride)
        
        # Limit the number of patches to process (max 4 patches)
        max_patches = 4
        patches = []
        
        for m in range(min(windows.shape[0], 2)):
            for n in range(min(windows.shape[1], 2)):
                if len(patches) < max_patches:
                    patches.append(windows[m][n][0])
        
        # If we have no patches (rare case), create one from the center
        if not patches and h >= window_shape[0] and w >= window_shape[1]:
            center_h = h // 2 - window_shape[0] // 2
            center_w = w // 2 - window_shape[1] // 2
            patches = [image[center_h:center_h+window_shape[0], center_w:center_w+window_shape[1]]]
    
    except Exception as e:
        # Fallback for any errors in patch extraction
        logger.warning(f"Error in patch extraction: {str(e)}. Using center crop fallback.")
        # Take center crop as fallback
        if h >= window_shape[0] and w >= window_shape[1]:
            center_h = h // 2 - window_shape[0] // 2
            center_w = w // 2 - window_shape[1] // 2
            patches = [image[center_h:center_h+window_shape[0], center_w:center_w+window_shape[1]]]
        else:
            # If image is too small, resize and take whole image
            image = cv2.resize(image, (window_shape[1], window_shape[0]))
            patches = [image]
    
    # Process patches through the CNN
    y = []
    with torch.no_grad():
        for patch in patches:
            img_tensor = transform(patch)
            img_tensor.unsqueeze_(0)
            img_variable = Variable(img_tensor.double())
            # Get features directly from the model
            yi = model(img_variable)
            y.append(yi)
    
    # Stack features and compute mean
    if y:
        y = np.vstack(tuple(y))
        # Use mean pooling to combine features
        y_hat = np.array(y).mean(axis=0)
        return y_hat
    else:
        # Fallback: return zeros if no patches could be processed
        logger.warning("No patches could be processed, returning zero vector")
        return np.zeros(400)

def get_feature_vector(image_path_or_data, model):
    """
    Extract feature vector from an image with caching and optimizations
    :param image_path_or_data: Path to the image file or file-like object or bytes
    :param model: The pre-trained CNN model
    :returns: Feature vector for the image
    """
    # Create a cache key based on the input type
    cache_key = None
    if isinstance(image_path_or_data, str):
        cache_key = f"file:{image_path_or_data}"
    elif isinstance(image_path_or_data, bytes):
        cache_key = f"bytes:{hash(image_path_or_data)}"
    elif hasattr(image_path_or_data, 'read') and hasattr(image_path_or_data, 'seek'):
        # Get the current position
        pos = image_path_or_data.tell()
        # Read the content
        content = image_path_or_data.read()
        # Reset the position
        image_path_or_data.seek(pos)
        # Create a hash of the content
        cache_key = f"buffer:{hash(content)}"
    
    # Check if we have this image in cache
    if cache_key in feature_vector_cache:
        logger.debug(f"Using cached feature vector for: {cache_key}")
        return feature_vector_cache[cache_key]
    
    start_time = time.time()
    logger.debug(f"Extracting feature vector from: {cache_key}")
    feature_vector = np.empty((1, 400))
    
    # Read the image based on input type
    if isinstance(image_path_or_data, str):
        # It's a file path
        file_ext = os.path.splitext(image_path_or_data)[1].lower()
        logger.debug(f"File extension: {file_ext}")
        
        if file_ext in ['.tif', '.tiff']:
            # Use cv2.IMREAD_UNCHANGED for TIFF images to preserve all channels
            img = cv2.imread(image_path_or_data, cv2.IMREAD_UNCHANGED)
            logger.debug(f"TIFF image loaded, shape: {img.shape if img is not None else 'None'}, dtype: {img.dtype if img is not None else 'None'}")
            
            # Convert to BGR if needed (some TIFF images might have more than 3 channels)
            if img is not None and len(img.shape) > 2:
                if img.shape[2] > 3:
                    logger.debug("TIFF image has more than 3 channels, converting to BGR")
                    img = img[:, :, :3]  # Take only the first 3 channels
        else:
            # Regular image loading for JPG, PNG, etc.
            img = cv2.imread(image_path_or_data)
            logger.debug(f"Regular image loaded, shape: {img.shape if img is not None else 'None'}")
    
    elif isinstance(image_path_or_data, bytes):
        # It's raw bytes, convert to numpy array
        nparr = np.frombuffer(image_path_or_data, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        logger.debug(f"Image loaded from bytes, shape: {img.shape if img is not None else 'None'}")
    
    elif hasattr(image_path_or_data, 'read') and hasattr(image_path_or_data, 'seek'):
        # It's a file-like object
        # Save current position
        pos = image_path_or_data.tell()
        # Get image data
        image_data = image_path_or_data.read()
        # Reset position
        image_path_or_data.seek(pos)
        
        # Convert to numpy array
        nparr = np.frombuffer(image_data, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        logger.debug(f"Image loaded from buffer, shape: {img.shape if img is not None else 'None'}")
    
    else:
        logger.error(f"Unsupported image input type: {type(image_path_or_data)}")
        raise ValueError(f"Unsupported image input type: {type(image_path_or_data)}")
    
    if img is None:
        logger.error(f"Failed to load image from: {cache_key}")
        raise ValueError("Invalid image data")
    
    # Resize large images to reduce processing time
    h, w = img.shape[:2]
    max_dimension = 1024
    if max(h, w) > max_dimension:
        scale = max_dimension / max(h, w)
        new_h, new_w = int(h * scale), int(w * scale)
        img = cv2.resize(img, (new_w, new_h))
        logger.debug(f"Resized image to {new_w}x{new_h} for faster processing")
    
    # Use the optimized function instead of the original get_patch_yi
    feature_vector[0, :] = optimized_get_patch_yi(model, img)
    
    # Cache the result
    feature_vector_cache[cache_key] = feature_vector
    
    elapsed_time = time.time() - start_time
    logger.debug(f"Feature vector extracted in {elapsed_time:.2f} seconds")
    return feature_vector