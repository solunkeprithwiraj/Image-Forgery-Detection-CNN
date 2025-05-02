import cv2
import numpy as np
import torch
from torchvision import transforms
import torchvision.transforms.functional as TF
from PIL import Image
import io

def preprocess_web_image(image_path, normalize=True):
    """
    Enhanced preprocessing for web images to handle different formats, 
    compression artifacts, and make them compatible with the trained model.
    
    Args:
        image_path: Path to the web image
        normalize: Whether to apply normalization
        
    Returns:
        Preprocessed image compatible with the model
    """
    try:
        # Load image with OpenCV (handles most formats)
        image = cv2.imread(image_path)
        
        # Check if image was loaded successfully
        if image is None:
            # Try with PIL as fallback (better for some formats)
            pil_image = Image.open(image_path).convert('RGB')
            image = np.array(pil_image)
            # Convert RGB to BGR for OpenCV compatibility
            image = image[:, :, ::-1].copy()
        
        # Apply basic preprocessing to match training distribution
        # 1. Histogram equalization for channel normalization
        channels = cv2.split(image)
        eq_channels = []
        for ch in channels:
            eq_channels.append(cv2.equalizeHist(ch))
        image = cv2.merge(eq_channels)
        
        # 2. Denoise image to reduce compression artifacts
        image = cv2.fastNlMeansDenoisingColored(image, None, 10, 10, 7, 21)
        
        # 3. Match the color statistics to the training distribution
        # (Simple approach: adjust contrast and brightness)
        image = cv2.convertScaleAbs(image, alpha=1.1, beta=5)
        
        # Return the processed image
        return image
        
    except Exception as e:
        print(f"Error preprocessing image: {e}")
        return None
        
def get_augmented_patches(image, patch_size=128, stride=64):
    """
    Extract patches from image with multiple augmentations for robust prediction.
    
    Args:
        image: Input image
        patch_size: Size of patches to extract
        stride: Stride for patch extraction
        
    Returns:
        List of augmented patches
    """
    height, width = image.shape[:2]
    patches = []
    
    # Define augmentations
    augmentations = [
        # Original
        lambda x: x,
        # Rotation
        lambda x: cv2.rotate(x, cv2.ROTATE_90_CLOCKWISE),
        lambda x: cv2.rotate(x, cv2.ROTATE_180),
        lambda x: cv2.rotate(x, cv2.ROTATE_90_COUNTERCLOCKWISE),
        # Flip
        lambda x: cv2.flip(x, 1),  # horizontal
    ]
    
    # Extract patches with sliding window
    for y in range(0, height - patch_size + 1, stride):
        for x in range(0, width - patch_size + 1, stride):
            patch = image[y:y+patch_size, x:x+patch_size]
            
            # Skip very uniform patches (likely not informative)
            if patch.std() < 10:
                continue
                
            # Apply all augmentations
            for aug_func in augmentations:
                aug_patch = aug_func(patch)
                if aug_patch.shape[0] == patch_size and aug_patch.shape[1] == patch_size:
                    patches.append(aug_patch)
    
    return patches

def patches_to_tensors(patches):
    """
    Convert patches to tensors for model input
    
    Args:
        patches: List of image patches
        
    Returns:
        Tensor of patches
    """
    # Define the transform
    transform = transforms.Compose([
        transforms.ToTensor(),
        # Add normalization to match the training distribution
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                            std=[0.229, 0.224, 0.225])
    ])
    
    # Process all patches
    patch_tensors = []
    for patch in patches:
        # Convert to RGB if it's BGR (OpenCV format)
        if patch.shape[2] == 3:
            patch_rgb = cv2.cvtColor(patch, cv2.COLOR_BGR2RGB)
        else:
            patch_rgb = patch
            
        # Convert to PIL Image
        pil_patch = Image.fromarray(patch_rgb)
        
        # Apply transform
        tensor = transform(pil_patch)
        patch_tensors.append(tensor)
        
    # Stack all tensors
    if patch_tensors:
        return torch.stack(patch_tensors)
    else:
        return None

def robust_prediction(model, image_path, classifier, confidence_threshold=0.7):
    """
    Makes robust predictions using ensemble of patch predictions
    
    Args:
        model: CNN model for feature extraction
        image_path: Path to the web image
        classifier: SVM or other classifier for final prediction
        confidence_threshold: Threshold for confidence
        
    Returns:
        Dictionary with prediction result and confidence
    """
    # Preprocess image
    processed_image = preprocess_web_image(image_path)
    if processed_image is None:
        return {"error": "Failed to process image"}
    
    # Extract augmented patches
    patches = get_augmented_patches(processed_image)
    if not patches:
        return {"error": "No valid patches extracted from image"}
    
    # Convert to tensors
    patch_tensors = patches_to_tensors(patches)
    if patch_tensors is None:
        return {"error": "Failed to convert patches to tensors"}
    
    # Extract features using model
    model.eval()
    features_list = []
    
    with torch.no_grad():
        for patch_tensor in patch_tensors:
            # Add batch dimension
            patch_tensor = patch_tensor.unsqueeze(0)
            
            # Use CUDA if available
            if torch.cuda.is_available():
                patch_tensor = patch_tensor.cuda()
                model = model.cuda()
            
            # Extract features
            if hasattr(model, 'features'):
                features = model.features(patch_tensor.float())
            else:
                # Forward through the model excluding the classification layer
                # Adapt this based on your specific model architecture
                features = model(patch_tensor.float())
                
            # Add to features list
            features_list.append(features.cpu().numpy().flatten())
    
    # Create feature vector from all patches
    if features_list:
        # Stack features and get average (robust aggregate)
        all_features = np.vstack(features_list)
        avg_features = np.mean(all_features, axis=0).reshape(1, -1)
        
        # Predict using classifier
        prediction = classifier.predict(avg_features)[0]
        
        # Get confidence
        try:
            if hasattr(classifier, 'predict_proba'):
                probabilities = classifier.predict_proba(avg_features)[0]
                confidence = probabilities[prediction]
            elif hasattr(classifier, 'decision_function'):
                decision_score = classifier.decision_function(avg_features)[0]
                confidence = 1.0 / (1.0 + np.exp(-np.abs(decision_score)))
            else:
                confidence = 0.5  # Default confidence
                
            # Return results
            return {
                "prediction": int(prediction),
                "prediction_text": "Tampered" if prediction == 1 else "Authentic",
                "confidence": float(confidence),
                "is_reliable": confidence >= confidence_threshold,
                "num_patches_analyzed": len(patches)
            }
            
        except Exception as e:
            return {"error": f"Error in prediction: {str(e)}"}
    else:
        return {"error": "No features extracted from image"} 