import os
import sys
import torch
import numpy as np
from joblib import load
import cv2

# Add the parent directory to the path so we can import from src
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import the model and feature extraction functions
from src.cnn.cnn import CNN
from src.feature_fusion.feature_vector_generation import get_patch_yi

def test_models():
    print("Starting model test...")
    
    # Get paths
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    cnn_path = os.path.join(project_root, 'data', 'output', 'pre_trained_cnn', 'CASIA2_WithRot_LR001_b128_nodrop.pt')
    svm_path = os.path.join(project_root, 'data', 'output', 'pre_trained_svm', 'CASIA2_WithRot_LR001_b128_nodrop.pt')
    test_image_path = os.path.join(project_root, 'data', 'test_images', 'Au_ani_00002.jpg')
    
    print(f"CNN model path: {cnn_path}")
    print(f"SVM model path: {svm_path}")
    print(f"Test image path: {test_image_path}")
    
    # Check if files exist
    print(f"CNN model exists: {os.path.exists(cnn_path)}")
    print(f"SVM model exists: {os.path.exists(svm_path)}")
    print(f"Test image exists: {os.path.exists(test_image_path)}")
    
    # Load CNN model
    try:
        print("Loading CNN model...")
        with torch.no_grad():
            cnn_model = CNN()
            cnn_model.load_state_dict(torch.load(cnn_path, map_location=lambda storage, loc: storage))
            cnn_model.eval()
            cnn_model = cnn_model.double()
        print("CNN model loaded successfully")
    except Exception as e:
        print(f"Error loading CNN model: {str(e)}")
        return
    
    # Load SVM model
    try:
        print("Loading SVM model...")
        svm_model = load(svm_path)
        print("SVM model loaded successfully")
    except Exception as e:
        print(f"Error loading SVM model: {str(e)}")
        return
    
    # Test feature extraction
    try:
        print("Testing feature extraction...")
        image = cv2.imread(test_image_path)
        if image is None:
            print(f"Failed to load test image: {test_image_path}")
            return
        
        print(f"Image shape: {image.shape}")
        feature_vector = np.empty((1, 400))
        feature_vector[0, :] = get_patch_yi(cnn_model, image)
        print(f"Feature vector shape: {feature_vector.shape}")
        print("Feature extraction successful")
    except Exception as e:
        print(f"Error in feature extraction: {str(e)}")
        return
    
    # Test prediction
    try:
        print("Testing prediction...")
        prediction = svm_model.predict(feature_vector)[0]
        probability = svm_model.predict_proba(feature_vector)[0]
        print(f"Prediction: {prediction} (0=authentic, 1=tampered)")
        print(f"Probability: {probability}")
        print("Prediction successful")
    except Exception as e:
        print(f"Error in prediction: {str(e)}")
        return
    
    print("All tests passed successfully!")

if __name__ == "__main__":
    test_models() 