import os
import sys
import torch
import numpy as np
from skimage import io
import matplotlib.pyplot as plt
from joblib import load
import cv2
import argparse
from datetime import datetime
import traceback
from torch.autograd import Variable
import torchvision.transforms as transforms

# Add the parent directory to the path so we can import from project modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import from project modules
from models.cnn import CNN
from utils.feature_vector_generation import get_patch_yi

def load_cnn_model():
    try:
        model_path = os.path.join('data', 'output', 'pre_trained_cnn', 'CASIA2_WithRot_LR001_b128_nodrop.pt')
        
        if not os.path.exists(model_path):
            print(f"Model file not found at: {model_path}")
            return None
        
        print(f"Loading CNN model from: {model_path}")
        model = CNN()
        model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
        model.eval()
        return model
    except Exception as e:
        print(f"Error loading CNN model: {e}")
        import traceback
        traceback.print_exc()
        return None

def load_svm_model():
    try:
        model_path = os.path.join('data', 'output', 'pre_trained_svm', 'CASIA2_WithRot_LR001_b128_nodrop.pt')
        
        if not os.path.exists(model_path):
            print(f"Model file not found at: {model_path}")
            return None
        
        print(f"Loading SVM model from: {model_path}")
        model = load(model_path)
        return model
    except Exception as e:
        print(f"Error loading SVM model: {e}")
        import traceback
        traceback.print_exc()
        return None

def test_image_processing(image_path):
    # Load models
    print("Loading models...")
    cnn_model = load_cnn_model()
    svm_model = load_svm_model()
    
    if cnn_model is None or svm_model is None:
        print("Failed to load models")
        return
    
    print("Models loaded successfully")
    
    # Load and process image
    try:
        # Load and preprocess the image
        image = cv2.imread(image_path)
        if image is None:
            print(f"Failed to load image: {image_path}")
            return
        
        print(f"Image shape: {image.shape}")
        
        # Convert model to float to match input tensor type
        cnn_model = cnn_model.float()
        
        # Extract patches and features
        print("Extracting features...")
        feature_vector = get_patch_yi(cnn_model, image)
        
        # Reshape to match expected format for classifier
        feature_vector = feature_vector.reshape(1, -1)
        
        # Predict
        print("Making prediction...")
        prediction = svm_model.predict(feature_vector)[0]
        
        # Try to get probability
        try:
            probability = svm_model.predict_proba(feature_vector)[0]
            confidence = float(probability[prediction])
        except:
            print("predict_proba not available, using decision_function instead")
            try:
                decision_scores = svm_model.decision_function(feature_vector)[0]
                confidence = 1.0 / (1.0 + np.exp(-np.abs(decision_scores)))
            except:
                print("decision_function not available, using default confidence")
                confidence = 0.95
        
        print(f"Prediction: {prediction} ({'Tampered' if prediction == 1 else 'Authentic'})")
        print(f"Confidence: {confidence:.2f}")
        
        return True
    except Exception as e:
        print(f"Error processing image: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    # Check if an image path is provided as a command-line argument
    if len(sys.argv) > 1:
        image_path = sys.argv[1]
    else:
        # Use a default image path if none is provided
        image_path = input("Enter the path to an image file: ")
    
    if os.path.exists(image_path):
        test_image_processing(image_path)
    else:
        print(f"Image file not found: {image_path}") 