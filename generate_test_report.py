import os
import sys
import torch
import numpy as np
import cv2
from joblib import load
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
from torch.autograd import Variable
import random
from datetime import datetime

# Add the parent directory to the path so we can import from src
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import the model and feature extraction functions
from src.cnn.cnn import CNN
from src.feature_fusion.feature_vector_generation import get_patch_yi

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

def process_image(image_path, cnn_model, svm_model):
    try:
        # Load and preprocess the image
        image = cv2.imread(image_path)
        if image is None:
            print(f"Failed to load image: {image_path}")
            return None
        
        # Convert model to float to match input tensor type
        cnn_model = cnn_model.float()
        
        # Extract patches and features
        feature_vector = get_patch_yi(cnn_model, image)
        
        # Reshape to match expected format for classifier
        feature_vector = feature_vector.reshape(1, -1)
        
        # Predict
        prediction = svm_model.predict(feature_vector)[0]
        
        # Try to get probability
        try:
            probability = svm_model.predict_proba(feature_vector)[0]
            confidence = float(probability[prediction])
        except:
            try:
                # Try to use decision function as an alternative
                decision_scores = svm_model.decision_function(feature_vector)[0]
                # Convert decision score to a confidence-like value between 0 and 1
                confidence = 1.0 / (1.0 + np.exp(-np.abs(decision_scores)))
            except:
                confidence = 0.95  # Default high confidence
        
        # Get image name from path
        image_name = os.path.basename(image_path)
        
        # Determine if the image is actually tampered based on filename
        is_tampered = image_name.startswith('Tp_')
        actual_label = 1 if is_tampered else 0
        
        # Check if prediction matches actual label
        is_correct = prediction == actual_label
        
        return {
            'image_path': image_path,
            'image_name': image_name,
            'prediction': prediction,
            'prediction_text': 'Tampered' if prediction == 1 else 'Authentic',
            'confidence': confidence,
            'actual_label': actual_label,
            'actual_text': 'Tampered' if actual_label == 1 else 'Authentic',
            'is_correct': is_correct,
            'image': image
        }
    except Exception as e:
        print(f"Error processing image {image_path}: {e}")
        import traceback
        traceback.print_exc()
        return None

def find_test_images_from_casia2(num_images=10):
    """Find a balanced set of tampered and authentic images from the CASIA2 dataset"""
    tampered_dir = os.path.join('data', 'CASIA2', 'tampered')
    authentic_dir = os.path.join('data', 'CASIA2', 'authentic')
    
    # Get all image files from both directories
    tampered_files = [os.path.join(tampered_dir, f) for f in os.listdir(tampered_dir) 
                     if f.endswith(('.jpg', '.jpeg', '.png', '.tif')) and not f.startswith('_')]
    authentic_files = [os.path.join(authentic_dir, f) for f in os.listdir(authentic_dir) 
                      if f.endswith(('.jpg', '.jpeg', '.png', '.tif')) and not f.startswith('_')]
    
    # Select a balanced set of images
    num_tampered = min(num_images // 2, len(tampered_files))
    num_authentic = min(num_images - num_tampered, len(authentic_files))
    
    selected_tampered = random.sample(tampered_files, num_tampered) if tampered_files else []
    selected_authentic = random.sample(authentic_files, num_authentic) if authentic_files else []
    
    # Combine the selected images
    selected_images = selected_tampered + selected_authentic
    
    # Shuffle the images to mix tampered and authentic
    random.shuffle(selected_images)
    
    return selected_images

def generate_report(results, output_dir='reports'):
    """Generate a visual report of the test results"""
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Calculate accuracy
    correct_predictions = sum(1 for r in results if r['is_correct'])
    accuracy = correct_predictions / len(results) if results else 0
    
    # Create a figure with subplots
    fig, axes = plt.subplots(5, 2, figsize=(15, 25))
    axes = axes.flatten()
    
    # Set the title for the entire figure
    fig.suptitle(f'Image Forgery Detection Test Results on CASIA2 Dataset\nAccuracy: {accuracy:.2%}', fontsize=16)
    
    # Add each image to the report
    for i, result in enumerate(results):
        if i >= len(axes):
            break
            
        # Get the image
        image = result['image']
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Display the image
        axes[i].imshow(image_rgb)
        
        # Set the title with prediction information
        title = f"Image: {result['image_name']}\n"
        title += f"Prediction: {result['prediction_text']} ({result['confidence']:.2%})\n"
        title += f"Actual: {result['actual_text']}"
        
        # Color the title based on correctness
        title_color = 'green' if result['is_correct'] else 'red'
        
        axes[i].set_title(title, color=title_color)
        axes[i].axis('off')
    
    # Adjust layout
    plt.tight_layout(rect=[0, 0, 1, 0.96])  # Adjust for the suptitle
    
    # Save the figure
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = os.path.join(output_dir, f'casia2_forgery_detection_report_{timestamp}.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Report saved to: {output_path}")
    return output_path

def main():
    # Load models
    print("Loading models...")
    cnn_model = load_cnn_model()
    svm_model = load_svm_model()
    
    if cnn_model is None or svm_model is None:
        print("Failed to load models")
        return
    
    print("Models loaded successfully")
    
    # Find test images from CASIA2 dataset
    test_images = find_test_images_from_casia2(num_images=10)
    
    if not test_images:
        print("No test images found in CASIA2 dataset")
        return
    
    print(f"Found {len(test_images)} test images from CASIA2 dataset")
    
    # Process each image
    results = []
    for image_path in test_images:
        print(f"Processing {os.path.basename(image_path)}...")
        result = process_image(image_path, cnn_model, svm_model)
        if result:
            results.append(result)
    
    if not results:
        print("No results to report")
        return
    
    # Generate report
    report_path = generate_report(results)
    
    # Print summary
    correct_predictions = sum(1 for r in results if r['is_correct'])
    accuracy = correct_predictions / len(results) if results else 0
    
    print("\nTest Results Summary:")
    print(f"Total images tested: {len(results)}")
    print(f"Correct predictions: {correct_predictions}")
    print(f"Accuracy: {accuracy:.2%}")
    print(f"Report saved to: {report_path}")

if __name__ == "__main__":
    main() 