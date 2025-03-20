import os
import sys
import torch
import numpy as np
from flask import Flask, render_template, request, jsonify
from werkzeug.utils import secure_filename
import cv2
from joblib import load
import uuid
import traceback

# Add the parent directory to the path so we can import from src
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import the model and feature extraction functions
from src.cnn.cnn import CNN
from src.feature_fusion.feature_vector_generation import get_patch_yi

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'static', 'uploads')
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max upload size

# Create upload folder if it doesn't exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Load the CNN model
def load_cnn_model():
    try:
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        model_path = os.path.join(project_root, 'data', 'output', 'pre_trained_cnn', 'CASIA2_WithRot_LR001_b128_nodrop.pt')
        print(f"Loading CNN model from: {model_path}")
        
        with torch.no_grad():
            model = CNN()
            model.load_state_dict(torch.load(model_path, map_location=lambda storage, loc: storage))
            model.eval()
            model = model.double()
        return model
    except Exception as e:
        print(f"Error loading CNN model: {str(e)}")
        traceback.print_exc()
        return None

# Load the SVM model
def load_svm_model():
    try:
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        svm_path = os.path.join(project_root, 'data', 'output', 'pre_trained_svm', 'CASIA2_WithRot_LR001_b128_nodrop.pt')
        print(f"Loading SVM model from: {svm_path}")
        return load(svm_path)
    except Exception as e:
        print(f"Error loading SVM model: {str(e)}")
        traceback.print_exc()
        return None

# Get feature vector for an image
def get_feature_vector(image_path, model):
    try:
        print(f"Processing image: {image_path}")
        image = cv2.imread(image_path)
        if image is None:
            print(f"Failed to load image: {image_path}")
            return None
        
        print(f"Image shape: {image.shape}")
        feature_vector = np.empty((1, 400))
        feature_vector[0, :] = get_patch_yi(model, image)
        return feature_vector
    except Exception as e:
        print(f"Error extracting features: {str(e)}")
        traceback.print_exc()
        return None

# Load models
print("Loading models...")
cnn_model = load_cnn_model()
svm_model = load_svm_model()
print("Models loaded successfully" if cnn_model is not None and svm_model is not None else "Failed to load models")

# Allowed file extensions
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'bmp', 'tiff'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/analyze', methods=['POST'])
def analyze():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    
    file = request.files['file']
    
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    
    if file and allowed_file(file.filename):
        try:
            # Generate a unique filename
            filename = str(uuid.uuid4()) + '_' + secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            print(f"File saved to: {filepath}")
            
            # Check if models are loaded
            if cnn_model is None or svm_model is None:
                print("Models not loaded properly")
                return jsonify({'error': 'Models not loaded properly'}), 500
            
            # Extract features and predict
            feature_vector = get_feature_vector(filepath, cnn_model)
            if feature_vector is None:
                print("Failed to extract features")
                return jsonify({'error': 'Failed to extract features'}), 500
            
            prediction = svm_model.predict(feature_vector)[0]
            
            # Try to get probability if available, otherwise use a default confidence
            try:
                probability = svm_model.predict_proba(feature_vector)[0]
                confidence = float(probability[prediction])
            except:
                print("predict_proba not available, using decision_function instead")
                try:
                    # Try to use decision function as an alternative
                    decision_scores = svm_model.decision_function(feature_vector)[0]
                    # Convert decision score to a confidence-like value between 0 and 1
                    confidence = 1.0 / (1.0 + np.exp(-np.abs(decision_scores)))
                except:
                    print("decision_function not available, using default confidence")
                    confidence = 0.95  # Default high confidence
            
            print(f"Prediction: {prediction}, Confidence: {confidence}")
            
            # Format the result
            result = {
                'prediction': int(prediction),
                'prediction_text': 'Tampered' if prediction == 1 else 'Authentic',
                'confidence': float(confidence),
                'image_path': os.path.join('uploads', filename)
            }
            
            return jsonify(result)
        except Exception as e:
            print(f"Error in analyze route: {str(e)}")
            traceback.print_exc()
            return jsonify({'error': str(e)}), 500
    
    return jsonify({'error': 'File type not allowed'}), 400

if __name__ == '__main__':
    app.run(debug=True) 