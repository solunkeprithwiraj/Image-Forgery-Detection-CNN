import os
import torch
import numpy as np
import cv2
from PIL import Image, ImageChops
import PIL.ImageOps
from io import BytesIO
import base64
from datetime import datetime
import uuid
from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
import torch.nn.functional as F
from models.cnn import ImprovedCNN, CNN
from models.SRM_filters import get_filters
from werkzeug.utils import secure_filename
import tempfile
import joblib
import glob

app = Flask(__name__)
CORS(app)

# Configuration
UPLOAD_FOLDER = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'uploads')
OUTPUT_FOLDER = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'outputs')
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'tif', 'tiff'}

# Create upload and output directories if they don't exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# Path to the pre-trained models
CNN_MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 
                             'data/output/pre_trained_cnn/CASIA2_WithRot_LR001_b128_nodrop.pt')
SVM_MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 
                             'data/output/pre_trained_svm/CASIA2_WithRot_LR001_b128_nodrop.pt')

# Load CNN model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def load_cnn_model():
    model = CNN()
    model.load_state_dict(torch.load(CNN_MODEL_PATH, map_location=device))
    model.to(device)
    model.eval()
    return model

def load_ensemble_models():
    models = []
    model_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data/output/pre_trained_cnn')
    for model_file in os.listdir(model_dir):
        if model_file.endswith('.pt'):
            model_path = os.path.join(model_dir, model_file)
            try:
                model = CNN()
                model.load_state_dict(torch.load(model_path, map_location=device))
                model.to(device)
                model.eval()
                models.append({"model": model, "name": model_file})
            except Exception as e:
                print(f"Error loading model {model_file}: {str(e)}")
    return models

# Global variables for models
cnn_model = None
ensemble_models = None

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def preprocess_image(image_path, target_size=(128, 128)):
    """Preprocess the image for model input"""
    try:
        img = cv2.imread(image_path)
        if img is None:
            # Try with PIL if OpenCV fails
            img = Image.open(image_path)
            img = np.array(img)
            if len(img.shape) == 2:  # Convert grayscale to RGB
                img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
            elif img.shape[2] == 4:  # Convert RGBA to RGB
                img = cv2.cvtColor(img, cv2.COLOR_RGBA2RGB)
        
        # Resize the image
        img = cv2.resize(img, target_size)
        
        # Convert to tensor and normalize
        img = img.astype(np.float32) / 255.0
        img = np.transpose(img, (2, 0, 1))  # HWC to CHW
        img = torch.from_numpy(img).unsqueeze(0)  # Add batch dimension
        
        return img
    except Exception as e:
        print(f"Error in preprocessing: {str(e)}")
        return None

def generate_heatmap(img, model):
    """Generate a heatmap highlighting potential tampering areas"""
    try:
        # This is a placeholder implementation - you'll need to modify based on your model's specifics
        # In a real implementation, this might use gradient-based methods like Grad-CAM
        
        # Simple approach: use model's feature maps
        with torch.no_grad():
            img = img.to(device)
            
            # For CNN class, extract features using the features method
            if hasattr(model, 'features'):
                features = model.features(img)
            else:
                # Fallback if features method doesn't exist
                # Forward pass until the last conv layer
                features = model(img)
            
            # Simple activation visualization
            if isinstance(features, torch.Tensor):
                # If features is a tensor, take the mean across channels
                heatmap = torch.mean(features, dim=1).squeeze().cpu().numpy()
            else:
                # If features is not a tensor (e.g., it's the output of a forward pass),
                # create a dummy heatmap
                heatmap = np.ones((img.shape[2], img.shape[3]), dtype=np.float32)
            
            # Ensure heatmap has a valid shape
            if heatmap.size == 0:
                heatmap = np.ones((img.shape[2], img.shape[3]), dtype=np.float32)
                
            # Resize to a consistent size
            heatmap = cv2.resize(heatmap, (128, 128))
            
            # Normalize the heatmap
            heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + 1e-8)
            heatmap = np.uint8(255 * heatmap)
            
            # Apply colormap
            heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
            
            return heatmap
    except Exception as e:
        print(f"Error generating heatmap: {str(e)}")
        return None

def error_localization(img_path, model, method='heatmap'):
    """Localize the tampering in the image using various methods"""
    img = preprocess_image(img_path)
    if img is None:
        return None
    
    original_img = cv2.imread(img_path)
    original_shape = original_img.shape[:2]  # (height, width)
    
    if method == 'heatmap':
        heatmap = generate_heatmap(img, model)
        if heatmap is None:
            return None
        # Resize heatmap to match original image
        heatmap = cv2.resize(heatmap, (original_shape[1], original_shape[0]))
        return heatmap
    
    elif method == 'ela':
        # Error Level Analysis
        ela_image = perform_ela_analysis(img_path, quality=90, scale=15)
        if ela_image is None:
            return None
        
        # Resize ELA image to match original image
        ela_image = cv2.resize(ela_image, (original_shape[1], original_shape[0]))
        return ela_image
    
    elif method == 'overlay':
        # Create an overlay of the heatmap on the original image
        heatmap = generate_heatmap(img, model)
        if heatmap is None:
            return None
        
        # Resize heatmap to match original image
        heatmap = cv2.resize(heatmap, (original_shape[1], original_shape[0]))
        
        # Create overlay with transparency
        overlay = cv2.addWeighted(original_img, 0.7, heatmap, 0.3, 0)
        return overlay
    
    elif method == 'contour':
        # Generate contours around potential tampered regions
        heatmap = generate_heatmap(img, model)
        if heatmap is None:
            return None
        
        # Resize heatmap to a consistent size for processing
        heatmap_resized = cv2.resize(heatmap, (128, 128))
        
        # Convert heatmap to grayscale for contour detection
        gray_heatmap = cv2.cvtColor(heatmap_resized, cv2.COLOR_BGR2GRAY)
        
        # Apply threshold to create binary image
        _, thresh = cv2.threshold(gray_heatmap, 150, 255, cv2.THRESH_BINARY)
        
        # Find contours
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Create a blank image for drawing contours
        contour_img = np.zeros((128, 128, 3), dtype=np.uint8)
        
        # Draw contours on blank image
        cv2.drawContours(contour_img, contours, -1, (0, 255, 0), 2)
        
        # Resize contour image to match original image
        contour_img = cv2.resize(contour_img, (original_shape[1], original_shape[0]))
        
        # Overlay contours on original image
        result = cv2.addWeighted(original_img, 0.9, contour_img, 0.8, 0)
        return result
    
    elif method == 'mask':
        # Create a binary mask of potential tampered regions
        heatmap = generate_heatmap(img, model)
        if heatmap is None:
            return None
        
        # Process at a consistent size
        heatmap_resized = cv2.resize(heatmap, (128, 128))
        
        # Convert heatmap to grayscale
        gray_heatmap = cv2.cvtColor(heatmap_resized, cv2.COLOR_BGR2GRAY)
        
        # Apply threshold to create binary mask
        _, mask = cv2.threshold(gray_heatmap, 150, 255, cv2.THRESH_BINARY)
        
        # Create a colored mask (red for tampered regions)
        mask_colored = np.zeros((128, 128, 3), dtype=np.uint8)
        mask_colored[:, :, 2] = mask  # Red channel
        
        # Resize mask to match original image
        mask_colored = cv2.resize(mask_colored, (original_shape[1], original_shape[0]))
        
        # Combine original and mask
        result = cv2.addWeighted(original_img, 0.8, mask_colored, 0.7, 0)
        return result
    
    elif method == 'edge':
        # Detect edges in the ELA result
        ela_image = perform_ela_analysis(img_path)
        if ela_image is None:
            return None
        
        # Process at a consistent size
        ela_resized = cv2.resize(ela_image, (128, 128))
        
        # Convert to grayscale if needed
        if len(ela_resized.shape) == 3:
            ela_gray = cv2.cvtColor(ela_resized, cv2.COLOR_BGR2GRAY)
        else:
            ela_gray = ela_resized
        
        # Apply edge detection
        edges = cv2.Canny(ela_gray, 100, 200)
        
        # Create colored edge overlay
        edge_overlay = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
        edge_overlay[:, :, 0] = 0  # Set blue channel to 0
        edge_overlay[:, :, 1] = 0  # Set green channel to 0
        
        # Resize edge overlay to match original image
        edge_overlay = cv2.resize(edge_overlay, (original_shape[1], original_shape[0]))
        
        # Combine original and edges
        result = cv2.addWeighted(original_img, 0.8, edge_overlay, 1.0, 0)
        return result
    
    elif method == 'highlight':
        # Highlight the anomalies from ELA
        ela_image = perform_ela_analysis(img_path)
        if ela_image is None:
            return None
        
        # Process at a consistent size
        ela_resized = cv2.resize(ela_image, (128, 128))
        
        # Convert to grayscale if needed
        if len(ela_resized.shape) == 3:
            ela_gray = cv2.cvtColor(ela_resized, cv2.COLOR_BGR2GRAY)
        else:
            ela_gray = ela_resized
        
        # Apply adaptive threshold to identify anomalies
        _, thresh = cv2.threshold(ela_gray, np.mean(ela_gray) * 1.5, 255, cv2.THRESH_BINARY)
        
        # Create highlight overlay
        highlight = np.zeros((128, 128, 3), dtype=np.uint8)
        highlight[:, :, 2] = thresh  # Red channel
        
        # Resize highlight to match original image
        highlight = cv2.resize(highlight, (original_shape[1], original_shape[0]))
        
        # Combine original and highlight
        result = cv2.addWeighted(original_img, 0.8, highlight, 0.6, 0)
        return result
    
    # Default return if method not implemented
    return None

def resize_output_to_original(image, original_path):
    """Resize an output image to match the dimensions of the original image"""
    try:
        # Read original image to get its dimensions
        original_img = cv2.imread(original_path)
        if original_img is None:
            # Try with PIL if OpenCV fails
            with Image.open(original_path) as img:
                original_shape = img.size
                # Convert shape from (width, height) to (height, width)
                original_shape = (original_shape[1], original_shape[0])
        else:
            original_shape = original_img.shape[:2]
        
        # Resize the image
        resized_image = cv2.resize(image, (original_shape[1], original_shape[0]))
        return resized_image
    except Exception as e:
        print(f"Error resizing output image: {str(e)}")
        return image  # Return original image if resizing fails

@app.route('/api/analyze', methods=['POST'])
def analyze_image():
    global cnn_model
    
    # Load model if not already loaded
    if cnn_model is None:
        cnn_model = load_cnn_model()
    
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    
    file = request.files['file']
    
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    
    if file and allowed_file(file.filename):
        # Generate unique filename
        filename = str(uuid.uuid4()) + '_' + secure_filename(file.filename)
        file_path = os.path.join(UPLOAD_FOLDER, filename)
        file.save(file_path)
        
        # Process the image
        img = preprocess_image(file_path)
        if img is None:
            return jsonify({'error': 'Failed to process image'}), 400
        
        # Analyze with model
        with torch.no_grad():
            img = img.to(device)
            output = cnn_model(img)
            
            # For CNN model, the output in test mode is the raw probabilities
            # but we need to make sure the model is in evaluation mode
            cnn_model.eval()
            
            # Get the prediction
            pred_class = torch.argmax(output, dim=1).item()
            confidence = output[0][pred_class].item()
        
        # Generate timestamp
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        # Check if localization is requested
        show_localization = request.form.get('show_localization', 'false').lower() == 'true'
        show_ela = request.form.get('show_ela', 'false').lower() == 'true'
        localization_methods = request.form.getlist('localization_methods[]')
        
        result = {
            'is_tampered': bool(pred_class),
            'confidence': confidence,
            'message': 'Image is likely tampered' if pred_class else 'Image appears authentic',
            'method': 'CNN',
            'timestamp': timestamp,
            'input_image_path': f'/uploads/{filename}'
        }
        
        # Generate ELA analysis if requested
        if show_ela:
            ela_image = perform_ela_analysis(file_path)
            if ela_image is not None:
                # Resize ELA image for display
                ela_image_display = resize_output_to_original(ela_image, file_path)
                
                # Save the ELA image
                ela_filename = f"ela_{filename}"
                ela_path = os.path.join(OUTPUT_FOLDER, ela_filename)
                cv2.imwrite(ela_path, ela_image_display)
                result['ela_path'] = f'/outputs/{ela_filename}'
        
        # Generate localizations if requested
        if show_localization and localization_methods:
            for method in localization_methods:
                if method in ['heatmap', 'overlay', 'contour', 'mask', 'edge', 'highlight']:
                    localization_img = error_localization(file_path, cnn_model, method)
                    if localization_img is not None:
                        # Save the localization image
                        loc_filename = f"{method}_{filename}"
                        loc_path = os.path.join(OUTPUT_FOLDER, loc_filename)
                        cv2.imwrite(loc_path, localization_img)
                        
                        # Add path to result
                        result[f'{method}_path'] = f'/outputs/{loc_filename}'
        
        return jsonify(result)
    
    return jsonify({'error': 'File type not allowed'}), 400

@app.route('/api/analyze/ensemble', methods=['POST'])
def analyze_image_ensemble():
    global ensemble_models
    
    # Load models if not already loaded
    if ensemble_models is None:
        ensemble_models = load_ensemble_models()
    
    if not ensemble_models:
        return jsonify({'error': 'No ensemble models available'}), 500
    
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    
    file = request.files['file']
    
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    
    if file and allowed_file(file.filename):
        # Generate unique filename
        filename = str(uuid.uuid4()) + '_' + secure_filename(file.filename)
        file_path = os.path.join(UPLOAD_FOLDER, filename)
        file.save(file_path)
        
        # Process the image
        img = preprocess_image(file_path)
        if img is None:
            return jsonify({'error': 'Failed to process image'}), 400
        
        # Analyze with ensemble of models
        ensemble_results = []
        tampered_votes = 0
        
        for model_info in ensemble_models:
            model = model_info["model"]
            model_name = model_info["name"]
            
            # Ensure model is in evaluation mode
            model.eval()
            
            with torch.no_grad():
                img_tensor = img.to(device)
                output = model(img_tensor)
                
                # Get prediction
                pred_class = torch.argmax(output, dim=1).item()
                confidence = output[0][pred_class].item()
                
                # Count vote
                if pred_class:  # If tampered
                    tampered_votes += 1
                
                # Add to ensemble results
                ensemble_results.append({
                    'model_name': model_name,
                    'prediction': 'tampered' if pred_class else 'authentic',
                    'confidence': confidence
                })
        
        # Generate timestamp
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        # Calculate ensemble decision
        ensemble_size = len(ensemble_models)
        is_tampered = tampered_votes > ensemble_size // 2
        
        # Determine consensus level
        if tampered_votes == ensemble_size or tampered_votes == 0:
            consensus = "Strong"
        elif abs(tampered_votes - (ensemble_size - tampered_votes)) <= 1:
            consensus = "Weak"
        else:
            consensus = "Moderate"
        
        # Check if localization is requested
        show_localization = request.form.get('show_localization', 'false').lower() == 'true'
        show_ela = request.form.get('show_ela', 'false').lower() == 'true'
        localization_methods = request.form.getlist('localization_methods[]')
        
        # Prepare result
        result = {
            'is_tampered': is_tampered,
            'confidence': tampered_votes / ensemble_size if is_tampered else (ensemble_size - tampered_votes) / ensemble_size,
            'message': f'Ensemble predicts image is {("tampered" if is_tampered else "authentic")} with {consensus.lower()} consensus',
            'method': 'Ensemble CNN',
            'timestamp': timestamp,
            'input_image_path': f'/uploads/{filename}',
            'ensemble_detail': {
                'ensemble_size': ensemble_size,
                'tampered_votes': tampered_votes,
                'authentic_votes': ensemble_size - tampered_votes,
                'consensus_level': consensus,
                'model_predictions': ensemble_results
            }
        }
        
        # Generate ELA analysis if requested
        if show_ela:
            ela_image = perform_ela_analysis(file_path)
            if ela_image is not None:
                # Resize ELA image for display
                ela_image_display = resize_output_to_original(ela_image, file_path)
                
                # Save the ELA image
                ela_filename = f"ela_{filename}"
                ela_path = os.path.join(OUTPUT_FOLDER, ela_filename)
                cv2.imwrite(ela_path, ela_image_display)
                result['ela_path'] = f'/outputs/{ela_filename}'
        
        # Generate localizations if requested
        if show_localization and localization_methods:
            # Use the most confident model for localization
            best_model = None
            best_confidence = 0
            
            for model_info in ensemble_models:
                model = model_info["model"]
                model.eval()
                
                with torch.no_grad():
                    img_tensor = img.to(device)
                    output = model(img_tensor)
                    
                    pred_class = torch.argmax(output, dim=1).item()
                    confidence = output[0][pred_class].item()
                    
                    if confidence > best_confidence:
                        best_confidence = confidence
                        best_model = model
            
            if best_model:
                for method in localization_methods:
                    try:
                        if method in ['heatmap', 'overlay', 'contour', 'mask', 'edge', 'highlight']:
                            localization_img = error_localization(file_path, best_model, method)
                            if localization_img is not None:
                                # Save the localization image
                                loc_filename = f"{method}_{filename}"
                                loc_path = os.path.join(OUTPUT_FOLDER, loc_filename)
                                cv2.imwrite(loc_path, localization_img)
                                
                                # Add path to result
                                result[f'{method}_path'] = f'/outputs/{loc_filename}'
                    except Exception as e:
                        print(f"Error processing localization method {method}: {str(e)}")
                        continue
        
        return jsonify(result)
    
    return jsonify({'error': 'File type not allowed'}), 400

@app.route('/api/convert-tiff', methods=['POST'])
def convert_tiff():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    
    file = request.files['file']
    
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    
    if file and file.filename.lower().endswith(('.tif', '.tiff')):
        # Save the TIFF file
        tiff_filename = str(uuid.uuid4()) + '_' + secure_filename(file.filename)
        tiff_path = os.path.join(UPLOAD_FOLDER, tiff_filename)
        file.save(tiff_path)
        
        # Convert TIFF to JPEG
        try:
            img = Image.open(tiff_path)
            jpeg_filename = tiff_filename.rsplit('.', 1)[0] + '.jpg'
            jpeg_path = os.path.join(OUTPUT_FOLDER, jpeg_filename)
            
            # Convert to RGB if needed
            if img.mode != 'RGB':
                img = img.convert('RGB')
                
            img.save(jpeg_path, 'JPEG')
            
            return jsonify({'preview_url': f'/outputs/{jpeg_filename}'})
        except Exception as e:
            return jsonify({'error': f'Failed to convert TIFF: {str(e)}'}), 500
    
    return jsonify({'error': 'Not a TIFF file'}), 400

@app.route('/api/view-tiff/<path:tiff_path>', methods=['GET'])
def view_tiff(tiff_path):
    try:
        # Ensure the path is secure
        if '..' in tiff_path:
            return jsonify({'error': 'Invalid path'}), 400
        
        # Form the complete path
        complete_path = os.path.join(UPLOAD_FOLDER, tiff_path)
        
        # Convert TIFF to JPEG
        img = Image.open(complete_path)
        
        # Convert to RGB if needed
        if img.mode != 'RGB':
            img = img.convert('RGB')
        
        # Save as JPEG
        jpeg_filename = tiff_path.rsplit('.', 1)[0] + '.jpg'
        jpeg_path = os.path.join(OUTPUT_FOLDER, jpeg_filename)
        img.save(jpeg_path, 'JPEG')
        
        return jsonify({'preview_url': f'/outputs/{jpeg_filename}'})
    except Exception as e:
        return jsonify({'error': f'Failed to process TIFF: {str(e)}'}), 500

# Serve static files
@app.route('/uploads/<path:filename>')
def serve_upload(filename):
    return send_file(os.path.join(UPLOAD_FOLDER, filename))

@app.route('/outputs/<path:filename>')
def serve_output(filename):
    return send_file(os.path.join(OUTPUT_FOLDER, filename))

def perform_ela_analysis(image_path, quality=90, scale=15):
    """
    Performs Error Level Analysis (ELA) on an image to detect potential manipulation.
    
    ELA works by resaving an image at a specified quality level and comparing the differences.
    Areas with high differences often indicate manipulation.
    
    Args:
        image_path: Path to the image file
        quality: JPEG quality level to resave the image at (0-100)
        scale: Scale factor to amplify the differences
        
    Returns:
        Numpy array containing the ELA image
    """
    try:
        # Open the original image
        original = Image.open(image_path).convert('RGB')
        
        # Create a temporary file for the resaved image
        temp_file = os.path.join(tempfile.gettempdir(), 'ela_temp.jpg')
        
        # Save the image with specified quality
        original.save(temp_file, 'JPEG', quality=quality)
        
        # Open the resaved image
        resaved = Image.open(temp_file)
        
        # Calculate the difference between the original and resaved images
        ela_image = ImageChops.difference(original, resaved)
        
        # Scale the difference to make it more visible
        extrema = ela_image.getextrema()
        max_diff = max([ex[1] for ex in extrema])
        if max_diff == 0:
            # No difference found, return grayscale version of original
            gray_image = PIL.ImageOps.grayscale(original)
            # Resize to a consistent size for processing
            gray_image = gray_image.resize((128, 128))
            return np.array(gray_image)
        
        # Scale the difference by multiplying the pixel values
        ela_image = ImageChops.multiply(ela_image, Image.new('RGB', ela_image.size, (scale, scale, scale)))
        
        # Resize to a consistent size for processing
        ela_image = ela_image.resize((128, 128))
        
        # Convert to numpy array
        ela_array = np.array(ela_image)
        
        return ela_array
    
    except Exception as e:
        print(f"Error in ELA analysis: {str(e)}")
        return None

@app.route('/api/analyze/ela', methods=['POST'])
def analyze_image_ela():
    """Endpoint for performing standalone Error Level Analysis on an image"""
    
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    
    file = request.files['file']
    
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    
    if file and allowed_file(file.filename):
        # Generate unique filename
        filename = str(uuid.uuid4()) + '_' + secure_filename(file.filename)
        file_path = os.path.join(UPLOAD_FOLDER, filename)
        file.save(file_path)
        
        # Get quality parameter if provided
        try:
            quality = int(request.form.get('quality', 90))
            # Ensure quality is in valid range
            quality = max(1, min(100, quality))
        except ValueError:
            quality = 90
        
        # Get scale parameter if provided
        try:
            scale = int(request.form.get('scale', 15))
            # Ensure scale is in valid range
            scale = max(1, min(50, scale))
        except ValueError:
            scale = 15
        
        # Perform ELA analysis
        ela_image = perform_ela_analysis(file_path, quality=quality, scale=scale)
        
        if ela_image is None:
            return jsonify({'error': 'Failed to generate ELA image'}), 500
        
        # Resize ELA image for display
        ela_image_display = resize_output_to_original(ela_image, file_path)
        
        # Save the ELA image
        ela_filename = f"ela_{filename}"
        ela_path = os.path.join(OUTPUT_FOLDER, ela_filename)
        cv2.imwrite(ela_path, ela_image_display)
        
        # Generate timestamp
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        # Prepare result
        result = {
            'message': 'ELA analysis completed successfully',
            'method': 'ELA',
            'timestamp': timestamp,
            'input_image_path': f'/uploads/{filename}',
            'ela_path': f'/outputs/{ela_filename}',
            'parameters': {
                'quality': quality,
                'scale': scale
            }
        }
        
        return jsonify(result)
    
    return jsonify({'error': 'File type not allowed'}), 400

@app.route('/api/analyze/batch', methods=['POST'])
def analyze_batch():
    """Analyze a batch of images from a specified directory"""
    global cnn_model
    
    # Load model if not already loaded
    if cnn_model is None:
        cnn_model = load_cnn_model()
    
    if 'directory' not in request.json:
        return jsonify({'error': 'Directory path is required'}), 400
    
    directory_path = request.json['directory']
    use_ensemble = request.json.get('use_ensemble', False)
    
    # Check if directory exists
    if not os.path.isdir(directory_path):
        return jsonify({'error': f'Directory not found: {directory_path}'}), 404
    
    # Get all image files in the directory
    image_files = []
    for ext in ALLOWED_EXTENSIONS:
        image_files.extend(glob.glob(os.path.join(directory_path, f'*.{ext}')))
    
    if not image_files:
        return jsonify({'error': 'No valid images found in directory'}), 404
    
    results = []
    for img_path in image_files:
        try:
            # Process the image
            img = preprocess_image(img_path)
            if img is None:
                results.append({
                    'filename': os.path.basename(img_path),
                    'status': 'error',
                    'message': 'Failed to process image'
                })
                continue
            
            # Use the appropriate model based on user request
            if use_ensemble and ensemble_models is None:
                # Load ensemble models if needed
                global ensemble_models
                ensemble_models = load_ensemble_models()
            
            if use_ensemble and ensemble_models:
                # Analyze with ensemble of models
                tampered_votes = 0
                model_predictions = []
                
                for model_info in ensemble_models:
                    model = model_info["model"]
                    model_name = model_info["name"]
                    
                    # Ensure model is in evaluation mode
                    model.eval()
                    
                    with torch.no_grad():
                        img_tensor = img.to(device)
                        output = model(img_tensor)
                        
                        # Get prediction
                        pred_class = torch.argmax(output, dim=1).item()
                        confidence = output[0][pred_class].item()
                        
                        # Count vote
                        if pred_class:  # If tampered
                            tampered_votes += 1
                        
                        model_predictions.append({
                            'model_name': model_name,
                            'prediction': 'tampered' if pred_class else 'authentic',
                            'confidence': confidence
                        })
                
                # Calculate ensemble decision
                ensemble_size = len(ensemble_models)
                is_tampered = tampered_votes > ensemble_size // 2
                
                # Determine consensus level
                if tampered_votes == ensemble_size or tampered_votes == 0:
                    consensus = "Strong"
                elif abs(tampered_votes - (ensemble_size - tampered_votes)) <= 1:
                    consensus = "Weak"
                else:
                    consensus = "Moderate"
                
                results.append({
                    'filename': os.path.basename(img_path),
                    'status': 'success',
                    'is_tampered': is_tampered,
                    'confidence': tampered_votes / ensemble_size if is_tampered else (ensemble_size - tampered_votes) / ensemble_size,
                    'method': 'Ensemble CNN',
                    'ensemble_detail': {
                        'ensemble_size': ensemble_size,
                        'tampered_votes': tampered_votes,
                        'authentic_votes': ensemble_size - tampered_votes,
                        'consensus_level': consensus,
                        'model_predictions': model_predictions
                    }
                })
            else:
                # Analyze with single model
                with torch.no_grad():
                    img_tensor = img.to(device)
                    output = cnn_model(img_tensor)
                    
                    # For CNN model, the output in test mode is the raw probabilities
                    cnn_model.eval()
                    
                    # Get the prediction
                    pred_class = torch.argmax(output, dim=1).item()
                    confidence = output[0][pred_class].item()
                
                results.append({
                    'filename': os.path.basename(img_path),
                    'status': 'success',
                    'is_tampered': bool(pred_class),
                    'confidence': confidence,
                    'method': 'CNN'
                })
        except Exception as e:
            results.append({
                'filename': os.path.basename(img_path),
                'status': 'error',
                'message': str(e)
            })
    
    return jsonify({
        'total_images': len(image_files),
        'processed_images': len(results),
        'tampered_count': sum(1 for r in results if r.get('status') == 'success' and r.get('is_tampered')),
        'authentic_count': sum(1 for r in results if r.get('status') == 'success' and not r.get('is_tampered')),
        'error_count': sum(1 for r in results if r.get('status') == 'error'),
        'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        'results': results
    })

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000) 