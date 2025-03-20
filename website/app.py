import os
import sys
import torch
import numpy as np
from flask import Flask, render_template, request, jsonify, after_this_request
from flask_cors import CORS
from werkzeug.utils import secure_filename
import cv2
from joblib import load
import uuid
import traceback
import matplotlib.pyplot as plt
import matplotlib
import torchvision.transforms as transforms
from torch.autograd import Variable
import time
matplotlib.use('Agg')  # Use Agg backend to avoid GUI issues

# Add the parent directory to the path so we can import from src
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import the model and feature extraction functions
from src.cnn.cnn import CNN
from src.feature_fusion.feature_vector_generation import get_patch_yi

app = Flask(__name__, static_url_path='/static', static_folder='static')
# Enable CORS for all routes
CORS(app, resources={r"/*": {"origins": "*", "supports_credentials": True}})
app.config['UPLOAD_FOLDER'] = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'static', 'uploads')
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max upload size

# Create upload folder if it doesn't exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Disable caching for all routes
@app.after_request
def add_header(response):
    response.headers['Cache-Control'] = 'no-store, no-cache, must-revalidate, post-check=0, pre-check=0, max-age=0'
    response.headers['Pragma'] = 'no-cache'
    response.headers['Expires'] = '-1'
    # Add CORS headers to ensure static files can be loaded
    response.headers['Access-Control-Allow-Origin'] = '*'
    response.headers['Access-Control-Allow-Methods'] = 'GET, POST, OPTIONS'
    response.headers['Access-Control-Allow-Headers'] = 'Content-Type'
    return response

# Load the CNN model
def load_cnn_model():
    try:
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        model_path = os.path.join(project_root, 'data', 'output', 'pre_trained_cnn', 'CASIA2_WithRot_LR001_b128_nodrop.pt')
        
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
        traceback.print_exc()
        return None

# Load the SVM model
def load_svm_model():
    try:
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        model_path = os.path.join(project_root, 'data', 'output', 'pre_trained_svm', 'CASIA2_WithRot_LR001_b128_nodrop.pt')
        
        if not os.path.exists(model_path):
            print(f"Model file not found at: {model_path}")
            return None
        
        print(f"Loading SVM model from: {model_path}")
        model = load(model_path)
        return model
    except Exception as e:
        print(f"Error loading SVM model: {e}")
        traceback.print_exc()
        return None

# Global variables for models
print("Loading models...")
cnn_model = load_cnn_model()
svm_model = load_svm_model()
print("Models loaded successfully" if cnn_model and svm_model else "Failed to load models")

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in {'png', 'jpg', 'jpeg', 'gif', 'tif', 'tiff'}

def get_feature_vector(image_path, model):
    try:
        # Check file extension
        file_ext = os.path.splitext(image_path)[1].lower()
        
        # Load and preprocess the image
        if file_ext in ['.tif', '.tiff']:
            # Use PIL for TIFF files
            from PIL import Image
            import numpy as np
            
            print(f"Loading TIFF image: {image_path}")
            pil_image = Image.open(image_path)
            # Convert to RGB if needed
            if pil_image.mode != 'RGB':
                pil_image = pil_image.convert('RGB')
            # Convert PIL image to numpy array for processing
            image = np.array(pil_image)
            # Convert from RGB to BGR for OpenCV compatibility
            image = image[:, :, ::-1].copy()
        else:
            # Use OpenCV for other image formats
            image = cv2.imread(image_path)
            
        if image is None:
            print(f"Failed to load image: {image_path}")
            return None
        
        print(f"Image shape: {image.shape}")
        
        # Convert model to float to match input tensor type
        model = model.float()
        
        # Extract patches and features using the correct function call
        # Pass both model and image to get_patch_yi
        feature_vector = get_patch_yi(model, image)
        
        # Reshape to match expected format for classifier
        feature_vector = feature_vector.reshape(1, -1)
        return feature_vector
    except Exception as e:
        print(f"Error extracting features: {e}")
        traceback.print_exc()
        return None

def localize_tampering(image_path, model, patch_size=64, stride=16, localization_methods=None):
    """
    Localize tampering in an image by analyzing patches and creating a heatmap.
    
    Args:
        image_path: Path to the image
        model: CNN model for feature extraction
        patch_size: Size of patches to analyze
        stride: Stride for sliding window (smaller stride for better resolution)
        localization_methods: List of localization methods to use
        
    Returns:
        Dictionary with paths to generated visualization images
    """
    try:
        import numpy as np
        
        # Default localization methods if none specified
        if localization_methods is None:
            localization_methods = ['heatmap', 'overlay', 'contour']
        
        # Check file extension
        file_ext = os.path.splitext(image_path)[1].lower()
        
        # Load the image
        if file_ext in ['.tif', '.tiff']:
            # Use PIL for TIFF files
            from PIL import Image
            import numpy as np
            
            print(f"Loading TIFF image for localization: {image_path}")
            pil_image = Image.open(image_path)
            # Convert to RGB if needed
            if pil_image.mode != 'RGB':
                pil_image = pil_image.convert('RGB')
            # Convert PIL image to numpy array for processing
            image = np.array(pil_image)
            # Convert from RGB to BGR for OpenCV compatibility
            image = image[:, :, ::-1].copy()
        else:
            # Use OpenCV for other image formats
            image = cv2.imread(image_path)
            
        if image is None:
            print(f"Failed to load image for localization: {image_path}")
            return {}
            
        # Create a heatmap of the same size as the image
        height, width = image.shape[:2]
        heatmap = np.zeros((height, width), dtype=np.float32)
        count = np.zeros((height, width), dtype=np.float32)
        
        print(f"Analyzing image for tampering localization, size: {width}x{height}")
        print(f"Using patch size: {patch_size}, stride: {stride}")
        
        # Create transform for patches
        transform = transforms.Compose([transforms.ToTensor()])
        
        # Sliding window to extract patches
        patch_scores = []
        patch_positions = []
        
        for y in range(0, height - patch_size + 1, stride):
            for x in range(0, width - patch_size + 1, stride):
                # Extract patch
                patch = image[y:y+patch_size, x:x+patch_size]
                
                # Convert to RGB if needed and normalize
                if patch.shape[2] == 3:  # RGB
                    patch_rgb = cv2.cvtColor(patch, cv2.COLOR_BGR2RGB)
                else:
                    patch_rgb = patch
                
                # Convert to tensor for model input
                img_tensor = transform(patch_rgb)
                img_tensor.unsqueeze_(0)  # Add batch dimension
                
                # Get features and prediction
                with torch.no_grad():
                    # Extract features using the model
                    features = model.features(img_tensor.float()).cpu().numpy().flatten().reshape(1, -1)
                    
                    # Predict using SVM
                    if hasattr(svm_model, 'decision_function'):
                        score = svm_model.decision_function(features)[0]
                        # Convert to probability-like value
                        prob = 1.0 / (1.0 + np.exp(-score))
                    else:
                        # If decision_function is not available, use predict
                        pred = svm_model.predict(features)[0]
                        prob = 0.9 if pred == 1 else 0.1
                
                # Store patch score and position for later analysis
                patch_scores.append(prob)
                patch_positions.append((x, y, x+patch_size, y+patch_size))
                
                # Add to heatmap (higher value = more likely to be tampered)
                heatmap[y:y+patch_size, x:x+patch_size] += prob
                count[y:y+patch_size, x:x+patch_size] += 1
        
        # Normalize heatmap
        count[count == 0] = 1  # Avoid division by zero
        heatmap = heatmap / count
        
        # Apply Gaussian blur to smooth the heatmap
        heatmap = cv2.GaussianBlur(heatmap, (15, 15), 0)
        
        # Normalize to 0-1 range
        heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + 1e-8)
        
        # Create a binary mask for tampered regions (threshold the heatmap)
        # Adaptive thresholding based on the distribution of values
        threshold = np.mean(heatmap) + 0.8 * np.std(heatmap)
        binary_mask = (heatmap > threshold).astype(np.uint8) * 255
        
        # Apply morphological operations to clean up the mask
        kernel = np.ones((5, 5), np.uint8)
        binary_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_OPEN, kernel)
        binary_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_CLOSE, kernel)
        
        # Find contours in the binary mask
        contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Filter contours by area to remove small noise
        min_contour_area = (width * height) * 0.001  # Minimum 0.1% of image area
        contours = [cnt for cnt in contours if cv2.contourArea(cnt) > min_contour_area]
        
        result_paths = {}
        
        # Generate requested visualizations
        if 'heatmap' in localization_methods:
            # Create heatmap visualization
            plt.figure(figsize=(10, 8))
            plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            plt.imshow(heatmap, cmap='jet', alpha=0.5)
            plt.axis('off')
            plt.tight_layout()
            
            # Save heatmap
            heatmap_filename = f"heatmap_{os.path.basename(image_path)}"
            heatmap_path = os.path.join(app.config['UPLOAD_FOLDER'], heatmap_filename)
            plt.savefig(heatmap_path, bbox_inches='tight', pad_inches=0)
            plt.close()
            
            result_paths['heatmap'] = os.path.join('uploads', heatmap_filename).replace('\\', '/')
        
        if 'overlay' in localization_methods:
            # Create overlay image with enhanced visualization
            overlay = image.copy()
            heatmap_colored = cv2.applyColorMap((heatmap * 255).astype(np.uint8), cv2.COLORMAP_JET)
            overlay = cv2.addWeighted(overlay, 0.7, heatmap_colored, 0.3, 0)
            
            # Save overlay
            overlay_filename = f"overlay_{os.path.basename(image_path)}"
            overlay_path = os.path.join(app.config['UPLOAD_FOLDER'], overlay_filename)
            cv2.imwrite(overlay_path, overlay)
            
            result_paths['overlay'] = os.path.join('uploads', overlay_filename).replace('\\', '/')
        
        if 'contour' in localization_methods:
            # Create contour image
            contour_img = image.copy()
            cv2.drawContours(contour_img, contours, -1, (0, 255, 0), 2)
            
            # Add text labels for tampered regions
            for i, cnt in enumerate(contours):
                # Get bounding box of contour
                x, y, w, h = cv2.boundingRect(cnt)
                # Calculate centroid
                M = cv2.moments(cnt)
                if M["m00"] != 0:
                    cx = int(M["m10"] / M["m00"])
                    cy = int(M["m01"] / M["m00"])
                else:
                    cx, cy = x + w//2, y + h//2
                    
                # Add label
                cv2.putText(contour_img, f"Tampered Region {i+1}", (cx-60, cy-15), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                cv2.putText(contour_img, f"Tampered Region {i+1}", (cx-60, cy-15), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 1)
            
            # Save contour image
            contour_filename = f"contour_{os.path.basename(image_path)}"
            contour_path = os.path.join(app.config['UPLOAD_FOLDER'], contour_filename)
            cv2.imwrite(contour_path, contour_img)
            
            result_paths['contour'] = os.path.join('uploads', contour_filename).replace('\\', '/')
        
        # Additional visualization methods
        if 'mask' in localization_methods:
            # Create a color mask for tampered regions
            mask_img = image.copy()
            colored_mask = np.zeros_like(mask_img)
            
            # Fill contours with semi-transparent color
            cv2.drawContours(colored_mask, contours, -1, (0, 0, 255), -1)  # Fill with red
            mask_img = cv2.addWeighted(mask_img, 1.0, colored_mask, 0.5, 0)
            
            # Save mask image
            mask_filename = f"mask_{os.path.basename(image_path)}"
            mask_path = os.path.join(app.config['UPLOAD_FOLDER'], mask_filename)
            cv2.imwrite(mask_path, mask_img)
            
            result_paths['mask'] = os.path.join('uploads', mask_filename).replace('\\', '/')
        
        if 'edge' in localization_methods:
            # Create edge detection image to highlight manipulation boundaries
            edge_img = image.copy()
            
            # Create a mask from contours
            edge_mask = np.zeros((height, width), dtype=np.uint8)
            cv2.drawContours(edge_mask, contours, -1, 255, -1)
            
            # Get edges using Canny edge detector
            edges = cv2.Canny(edge_mask, 100, 200)
            
            # Dilate edges for better visibility
            kernel = np.ones((3, 3), np.uint8)
            edges = cv2.dilate(edges, kernel, iterations=1)
            
            # Overlay edges on original image with distinctive color
            edge_img[edges > 0] = [0, 255, 255]  # Yellow edges
            
            # Save edge image
            edge_filename = f"edge_{os.path.basename(image_path)}"
            edge_path = os.path.join(app.config['UPLOAD_FOLDER'], edge_filename)
            cv2.imwrite(edge_path, edge_img)
            
            result_paths['edge'] = os.path.join('uploads', edge_filename).replace('\\', '/')
        
        if 'highlight' in localization_methods:
            # Create a highlighted image with bounding boxes
            highlight_img = image.copy()
            
            for cnt in contours:
                x, y, w, h = cv2.boundingRect(cnt)
                # Draw rectangle
                cv2.rectangle(highlight_img, (x, y), (x+w, y+h), (0, 165, 255), 2)
                
                # Add info text
                area_percent = (cv2.contourArea(cnt) / (width * height)) * 100
                cv2.putText(highlight_img, f"Area: {area_percent:.1f}%", (x, y-10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 165, 255), 1)
            
            # Save highlight image
            highlight_filename = f"highlight_{os.path.basename(image_path)}"
            highlight_path = os.path.join(app.config['UPLOAD_FOLDER'], highlight_filename)
            cv2.imwrite(highlight_path, highlight_img)
            
            result_paths['highlight'] = os.path.join('uploads', highlight_filename).replace('\\', '/')
            
        # Store the number of tampered regions detected
        result_paths['tampered_regions_count'] = len(contours)
        print(f"Tampering localization completed. Found {len(contours)} tampered regions.")
        
        return result_paths
    except Exception as e:
        print(f"Error in localization: {e}")
        traceback.print_exc()
        return {}

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/api/analyze', methods=['POST'])
def api_analyze():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    
    file = request.files['file']
    show_localization = request.form.get('show_localization', 'false') == 'true'
    show_ela = request.form.get('show_ela', 'false') == 'true'
    
    # Get selected localization methods or use defaults
    localization_methods = request.form.getlist('localization_methods[]')
    if not localization_methods and show_localization:
        localization_methods = ['heatmap', 'overlay', 'contour', 'mask', 'edge', 'highlight']
    
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    
    if file and allowed_file(file.filename):
        try:
            # Generate a unique filename with timestamp to prevent caching issues
            timestamp = int(time.time())
            filename = f"{timestamp}_{str(uuid.uuid4())}_{secure_filename(file.filename)}"
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
            print(f"Raw prediction value: {prediction}")
            
            # Try to get probability if available, otherwise use a default confidence
            try:
                probability = svm_model.predict_proba(feature_vector)[0]
                confidence = float(probability[1] if prediction == 1 else probability[0])
                print(f"Probabilities: {probability}, Using confidence: {confidence}")
            except:
                print("predict_proba not available, using decision_function instead")
                try:
                    # Try to use decision function as an alternative
                    decision_scores = svm_model.decision_function(feature_vector)[0]
                    # Convert decision score to a confidence-like value between 0 and 1
                    confidence = 1.0 / (1.0 + np.exp(-np.abs(decision_scores)))
                    print(f"Decision score: {decision_scores}, Converted confidence: {confidence}")
                except:
                    print("decision_function not available, using default confidence")
                    confidence = 0.95  # Default high confidence
            
            # In SVM, class 1 typically means the positive class (tampered)
            # But let's double check by examining the decision boundary
            is_tampered = bool(prediction == 1)
            
            print(f"Prediction: {prediction}, Is tampered: {is_tampered}, Confidence: {confidence}")
            
            # Format the result
            result = {
                'is_tampered': is_tampered,
                'confidence': float(confidence),
                'message': 'Image analysis detected signs of manipulation' if is_tampered else 'No signs of manipulation detected',
                'method': 'CNN-SVM Hybrid Analysis',
                'timestamp': time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                'input_image_path': f"/static/uploads/{filename}"
            }
            
            # If the image is predicted as tampered and localization is requested, generate visualizations
            if is_tampered and show_localization:
                print(f"Generating tampering localization with methods: {localization_methods}")
                localization_results = localize_tampering(filepath, cnn_model, localization_methods=localization_methods)
                
                if localization_results:
                    # Add all generated visualization paths to the result
                    for key, path in localization_results.items():
                        if key != 'tampered_regions_count':
                            # Ensure path has the correct format with forward slashes
                            path = path.replace('\\', '/')
                            formatted_path = f"/static/{path}" if not path.startswith('/static/') else path
                            result[f"{key}_path"] = formatted_path
                            print(f"Added visualization path: {key}_path = {formatted_path}")
            
            # If ELA is requested, generate ELA visualization
            if show_ela:
                print("Generating Error Level Analysis...")
                ela_path = perform_ela(filepath)
                if ela_path:
                    # Ensure path has forward slashes
                    ela_path = ela_path.replace('\\', '/')
                    formatted_ela_path = f"/static/{ela_path}" if not ela_path.startswith('/static/') else ela_path
                    result['ela_path'] = formatted_ela_path
                    
                    # Verify the ELA path exists
                    ela_full_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'static', ela_path)
                    if os.path.exists(ela_full_path):
                        print(f"ELA file verified to exist at: {ela_full_path}")
                        print(f"ELA file size: {os.path.getsize(ela_full_path)} bytes")
                    else:
                        print(f"WARNING: ELA file does not exist at: {ela_full_path}")
                    
                    print(f"Added ELA path: {formatted_ela_path}")
                else:
                    print("WARNING: Failed to generate ELA image, ela_path is None")
            
            return jsonify(result)
        except Exception as e:
            print(f"Error in analysis: {e}")
            traceback.print_exc()
            return jsonify({'error': str(e)}), 500
    
    return jsonify({'error': 'Invalid file type'}), 400

@app.route('/api/convert-tiff', methods=['POST'])
def convert_tiff():
    """
    Convert a TIFF file to JPEG for browser preview
    """
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    
    file = request.files['file']
    
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    
    if file and (file.filename.lower().endswith('.tif') or file.filename.lower().endswith('.tiff')):
        try:
            # Generate a unique filename
            timestamp = int(time.time())
            filename = f"{timestamp}_{str(uuid.uuid4())}_preview.jpg"
            output_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            
            # Use PIL to convert TIFF to JPEG
            from PIL import Image
            
            # Open the TIFF file
            image = Image.open(file.stream)
            
            # Convert to RGB if needed
            if image.mode != 'RGB':
                image = image.convert('RGB')
            
            # Save as JPEG
            image.save(output_path, 'JPEG', quality=85)
            
            return jsonify({
                'preview_url': f"/static/uploads/{filename}"
            })
        except Exception as e:
            print(f"Error converting TIFF: {e}")
            traceback.print_exc()
            return jsonify({'error': str(e)}), 500
    
    return jsonify({'error': 'Invalid file type'}), 400

@app.route('/api/ela', methods=['POST'])
def error_level_analysis():
    """
    Perform Error Level Analysis (ELA) on an image to detect manipulation
    """
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    
    file = request.files['file']
    
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    
    if file and allowed_file(file.filename):
        try:
            # Generate a unique filename
            timestamp = int(time.time())
            filename = f"{timestamp}_{str(uuid.uuid4())}_{secure_filename(file.filename)}"
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            
            # Perform ELA
            ela_image_path = perform_ela(filepath)
            
            if ela_image_path:
                return jsonify({
                    'ela_image_path': f"/static/{ela_image_path}",
                    'original_path': f"/static/uploads/{filename}"
                })
            else:
                return jsonify({'error': 'Failed to generate ELA image'}), 500
                
        except Exception as e:
            print(f"Error in ELA analysis: {e}")
            traceback.print_exc()
            return jsonify({'error': str(e)}), 500
    
    return jsonify({'error': 'Invalid file type'}), 400

def perform_ela(image_path, quality=90):
    """
    Performs Error Level Analysis on an image
    
    Args:
        image_path: Path to the image
        quality: JPEG quality level for recompression
        
    Returns:
        Path to the ELA image
    """
    try:
        print(f"Starting ELA analysis for: {image_path}")
        # Check file extension
        file_ext = os.path.splitext(image_path)[1].lower()
        
        # Use PIL for image processing
        from PIL import Image, ImageChops, ImageEnhance
        
        # Open the original image
        if file_ext in ['.tif', '.tiff']:
            original = Image.open(image_path)
            if original.mode != 'RGB':
                original = original.convert('RGB')
        else:
            original = Image.open(image_path)
            if original.mode != 'RGB':
                original = original.convert('RGB')
        
        print(f"Original image size: {original.size}, mode: {original.mode}")
        
        # Save the image with specified quality
        temp_filename = f"temp_{os.path.basename(image_path)}"
        temp_path = os.path.join(app.config['UPLOAD_FOLDER'], temp_filename)
        original.save(temp_path, 'JPEG', quality=quality)
        print(f"Saved compressed image to: {temp_path}")
        
        # Open the saved image
        resaved = Image.open(temp_path)
        print(f"Resaved image size: {resaved.size}, mode: {resaved.mode}")
        
        # Calculate the difference
        diff = ImageChops.difference(original, resaved)
        
        # Amplify the difference for better visualization
        # Scale factor determines how much to amplify the differences
        scale_factor = 25  # Increased from 20 for better visibility
        diff = ImageChops.multiply(diff, Image.new('RGB', diff.size, (scale_factor, scale_factor, scale_factor)))
        
        # Enhance contrast
        enhancer = ImageEnhance.Contrast(diff)
        diff = enhancer.enhance(2.5)  # Increased contrast for better visibility
        
        # Apply auto-level to enhance visibility
        from PIL import ImageOps
        diff = ImageOps.autocontrast(diff, cutoff=0.1)
        
        # Add a color overlay to make differences more visible
        # Create a more vibrant output by using a colormap-like effect
        r, g, b = diff.split()
        diff = Image.merge("RGB", (
            ImageOps.equalize(r), 
            ImageOps.equalize(g), 
            ImageOps.equalize(b)
        ))
        
        # Further enhance brightness
        brightness_enhancer = ImageEnhance.Brightness(diff)
        diff = brightness_enhancer.enhance(1.3)
        
        # Save the ELA image
        ela_filename = f"ela_{os.path.basename(image_path)}"
        ela_path = os.path.join(app.config['UPLOAD_FOLDER'], ela_filename)
        diff.save(ela_path)
        print(f"Saved ELA image to: {ela_path}")
        
        # Verify the ELA image exists
        if not os.path.exists(ela_path):
            print(f"ERROR: ELA image was not saved! Path doesn't exist: {ela_path}")
            return None
            
        # Get file size to verify it's not empty
        file_size = os.path.getsize(ela_path)
        print(f"ELA image size: {file_size} bytes")
        if file_size < 1000:
            print(f"WARNING: ELA image is very small ({file_size} bytes), might be corrupted")
            
        # Clean up the temporary file
        try:
            os.remove(temp_path)
        except Exception as e:
            print(f"Warning: Failed to remove temp file: {e}")
        
        # Return the relative path for the API response - ensure it uses forward slashes
        result_path = os.path.join('uploads', ela_filename).replace('\\', '/')
        print(f"Returning ELA path: {result_path}")
        return result_path
    except Exception as e:
        print(f"Error in ELA: {e}")
        traceback.print_exc()
        return None

@app.route('/api/frequency', methods=['POST'])
def frequency_analysis():
    """
    Perform frequency domain analysis on an image to detect manipulation
    """
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    
    file = request.files['file']
    
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    
    if file and allowed_file(file.filename):
        try:
            # Generate a unique filename
            timestamp = int(time.time())
            filename = f"{timestamp}_{str(uuid.uuid4())}_{secure_filename(file.filename)}"
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            
            # Perform frequency analysis
            dft_image_path = perform_dft_analysis(filepath)
            
            if dft_image_path:
                return jsonify({
                    'dft_image_path': f"/static/{dft_image_path}",
                    'original_path': f"/static/uploads/{filename}"
                })
            else:
                return jsonify({'error': 'Failed to generate frequency analysis image'}), 500
                
        except Exception as e:
            print(f"Error in frequency analysis: {e}")
            traceback.print_exc()
            return jsonify({'error': str(e)}), 500
    
    return jsonify({'error': 'Invalid file type'}), 400

def perform_dft_analysis(image_path):
    """
    Performs frequency domain analysis on an image using Discrete Fourier Transform
    
    Args:
        image_path: Path to the image
        
    Returns:
        Path to the DFT visualization image
    """
    try:
        # Check file extension
        file_ext = os.path.splitext(image_path)[1].lower()
        
        # Load the image
        if file_ext in ['.tif', '.tiff']:
            from PIL import Image
            import numpy as np
            
            pil_image = Image.open(image_path)
            if pil_image.mode != 'RGB':
                pil_image = pil_image.convert('RGB')
            image = np.array(pil_image)
            # Convert from RGB to BGR for OpenCV compatibility
            image = image[:, :, ::-1].copy()
        else:
            image = cv2.imread(image_path)
        
        if image is None:
            print(f"Failed to load image for DFT analysis: {image_path}")
            return None
        
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Apply DFT
        dft = cv2.dft(np.float32(gray), flags=cv2.DFT_COMPLEX_OUTPUT)
        dft_shift = np.fft.fftshift(dft)
        
        # Calculate magnitude spectrum
        magnitude_spectrum = 20 * np.log(cv2.magnitude(dft_shift[:, :, 0], dft_shift[:, :, 1]) + 1)
        
        # Normalize for visualization
        magnitude_spectrum = cv2.normalize(magnitude_spectrum, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        
        # Apply colormap for better visualization
        magnitude_colored = cv2.applyColorMap(magnitude_spectrum, cv2.COLORMAP_JET)
        
        # Save the DFT image
        dft_filename = f"dft_{os.path.basename(image_path)}"
        dft_path = os.path.join(app.config['UPLOAD_FOLDER'], dft_filename)
        cv2.imwrite(dft_path, magnitude_colored)
        
        return os.path.join('uploads', dft_filename)
    except Exception as e:
        print(f"Error in DFT analysis: {e}")
        traceback.print_exc()
        return None

@app.route('/api/check-image/<path:img_path>', methods=['GET'])
def check_image(img_path):
    """
    Debug endpoint to check if an image exists and is accessible
    """
    try:
        # Normalize the path to check in the static directory
        if img_path.startswith('/static/'):
            img_path = img_path[8:]  # Remove /static/ prefix
        
        # Get the absolute path
        full_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'static', img_path)
        
        if os.path.exists(full_path):
            # Check if it's a valid image file
            from PIL import Image
            try:
                img = Image.open(full_path)
                img_info = {
                    'path': img_path,
                    'full_path': full_path,
                    'exists': True,
                    'size': os.path.getsize(full_path),
                    'dimensions': f"{img.width}x{img.height}",
                    'format': img.format,
                    'mode': img.mode,
                    'url': f"/static/{img_path}"
                }
                return jsonify(img_info)
            except Exception as e:
                return jsonify({
                    'path': img_path,
                    'full_path': full_path,
                    'exists': True,
                    'error': f"Not a valid image: {str(e)}",
                    'size': os.path.getsize(full_path)
                })
        else:
            return jsonify({
                'path': img_path,
                'full_path': full_path,
                'exists': False,
                'error': 'File not found'
            }), 404
    except Exception as e:
        return jsonify({
            'path': img_path,
            'error': str(e)
        }), 500

@app.route('/api/debug-image-path', methods=['GET'])
def debug_image_path():
    """
    Debug endpoint to check image paths and file existence
    """
    path = request.args.get('path', '')
    
    if not path:
        return jsonify({'error': 'No path provided'}), 400
    
    # If path starts with /static/, remove it
    if path.startswith('/static/'):
        path = path[8:]  # Remove /static/ prefix
    
    # Replace any backslashes with forward slashes
    path = path.replace('\\', '/')
    
    # Get the absolute path
    full_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'static', path)
    
    result = {
        'requested_path': path,
        'full_path': full_path,
        'exists': os.path.exists(full_path),
        'is_file': os.path.isfile(full_path) if os.path.exists(full_path) else False,
        'size': os.path.getsize(full_path) if os.path.exists(full_path) and os.path.isfile(full_path) else 0,
        'static_url': f'/static/{path}'
    }
    
    return jsonify(result)

@app.route('/api/list-directory', methods=['GET'])
def list_directory():
    """
    List contents of a directory within the static folder
    """
    path = request.args.get('path', '')
    
    # For security, only allow listing directories within the static folder
    base_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'static')
    
    # Join with the requested path and normalize
    full_path = os.path.normpath(os.path.join(base_path, path))
    
    # Security check - make sure we're still within the static folder
    if not full_path.startswith(base_path):
        return jsonify({'error': 'Invalid path - must be within the static directory'}), 403
    
    if not os.path.exists(full_path):
        return jsonify({'error': f'Path does not exist: {path}'}), 404
    
    if not os.path.isdir(full_path):
        return jsonify({'error': f'Path is not a directory: {path}'}), 400
        
    try:
        files = []
        directories = []
        
        # List contents
        for item in os.listdir(full_path):
            item_path = os.path.join(full_path, item)
            if os.path.isfile(item_path):
                # For files, include size and extension
                files.append({
                    'name': item,
                    'size': os.path.getsize(item_path),
                    'extension': os.path.splitext(item)[1],
                    'path': os.path.join(path, item).replace('\\', '/')
                })
            elif os.path.isdir(item_path):
                # For directories
                directories.append({
                    'name': item,
                    'path': os.path.join(path, item).replace('\\', '/')
                })
                
        # Sort results
        files.sort(key=lambda x: x['name'])
        directories.sort(key=lambda x: x['name'])
        
        return jsonify({
            'path': path,
            'full_path': full_path,
            'files': files,
            'directories': directories
        })
    except Exception as e:
        print(f"Error listing directory: {e}")
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000) 