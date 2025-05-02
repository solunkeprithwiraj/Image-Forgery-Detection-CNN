import os
import sys
import json
import argparse
import shutil
from datetime import datetime
import cv2
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont
from app import preprocess_image, load_cnn_model, load_ensemble_models, perform_ela_analysis

def create_report_directory():
    """Create a directory for the report with timestamp"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), f'reports/report_{timestamp}')
    os.makedirs(report_dir, exist_ok=True)
    return report_dir

def process_image(img_path, cnn_model, ensemble_models, device, use_ensemble=False, generate_ela=True):
    """Process a single image and return the results"""
    
    # Check if file exists
    if not os.path.isfile(img_path):
        return {
            'filename': os.path.basename(img_path),
            'status': 'error',
            'message': 'File not found'
        }
        
    try:
        # Process the image
        img = preprocess_image(img_path)
        if img is None:
            return {
                'filename': os.path.basename(img_path),
                'status': 'error',
                'message': 'Failed to process image'
            }
        
        # Use ensemble model if requested and available
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
            
            result = {
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
            }
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
            
            result = {
                'filename': os.path.basename(img_path),
                'status': 'success',
                'is_tampered': bool(pred_class),
                'confidence': confidence,
                'method': 'CNN'
            }
        
        # Generate ELA if requested
        if generate_ela:
            ela_image = perform_ela_analysis(img_path)
            if ela_image is not None:
                result['ela_generated'] = True
            else:
                result['ela_generated'] = False
        
        return result
    
    except Exception as e:
        return {
            'filename': os.path.basename(img_path),
            'status': 'error',
            'message': str(e)
        }

def create_visual_report(image_path, result, report_dir):
    """Create a visual report for an image with prediction information"""
    try:
        # Open original image
        original = Image.open(image_path)
        
        # Resize if too large for display
        max_width = 800
        max_height = 600
        if original.width > max_width or original.height > max_height:
            original.thumbnail((max_width, max_height), Image.LANCZOS)
        
        # Create a new image with space for text
        margin = 30
        text_height = 200
        report_img = Image.new('RGB', (original.width, original.height + text_height), (255, 255, 255))
        report_img.paste(original, (0, 0))
        
        # Add text information
        draw = ImageDraw.Draw(report_img)
        
        # Try to use a nice font if available
        try:
            # Try common fonts that might be available
            font_paths = [
                "arial.ttf",  # Windows
                "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",  # Linux
                "/System/Library/Fonts/Helvetica.ttc"  # macOS
            ]
            
            font = None
            for font_path in font_paths:
                try:
                    font = ImageFont.truetype(font_path, 16)
                    break
                except:
                    continue
                
            if font is None:
                font = ImageFont.load_default()
        except:
            font = ImageFont.load_default()
        
        # Draw text
        y_pos = original.height + margin
        filename = os.path.basename(image_path)
        
        if result['status'] == 'success':
            # Determine text color based on prediction
            pred_color = (255, 0, 0) if result['is_tampered'] else (0, 128, 0)
            
            draw.text((margin, y_pos), f"Filename: {filename}", fill=(0, 0, 0), font=font)
            y_pos += 25
            
            draw.text((margin, y_pos), f"Prediction: {('TAMPERED' if result['is_tampered'] else 'AUTHENTIC')}", 
                    fill=pred_color, font=font)
            y_pos += 25
            
            draw.text((margin, y_pos), f"Confidence: {result['confidence']:.4f}", fill=(0, 0, 0), font=font)
            y_pos += 25
            
            draw.text((margin, y_pos), f"Method: {result['method']}", fill=(0, 0, 0), font=font)
            y_pos += 25
            
            # Add ensemble details if available
            if 'ensemble_detail' in result:
                details = result['ensemble_detail']
                draw.text((margin, y_pos), 
                        f"Consensus: {details['consensus_level']} ({details['tampered_votes']}/{details['ensemble_size']} tampered votes)", 
                        fill=(0, 0, 0), font=font)
        else:
            draw.text((margin, y_pos), f"Filename: {filename}", fill=(0, 0, 0), font=font)
            y_pos += 25
            draw.text((margin, y_pos), f"Error: {result['message']}", fill=(255, 0, 0), font=font)
        
        # Save the report image
        report_filename = f"report_{os.path.basename(image_path)}"
        report_path = os.path.join(report_dir, report_filename)
        report_img.save(report_path)
        
        # Generate ELA report if available
        if result.get('status') == 'success' and result.get('ela_generated', False):
            # Get ELA image
            ela_image = perform_ela_analysis(image_path)
            if ela_image is not None:
                # Resize ELA image to match original
                ela_pil = Image.fromarray(ela_image)
                ela_pil = ela_pil.resize((original.width, original.height), Image.LANCZOS)
                
                # Create ELA report
                ela_report = Image.new('RGB', (original.width, original.height + text_height), (255, 255, 255))
                ela_report.paste(ela_pil, (0, 0))
                
                # Add text to ELA report
                draw_ela = ImageDraw.Draw(ela_report)
                y_pos = original.height + margin
                
                draw_ela.text((margin, y_pos), f"ELA Analysis: {filename}", fill=(0, 0, 0), font=font)
                y_pos += 25
                
                draw_ela.text((margin, y_pos), 
                            "Brighter areas indicate potential manipulation", 
                            fill=(0, 0, 0), font=font)
                
                # Save ELA report
                ela_report_filename = f"ela_report_{os.path.basename(image_path)}"
                ela_report_path = os.path.join(report_dir, ela_report_filename)
                ela_report.save(ela_report_path)
        
        return report_path
    
    except Exception as e:
        print(f"Error creating visual report for {image_path}: {str(e)}")
        return None

def generate_summary_report(results, report_dir):
    """Generate a summary report of all processed images"""
    try:
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        total_images = len(results)
        processed_images = sum(1 for r in results if r.get('status') == 'success')
        tampered_count = sum(1 for r in results if r.get('status') == 'success' and r.get('is_tampered'))
        authentic_count = sum(1 for r in results if r.get('status') == 'success' and not r.get('is_tampered'))
        error_count = sum(1 for r in results if r.get('status') == 'error')
        
        summary_path = os.path.join(report_dir, 'summary.html')
        
        with open(summary_path, 'w') as f:
            f.write(f"""<!DOCTYPE html>
<html>
<head>
    <title>Image Tampering Detection Report</title>
    <style>
        body {{ font-family: Arial, sans-serif; line-height: 1.6; margin: 20px; }}
        h1 {{ color: #333; }}
        .summary {{ background-color: #f5f5f5; padding: 15px; border-radius: 5px; margin-bottom: 20px; }}
        table {{ border-collapse: collapse; width: 100%; }}
        th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
        th {{ background-color: #f2f2f2; }}
        tr:nth-child(even) {{ background-color: #f9f9f9; }}
        .tampered {{ color: red; font-weight: bold; }}
        .authentic {{ color: green; font-weight: bold; }}
        .error {{ color: orange; }}
    </style>
</head>
<body>
    <h1>Image Tampering Detection Report</h1>
    
    <div class="summary">
        <h2>Summary</h2>
        <p>Timestamp: {timestamp}</p>
        <p>Total Images: {total_images}</p>
        <p>Successfully Processed: {processed_images}</p>
        <p>Tampered Images: {tampered_count}</p>
        <p>Authentic Images: {authentic_count}</p>
        <p>Failed to Process: {error_count}</p>
    </div>
    
    <h2>Detailed Results</h2>
    <table>
        <tr>
            <th>Filename</th>
            <th>Status</th>
            <th>Prediction</th>
            <th>Confidence</th>
            <th>Method</th>
            <th>Reports</th>
        </tr>
""")
            
            for result in results:
                filename = result['filename']
                status = result['status']
                
                if status == 'success':
                    prediction = "TAMPERED" if result['is_tampered'] else "AUTHENTIC"
                    prediction_class = "tampered" if result['is_tampered'] else "authentic"
                    confidence = f"{result['confidence']:.4f}"
                    method = result['method']
                    
                    # Links to reports
                    report_link = f"<a href='report_{filename}'>View Report</a>"
                    if result.get('ela_generated', False):
                        ela_link = f"<a href='ela_report_{filename}'>View ELA</a>"
                        reports = f"{report_link} | {ela_link}"
                    else:
                        reports = report_link
                    
                    f.write(f"""
        <tr>
            <td>{filename}</td>
            <td>{status}</td>
            <td class="{prediction_class}">{prediction}</td>
            <td>{confidence}</td>
            <td>{method}</td>
            <td>{reports}</td>
        </tr>""")
                else:
                    # Error case
                    f.write(f"""
        <tr>
            <td>{filename}</td>
            <td>{status}</td>
            <td class="error" colspan="3">{result.get('message', 'Unknown error')}</td>
            <td>No reports available</td>
        </tr>""")
            
            f.write("""
    </table>
</body>
</html>""")
        
        # Also generate a JSON report
        json_path = os.path.join(report_dir, 'summary.json')
        with open(json_path, 'w') as f:
            json.dump({
                'total_images': total_images,
                'processed_images': processed_images,
                'tampered_count': tampered_count,
                'authentic_count': authentic_count,
                'error_count': error_count,
                'timestamp': timestamp,
                'results': results
            }, f, indent=4)
        
        return summary_path
    
    except Exception as e:
        print(f"Error generating summary report: {str(e)}")
        return None

def copy_images_to_report(input_dir, report_dir):
    """Copy all images from input directory to report directory for reference"""
    try:
        images_dir = os.path.join(report_dir, 'original_images')
        os.makedirs(images_dir, exist_ok=True)
        
        for filename in os.listdir(input_dir):
            if any(filename.lower().endswith(ext) for ext in ['.jpg', '.jpeg', '.png', '.tif', '.tiff']):
                src_path = os.path.join(input_dir, filename)
                dst_path = os.path.join(images_dir, filename)
                shutil.copy2(src_path, dst_path)
        
        return images_dir
    except Exception as e:
        print(f"Error copying images to report directory: {str(e)}")
        return None

def main():
    parser = argparse.ArgumentParser(description='Test image tampering detection on a directory of images')
    parser.add_argument('--input_dir', '-i', type=str, required=True, help='Directory containing images to test')
    parser.add_argument('--use_ensemble', '-e', action='store_true', help='Use ensemble model for prediction')
    parser.add_argument('--skip_ela', '-s', action='store_true', help='Skip ELA analysis to save time')
    
    args = parser.parse_args()
    
    # Check if input directory exists
    if not os.path.isdir(args.input_dir):
        print(f"Error: Input directory '{args.input_dir}' not found")
        return 1
    
    # Load models
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    print("Loading models...")
    cnn_model = load_cnn_model()
    ensemble_models = None
    if args.use_ensemble:
        print("Loading ensemble models...")
        ensemble_models = load_ensemble_models()
    
    # Create report directory
    report_dir = create_report_directory()
    print(f"Report will be saved to: {report_dir}")
    
    # Copy original images to report directory
    print("Copying original images to report directory...")
    copy_images_to_report(args.input_dir, report_dir)
    
    # Get all image files in the directory
    allowed_extensions = {'png', 'jpg', 'jpeg', 'tif', 'tiff'}
    image_files = []
    for filename in os.listdir(args.input_dir):
        if any(filename.lower().endswith(f'.{ext}') for ext in allowed_extensions):
            image_files.append(os.path.join(args.input_dir, filename))
    
    if not image_files:
        print("No valid images found in directory")
        return 1
    
    print(f"Found {len(image_files)} images to process")
    
    # Process images
    results = []
    for i, img_path in enumerate(image_files):
        print(f"Processing image {i+1}/{len(image_files)}: {os.path.basename(img_path)}")
        result = process_image(
            img_path, 
            cnn_model, 
            ensemble_models, 
            device, 
            use_ensemble=args.use_ensemble,
            generate_ela=not args.skip_ela
        )
        results.append(result)
        
        # Create visual report for this image
        if result['status'] == 'success':
            print(f"  - Result: {'TAMPERED' if result['is_tampered'] else 'AUTHENTIC'} (confidence: {result['confidence']:.4f})")
        else:
            print(f"  - Error: {result['message']}")
        
        print("  - Creating visual report...")
        create_visual_report(img_path, result, report_dir)
    
    # Generate summary report
    print("Generating summary report...")
    summary_path = generate_summary_report(results, report_dir)
    
    print("\nTesting completed!")
    print(f"Report saved to: {report_dir}")
    print(f"Summary report: {summary_path}")
    
    return 0

if __name__ == '__main__':
    sys.exit(main()) 