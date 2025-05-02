# Image Tampering Detection Backend

This is the Flask backend for the Image Tampering Detection application, which uses deep learning models to analyze and detect tampered images.

## Setup Instructions

1. Create a virtual environment:
```bash
python -m venv .venv
```

2. Activate the virtual environment:
- Windows:
```bash
.venv\Scripts\activate
```
- macOS/Linux:
```bash
source .venv/bin/activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Run the application:
```bash
python run.py
```

The API server will start on http://localhost:5000.

## API Endpoints

### 1. Analyze Image
- **URL**: `/api/analyze`
- **Method**: `POST`
- **Content-Type**: `multipart/form-data`
- **Parameters**:
  - `file`: The image file to analyze
  - `show_localization`: (Optional) Boolean to enable localization visualization
  - `localization_methods[]`: (Optional) Array of localization methods to use
  - `show_ela`: (Optional) Boolean to enable Error Level Analysis (ELA)
- **Response**: JSON with analysis results and image paths

### 2. Ensemble Analysis
- **URL**: `/api/analyze/ensemble`
- **Method**: `POST`
- **Content-Type**: `multipart/form-data`
- **Parameters**:
  - `file`: The image file to analyze
  - `show_localization`: (Optional) Boolean to enable localization visualization
  - `localization_methods[]`: (Optional) Array of localization methods to use
  - `show_ela`: (Optional) Boolean to enable Error Level Analysis (ELA)
- **Response**: JSON with ensemble analysis results and image paths

### 3. Error Level Analysis (ELA)
- **URL**: `/api/analyze/ela`
- **Method**: `POST`
- **Content-Type**: `multipart/form-data`
- **Parameters**:
  - `file`: The image file to analyze
  - `quality`: (Optional) JPEG quality for ELA analysis (1-100, default: 90)
  - `scale`: (Optional) Scale factor for ELA visualization (1-50, default: 15)
- **Response**: JSON with ELA analysis results and image paths

### 4. Convert TIFF
- **URL**: `/api/convert-tiff`
- **Method**: `POST`
- **Content-Type**: `multipart/form-data`
- **Parameters**:
  - `file`: The TIFF file to convert
- **Response**: JSON with the path to the converted image

### 5. View TIFF
- **URL**: `/api/view-tiff/<tiff_path>`
- **Method**: `GET`
- **Response**: JSON with the path to the viewable image

## Localization Methods

The API supports several methods for localizing potential tampering:

1. **heatmap**: Generates a heatmap highlighting areas with high probability of tampering
2. **ela**: Error Level Analysis to detect compression inconsistencies
3. **overlay**: Overlays the heatmap on the original image
4. **contour**: Draws contours around potential tampered regions
5. **mask**: Creates a binary mask showing potential tampered areas
6. **edge**: Detects edges in the ELA result to highlight boundaries of tampering
7. **highlight**: Highlights anomalies detected by ELA analysis

## Error Level Analysis (ELA)

ELA is a forensic method used to identify potential areas where an image has been modified. It works by:

1. Saving the image at a specified quality level (e.g., 90%)
2. Comparing the re-saved image with the original
3. Highlighting the differences between the two images
4. Scaling those differences to enhance visibility

Areas with high error levels (differences) often indicate manipulation, as they show inconsistencies in the JPEG compression patterns.

## Model Information

The application uses pre-trained models located in:
- CNN models: `/data/output/pre_trained_cnn/`
- SVM models: `/data/output/pre_trained_svm/`

## Directory Structure

- `app.py`: Main Flask application
- `run.py`: Script to run the Flask application
- `models/`: Model definitions
- `data/output/`: Pre-trained model weights
- `uploads/`: Directory for uploaded images
- `outputs/`: Directory for processed images and visualizations
