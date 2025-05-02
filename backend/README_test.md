# Image Tampering Detection Test Script

This script allows you to test multiple images at once using the image tampering detection models. It processes all images in a specified directory and generates a comprehensive report with predictions and visualizations.

## Features

- Process all images in a directory at once
- Generate visual reports for each image with prediction results
- Create Error Level Analysis (ELA) for each image
- Generate a summary HTML report with links to all individual image reports
- Support for both single model and ensemble model predictions

## Requirements

- All dependencies from the main application
- Additional dependencies: 
  - PIL/Pillow (for image processing)
  - argparse (for command line arguments)

## Usage

```bash
python test.py --input_dir /path/to/images [--use_ensemble] [--skip_ela]
```

### Arguments

- `--input_dir` or `-i`: Directory containing images to test (required)
- `--use_ensemble` or `-e`: Use ensemble model for prediction (optional)
- `--skip_ela` or `-s`: Skip ELA analysis to save time (optional)

## Output

The script will create a timestamped report directory under `backend/reports/` with the following contents:

- `summary.html`: Main summary report with statistics and links to individual reports
- `summary.json`: JSON format of all results for programmatic use
- Individual report images for each processed image with prediction information
- ELA analysis images for each processed image (unless skipped)
- `original_images/`: Copy of all original images for reference

## Example

```bash
# Process images using a single model
python test.py --input_dir ./test_images

# Process images using the ensemble model
python test.py --input_dir ./test_images --use_ensemble

# Process images without ELA analysis (faster)
python test.py --input_dir ./test_images --skip_ela
```

## Understanding the Report

The summary report includes:
- Total number of images processed
- Number of tampered and authentic images detected
- Table of results with links to detailed reports

Each individual image report displays:
- The original image
- Prediction (TAMPERED or AUTHENTIC)
- Confidence score
- Detection method used
- For ensemble predictions, consensus level and voting details
- ELA analysis showing potential manipulation areas (bright spots) 