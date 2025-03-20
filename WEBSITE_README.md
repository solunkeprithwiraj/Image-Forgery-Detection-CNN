# Image Forgery Detection Website

This website provides a user-friendly interface for detecting image forgery using a CNN-based model with advanced localization capabilities.

## Features

- **Image Forgery Detection**: Upload an image to determine if it has been tampered with
- **Tampering Localization**: Visualize exactly where an image has been tampered with
- **Multiple Visualization Options**: View heatmaps, overlays, and contour detection of tampered regions
- **Detailed Analysis**: Get information about the number of tampered regions and confidence scores

## How to Run the Website

1. Make sure all dependencies are installed:

   ```
   .\install_dependencies.bat
   ```

2. Start the website:

   ```
   .\restart_website.bat
   ```

3. Open your web browser and navigate to:
   ```
   http://127.0.0.1:5000
   ```

## Using the Website

1. **Upload an Image**: Click the "Choose File" button to select an image for analysis
2. **Enable Localization**: Check the "Show Localization" checkbox if you want to see where tampering occurred
3. **Analyze**: Click the "Analyze" button to process the image
4. **View Results**: The results will show whether the image is authentic or tampered
5. **Localization Views**: If tampering is detected and localization is enabled, you'll see:
   - Heatmap: Shows the probability of tampering across the image
   - Overlay: Combines the original image with the heatmap
   - Contour: Highlights the specific tampered regions with green outlines

## Test Results

The model was tested on a diverse set of images from the CASIA2 dataset, achieving an accuracy of 100% on the test set. The system correctly identified both tampered and authentic images with high confidence.

![Test Results](reports/casia2_forgery_detection_report_20250315_220137.png)

To generate your own test report:

```
.\generate_report.bat
```

## Troubleshooting

If you encounter any issues:

1. Make sure all dependencies are installed correctly
2. Check that the model files exist in the correct locations
3. Ensure the image format is supported (JPG, PNG, JPEG, GIF)
4. Try restarting the website using the restart_website.bat script

## Technical Details

The website uses:

- Flask for the web server
- PyTorch for the CNN model
- OpenCV for image processing
- XGBoost and SVM for classification
- Matplotlib for visualization

The model architecture includes:

- Convolutional layers for feature extraction
- Residual connections for better gradient flow
- Attention mechanisms to focus on important features
- SVM classifier for final decision making
