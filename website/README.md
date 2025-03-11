# Image Forgery Detection Website

A modern, responsive web application that integrates with the Image Forgery Detection CNN model to detect tampered images.

## Features

- User-friendly interface for image upload and analysis
- Drag-and-drop functionality for easy image uploading
- Real-time image forgery detection using a pre-trained CNN + SVM model
- Responsive design that works on desktop and mobile devices
- Detailed results with confidence scores

## Technologies Used

- **Backend**: Flask (Python)
- **Frontend**: HTML5, CSS3, JavaScript
- **CSS Framework**: Bootstrap 5
- **Icons**: Font Awesome
- **Machine Learning**: PyTorch, scikit-learn

## Setup and Installation

1. Install the required dependencies:

   ```
   pip install -r requirements.txt
   ```

2. Run the Flask application:

   ```
   python app.py
   ```

3. Open your browser and navigate to:
   ```
   http://localhost:5000
   ```

## Project Structure

```
website/
├── app.py                 # Main Flask application
├── requirements.txt       # Python dependencies
├── static/                # Static files
│   ├── css/               # CSS stylesheets
│   │   └── style.css      # Custom CSS
│   ├── js/                # JavaScript files
│   │   └── main.js        # Custom JS
│   ├── img/               # Images
│   │   └── hero-image.svg # Hero image
│   └── uploads/           # Uploaded images (created at runtime)
└── templates/             # HTML templates
    ├── base.html          # Base template
    ├── index.html         # Home page
    └── about.html         # About page
```

## How It Works

1. The user uploads an image through the web interface
2. The image is processed by the CNN model to extract features
3. The SVM classifier analyzes these features to determine if the image is authentic or tampered
4. The result is displayed to the user with a confidence score

## Model Performance

The model achieves the following accuracy on test datasets:

- CASIA2: 96.82% ± 1.19%
- NC2016: 84.89% ± 6.06%

## Credits

This website integrates the Image Forgery Detection CNN model developed by:

- Prithwiraj Solunke
- Omkar Kharmare
- Gaurav Ghadage

Original repository: [Image-Forgery-Detection-CNN](https://github.com/solunkeprithwiraj/Image-Forgery-Detection-CNN)
