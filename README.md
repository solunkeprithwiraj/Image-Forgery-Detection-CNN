# Image Tampering Detection System

This project provides a web-based system for detecting image tampering using deep learning. It consists of a Flask backend that uses pre-trained CNN and SVM models to analyze images, and a React frontend that provides a user-friendly interface.

## System Architecture

- **Backend**: Flask application with pre-trained PyTorch models
- **Frontend**: React application with TypeScript and Tailwind CSS
- **Models**: Pre-trained CNN and SVM models for image tampering detection

## Features

- Upload and analyze images to detect tampering
- Ensemble model analysis for higher accuracy
- Visual localization of tampered regions using various methods
- Support for multiple image formats including TIFF files
- Detailed analysis results with confidence scores

## Prerequisites

- Python 3.8+
- Node.js 16+
- Docker (optional)

## Setup and Running

### Option 1: Using Docker (Recommended)

The easiest way to run the application is using Docker and Docker Compose:

```bash
# Clone the repository
git clone <repository-url>
cd image-tampering-detection

# Start the application using Docker Compose
docker-compose up
```

This will start both the backend server on port 5000 and the frontend on port 3000.

### Option 2: Manual Setup

#### Backend Setup

1. Navigate to the backend directory:
```bash
cd backend
```

2. Create a virtual environment:
```bash
python -m venv .venv
```

3. Activate the virtual environment:
   - Windows:
   ```bash
   .venv\Scripts\activate
   ```
   - macOS/Linux:
   ```bash
   source .venv/bin/activate
   ```

4. Install dependencies:
```bash
pip install -r requirements.txt
```

5. Run the Flask application:
```bash
python run.py
```

#### Frontend Setup

1. Navigate to the frontend directory:
```bash
cd frontend
```

2. Install dependencies:
```bash
npm install
```

3. Start the development server:
```bash
npm run dev
```

## Usage

1. Open your browser and navigate to http://localhost:3000
2. Click on "Detect" in the navigation menu
3. Upload an image for analysis
4. View the analysis results including:
   - Tampering detection result
   - Confidence score
   - Visualization of tampered regions
   - Ensemble model results (optional)

## Project Structure

```
.
├── backend/                # Flask backend
│   ├── app.py              # Main Flask application
│   ├── models/             # Model definitions
│   ├── data/               # Data directory
│   │   └── output/         # Pre-trained models
│   ├── uploads/            # Uploaded images directory
│   └── outputs/            # Output images directory
├── frontend/               # React frontend
│   ├── src/                # Source code
│   │   ├── components/     # React components
│   │   ├── pages/          # Page components
│   │   └── services/       # API services
│   └── public/             # Static files
└── docker-compose.yml      # Docker Compose configuration
```

## API Endpoints

- `POST /api/analyze`: Analyze an image using a single model
- `POST /api/analyze/ensemble`: Analyze an image using an ensemble of models
- `POST /api/convert-tiff`: Convert a TIFF image to JPEG
- `GET /api/view-tiff/<path>`: View a TIFF image as JPEG 