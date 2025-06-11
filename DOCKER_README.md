# Docker Setup for Image Forgery Detection Application

This README provides instructions on how to run the Image Forgery Detection application using Docker.

## Prerequisites

- Docker installed on your machine
- Docker Compose installed on your machine

## Application Components

This application consists of two main components:

1. **Frontend**: A React-based web interface for uploading images and viewing detection results
2. **Backend**: A FastAPI-based API that handles the image forgery detection logic

## Running the Application

### Using Docker Compose (Recommended)

1. Build and start the containers:

```bash
docker-compose up -d
```

2. Access the application:
   - Frontend: `http://localhost` or `http://localhost:2496`
   - Backend API: `http://localhost:8000`

3. To stop the application:

```bash
docker-compose down
```

### Using Docker Directly (Advanced)

If you want to run the containers separately:

1. Build and run the backend:

```bash
cd backend
docker build -t image-forgery-detection-backend .
docker run -p 8000:8000 -d image-forgery-detection-backend
```

2. Build and run the frontend:

```bash
cd frontend
docker build -t image-forgery-detection-frontend .
docker run -p 80:80 -d image-forgery-detection-frontend
```

## API Documentation

Once the application is running, you can access the API documentation at:

- Swagger UI: `http://localhost:8000/docs`
- ReDoc: `http://localhost:8000/redoc`

## Important Endpoints

- `/forgery/comprehensive` - Comprehensive forgery detection using multiple methods
- `/forgery/copy-move` - Copy-move forgery detection
- `/forgery/splicing` - Image splicing detection
- `/forgery/inpainting` - Inpainting detection
- `/forgery/double-jpeg` - Double JPEG compression detection
- `/forgery/metadata` - Metadata analysis
- `/predict/` - Direct CNN model prediction

## Troubleshooting

If you encounter any issues:

1. Check the logs:

```bash
# View all logs
docker-compose logs

# View only backend logs
docker-compose logs backend

# View only frontend logs
docker-compose logs frontend
```

2. Ensure that the ports are not in use by other applications:
   - Port 80 is used by the frontend
   - Port 8000 is used by the backend API

3. Make sure you have sufficient disk space for the Docker images.

## Notes

- The application data directory is mounted as a volume to persist data.
- The frontend communicates with the backend via the `/api` proxy path.
- Machine learning models are loaded at startup, so the first request may take longer. 