# Docker Setup for Image Forgery Detection Application

This README provides instructions on how to run the Image Forgery Detection application using Docker.

## Prerequisites

- Docker installed on your machine
- Docker Compose installed on your machine

## Running the Application

### Using Docker Compose (Recommended)

1. Build and start the containers:

```bash
docker-compose up -d
```

2. The API will be available at `http://localhost:8000`

3. To stop the application:

```bash
docker-compose down
```

### Using Docker Directly

1. Build the Docker image:

```bash
docker build -t image-forgery-detection .
```

2. Run the container:

```bash
docker run -p 8000:8000 -d image-forgery-detection
```

3. The API will be available at `http://localhost:8000`

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
docker-compose logs
```

2. Ensure that the ports are not in use by other applications.

3. Make sure you have sufficient disk space for the Docker images.

## Notes

- The application data directory is mounted as a volume to persist data.
- The application runs on port 8000 by default.
- Machine learning models are loaded at startup, so the first request may take longer. 