FROM python:3.8-slim

# Set working directory
WORKDIR /app

# Install system dependencies required for OpenCV
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements-api.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements-api.txt


# Copy the backend code
COPY . .

# Set Python path to include app directory
ENV PYTHONPATH=/app:$PYTHONPATH

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Expose port 8000
EXPOSE 8000

# Run the FastAPI application
CMD ["python", "run_api.py"] 