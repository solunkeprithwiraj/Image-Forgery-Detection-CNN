FROM python:3.8-slim

# Set working directory
WORKDIR /app

# Copy requirements file
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create required directories
RUN mkdir -p uploads outputs

# Expose port for the API
EXPOSE 5000

# Define environment variable
ENV PYTHONPATH=/app

# Run the application
CMD ["gunicorn", "--bind", "0.0.0.0:5000", "app:app"] 