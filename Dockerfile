### Base image for Python backend
FROM python:3.9 AS backend

# Set working directory
WORKDIR /app

# Copy and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the entire project
COPY . .

# Expose the backend port
EXPOSE 5000

# Command to run backend
CMD ["python", "backend/app.py"]

### Base image for Node.js frontend
FROM node:18 AS frontend

# Set working directory
WORKDIR /frontend

# Copy frontend code
COPY react-frontend/ ./

# Install dependencies and build
RUN npm install && npm run build

### Final production image
FROM python:3.9

# Set working directory
WORKDIR /app

# Copy backend and trained models from backend stage
COPY --from=backend /app /app

# Copy built frontend from frontend stage
COPY --from=frontend /frontend/dist /app/backend/static

# Expose the backend port
EXPOSE 5000

# Command to run the application
CMD ["python", "backend/app.py"]
