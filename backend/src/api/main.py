import time
import os
from fastapi import FastAPI, Depends, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from src.api.routes.prediction_routes import router as prediction_router
from src.api.utils.logger import setup_logger

# Setup logger
logger = setup_logger()

# Create FastAPI app
app = FastAPI(
    title="Image Forgery Detection API",
    description="API for detecting tampered images using CNN and SVM models",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Import dependencies from utils
from src.api.utils.dependencies import get_models

# Include routers
app.include_router(
    prediction_router,
    prefix="/api",
    dependencies=[],  # Remove dependency from router level
    tags=["predictions"]
)

# Startup event
@app.on_event("startup")
async def startup_event():
    """
    Load models on startup
    """
    logger.info("API starting...")
    start_time = time.time()
    try:
        # Preload models
        get_models()
        logger.info(f"Models loaded successfully in {time.time() - start_time:.2f} seconds")
    except Exception as e:
        logger.error(f"Error during startup: {str(e)}", exc_info=True)

# Root endpoint
@app.get("/", tags=["status"])
async def root():
    """
    Root endpoint to check API status
    :returns: API status
    """
    return {"status": "online", "message": "Image Forgery Detection API is running"}

# Health check endpoint
@app.get("/health", tags=["status"])
async def health_check():
    """
    Health check endpoint
    :returns: API health status
    """
    try:
        # Try to load models to verify they're working
        cnn_model, svm_model = get_models()
        return {"status": "healthy", "models_loaded": True}
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"status": "unhealthy", "error": str(e)}
        )

# Exception handler
@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    """
    Global exception handler
    :param request: The request that caused the exception
    :param exc: The exception
    :returns: JSON response with error details
    """
    logger.error(f"Unhandled exception: {str(exc)}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={"detail": f"Internal server error: {str(exc)}"}  
    )

# Run the application
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("src.api.main:app", host="0.0.0.0", port=8000, reload=True)