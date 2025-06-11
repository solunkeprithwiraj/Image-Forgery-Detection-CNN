import uvicorn
import os
import sys

# Add the project root to the Python path
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

if __name__ == "__main__":
    print("Starting Image Forgery Detection API...")
    print(f"Project root: {project_root}")
    
    # Run the FastAPI application
    uvicorn.run(
        "src.api.main:app", 
        host="0.0.0.0", 
        port=8000, 
        reload=True,
        log_level="info"
    )