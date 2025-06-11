from fastapi import Depends, HTTPException
import os
import time
from src.api.services.model_loader import load_models
from src.api.utils.logger import logger

# Model dependency
def get_models():
    """
    Dependency to load models for each request
    :returns: Tuple of (cnn_model, svm_model)
    """
    try:
        # Get the absolute path to the project root directory
        project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
        
        # Define model paths
        cnn_model_path = os.path.join(project_root, 'data', 'output', 'pre_trained_cnn', 'CASIA2_NoRot_LR0001_b200_nodrop.pt')
        svm_model_path = os.path.join(project_root, 'data', 'output', 'pre_trained_svm', 'CASIA2_WithRot_LR001_b128_nodrop.pt')
        
        # Load models
        cnn_model, svm_model = load_models(cnn_model_path, svm_model_path)
        return cnn_model, svm_model
    except Exception as e:
        logger.error(f"Error loading models: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error loading models: {str(e)}")

# Function to inject models into route handlers
def inject_models(models=Depends(get_models)):
    """
    Dependency to inject models into route handlers
    :param models: Tuple of (cnn_model, svm_model) from get_models dependency
    :returns: Dictionary with cnn_model and svm_model
    """
    # get_models returns a tuple of (cnn_model, svm_model)
    # We unpack it here to pass to the route handlers
    return {"cnn_model": models[0], "svm_model": models[1]}