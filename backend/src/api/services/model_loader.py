import os
import time
import torch
from joblib import load
from sklearn import svm
from src.cnn.cnn import CNN
from src.api.utils.logger import logger

# Global variables for model caching
_cnn_model = None
_svm_model = None

def load_models(cnn_model_path=None, svm_model_path=None):
    """Load CNN and SVM models with caching and ensure probability=True for SVM
    
    :param cnn_model_path: Path to the CNN model file
    :param svm_model_path: Path to the SVM model file
    :returns: Tuple of (cnn_model, svm_model)
    """
    global _cnn_model, _svm_model
    
    # Return cached models if already loaded
    if _cnn_model is not None and _svm_model is not None:
        logger.info("Using cached models")
        return _cnn_model, _svm_model
    
    logger.info("Loading CNN and SVM models...")
    # If paths are not provided, use default paths
    if cnn_model_path is None or svm_model_path is None:
        # Get the absolute path to the project root directory
        project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
        
        # Set default paths if not provided
        if cnn_model_path is None:
            cnn_model_path = os.path.join(project_root, 'data', 'output', 'pre_trained_cnn', 'CASIA2_NoRot_LR0001_b200_nodrop.pt')
        
        if svm_model_path is None:
            svm_model_path = os.path.join(project_root, 'data', 'output', 'pre_trained_svm', 'CASIA2_WithRot_LR001_b128_nodrop.pt')
    
    # Load the pretrained CNN with the CASIA2 dataset
    start_time = time.time()
    logger.debug(f"Loading CNN model from: {cnn_model_path}")
    with torch.no_grad():
        _cnn_model = CNN()
        _cnn_model.load_state_dict(torch.load(cnn_model_path, map_location=lambda storage, loc: storage))
        _cnn_model.eval()
        _cnn_model = _cnn_model.double()
    
    # Load the pretrained svm model
    logger.debug(f"Loading SVM model from: {svm_model_path}")
    _svm_model = load(svm_model_path)
    
    # Ensure the SVM model has probability=True
    if hasattr(_svm_model, 'probability') and not _svm_model.probability:
        logger.warning("SVM model loaded without probability=True. Creating a new model with probability=True")
        # Create a new SVM model with the same parameters but with probability=True
        new_model = svm.SVC(
            kernel=_svm_model.kernel,
            C=_svm_model.C,
            gamma=_svm_model.gamma,
            probability=True
        )
        # Copy the support vectors and other attributes
        new_model.fit(_svm_model.support_vectors_, _svm_model.predict(_svm_model.support_vectors_))
        _svm_model = new_model
    
    elapsed_time = time.time() - start_time
    logger.info(f"Models loaded successfully in {elapsed_time:.2f} seconds")
    return _cnn_model, _svm_model