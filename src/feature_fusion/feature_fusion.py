import math
import torch
import numpy as np
import os
import sys
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# Add the parent directory to sys.path to allow imports from sibling modules
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))


def get_yi(model, patch):
    """
    Returns the patch's feature representation
    :param model: The pre-trained CNN object
    :param patch: The patch
    :returns: The feature representation of the patch
    """
    with torch.no_grad():
        model.eval()
        return model(patch)


class WrongOperationOption(Exception):
    pass


def get_y_hat(y: np.ndarray, operation: str, weights=None):
    """
    Fuses the image's patches feature representation
    :param y: The network object
    :param operation: Fusion operation ('max', 'mean', 'weighted', 'attention', 'pca')
    :param weights: Optional weights for weighted fusion
    :returns: The final feature representation of the entire image
    """
    if operation == "max":
        return np.array(y).max(axis=0, initial=-math.inf)
    elif operation == "mean":
        return np.array(y).mean(axis=0)
    elif operation == "weighted" and weights is not None:
        # Normalize weights to sum to 1
        weights = np.array(weights) / np.sum(weights)
        # Apply weights to each patch's features
        weighted_features = np.array(y) * weights[:, np.newaxis]
        return weighted_features.sum(axis=0)
    elif operation == "attention":
        # Self-attention based fusion
        features = np.array(y)
        # Calculate similarity matrix
        similarity = np.matmul(features, features.T)
        # Apply softmax to get attention weights
        attention_weights = np.exp(similarity) / np.sum(np.exp(similarity), axis=1, keepdims=True)
        # Apply attention weights
        attention_features = np.matmul(attention_weights, features)
        return attention_features.mean(axis=0)
    elif operation == "pca":
        # PCA-based fusion to reduce redundancy
        features = np.array(y)
        # Standardize features
        scaler = StandardScaler()
        features_std = scaler.fit_transform(features)
        # Apply PCA
        pca = PCA(n_components=min(len(features), features.shape[1]))
        pca_features = pca.fit_transform(features_std)
        # Reconstruct with principal components
        reconstructed = pca.inverse_transform(pca_features)
        # Inverse standardization
        reconstructed = scaler.inverse_transform(reconstructed)
        return reconstructed.mean(axis=0)
    else:
        raise WrongOperationOption("The operation can be 'max', 'mean', 'weighted', 'attention', or 'pca'")


def get_spatial_pyramid_features(model, patches, levels=3):
    """
    Implements spatial pyramid pooling for multi-scale feature extraction
    :param model: The pre-trained CNN object
    :param patches: List of patches organized in spatial order
    :param levels: Number of pyramid levels
    :returns: Multi-scale feature representation
    """
    # Extract features for all patches
    features = []
    for patch in patches:
        with torch.no_grad():
            model.eval()
            features.append(model(patch).cpu().numpy())
    
    features = np.array(features)
    
    # Initialize pyramid features
    pyramid_features = []
    
    # Process each level of the pyramid
    patch_count = len(patches)
    side_length = int(np.sqrt(patch_count))  # Assuming square grid of patches
    
    for level in range(levels):
        # Calculate grid size for this level
        grid_size = 2**level
        
        # Skip if grid is too large for the number of patches
        if grid_size > side_length:
            continue
        
        # Calculate cell size
        cell_height = side_length // grid_size
        cell_width = side_length // grid_size
        
        # Process each cell in the grid
        for i in range(grid_size):
            for j in range(grid_size):
                # Get patches in this cell
                cell_patches = []
                for y in range(i * cell_height, min((i + 1) * cell_height, side_length)):
                    for x in range(j * cell_width, min((j + 1) * cell_width, side_length)):
                        idx = y * side_length + x
                        if idx < patch_count:
                            cell_patches.append(features[idx])
                
                if cell_patches:
                    # Apply max pooling within the cell
                    cell_features = np.array(cell_patches).max(axis=0)
                    pyramid_features.append(cell_features)
    
    # Concatenate all pyramid features
    if pyramid_features:
        return np.concatenate(pyramid_features)
    else:
        return np.array([])


def ensemble_fusion(models, patch, weights=None):
    """
    Ensemble multiple models for more robust feature extraction
    :param models: List of pre-trained models
    :param patch: The input patch
    :param weights: Optional weights for each model
    :returns: Fused feature representation from multiple models
    """
    features = []
    
    # Get features from each model
    for model in models:
        with torch.no_grad():
            model.eval()
            features.append(model(patch).cpu().numpy())
    
    # Apply weights if provided, otherwise use equal weights
    if weights is not None:
        weights = np.array(weights) / np.sum(weights)
        weighted_features = [f * w for f, w in zip(features, weights)]
        return np.sum(weighted_features, axis=0)
    else:
        return np.mean(features, axis=0)
