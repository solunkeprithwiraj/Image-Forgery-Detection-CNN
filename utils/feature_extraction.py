import torch
from models.cnn import CNN
from utils.feature_vector_generation import create_feature_vectors


def extract_features(model_path, authentic_path, tampered_path, output_filename):
    """
    Extract features from images using a pre-trained CNN model
    
    Args:
        model_path: Path to the pre-trained CNN model
        authentic_path: Path to authentic images
        tampered_path: Path to tampered images
        output_filename: Output filename for the feature vectors
    """
    with torch.no_grad():
        model = CNN()
        model.load_state_dict(torch.load(model_path, map_location=lambda storage, loc: storage))
        model.eval()
        model = model.double()
        
        create_feature_vectors(model, tampered_path, authentic_path, output_filename)
        
    return output_filename

# Example usage
if __name__ == "__main__":
    with torch.no_grad():
        model = CNN()
        model.load_state_dict(torch.load('../data/output/pre_trained_cnn/CASIA2_WithRot_LR001_b128_nodrop.pt',
                                         map_location=lambda storage, loc: storage))
        model.eval()
        model = model.double()

        authentic_path = '../data/CASIA2/Au/*'
        tampered_path = '../data/CASIA2/Tp/*'
        output_filename = 'CASIA2_WithRot_LR001_b128_nodrop.csv'
        create_feature_vectors(model, tampered_path, authentic_path, output_filename)
