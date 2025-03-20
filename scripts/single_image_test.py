import torch
from models.cnn import CNN
from skimage import io
import matplotlib.pyplot as plt
from joblib import load
import numpy as np
import os

from utils.feature_vector_generation import get_patch_yi


def get_feature_vector(image_path: str, model):
    feature_vector = np.empty((1, 400))
    feature_vector[0, :] = get_patch_yi(model, io.imread(image_path))
    return feature_vector


# Get the current directory
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)

# Load the pretrained CNN with the CASIA2 dataset
with torch.no_grad():
    our_cnn = CNN()
    cnn_path = os.path.join(project_root, 'data', 'output', 'pre_trained_cnn', 'CASIA2_WithRot_LR001_b128_nodrop.pt')
    our_cnn.load_state_dict(torch.load(cnn_path, map_location=lambda storage, loc: storage))
    our_cnn.eval()
    our_cnn = our_cnn.double()

# Load the pretrained svm model
svm_path = os.path.join(project_root, 'data', 'output', 'pre_trained_svm', 'CASIA2_WithRot_LR001_b128_nodrop.pt')
svm_model = load(svm_path)

print("Labels are 0 for non-tampered and 1 for tampered")

# Probe the SVM model with a non-tampered image
non_tampered_image_path = os.path.join(project_root, 'data', 'test_images', 'Au_ani_00002.jpg')
non_tampered_image_feature_vector = get_feature_vector(non_tampered_image_path, our_cnn)
print("Non tampered prediction:", svm_model.predict(non_tampered_image_feature_vector))

# Probe the SVM model with a tampered image
tampered_image_path = os.path.join(project_root, 'data', 'test_images', 'Tp_D_CNN_M_B_nat00056_nat00099_11105.jpg')
tampered_image_feature_vector = get_feature_vector(tampered_image_path, our_cnn)
print("Tampered prediction:", svm_model.predict(tampered_image_feature_vector))
