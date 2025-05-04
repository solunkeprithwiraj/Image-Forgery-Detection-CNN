"""
Model implementations for Image Forgery Detection.
"""

from models.cnn import CNN, ImprovedCNN
from models.minimal_cnn import MinimalImprovedCNN, ResidualBlock, AttentionModule
from models.SVM import optimize_hyperparams, classify, print_confusion_matrix, find_misclassified
from models.resnet import (
    ResNet18Forensics,
    ResNet34Forensics,
    ResNet50Forensics,
    ResNet101Forensics,
    ResNet152Forensics
)
