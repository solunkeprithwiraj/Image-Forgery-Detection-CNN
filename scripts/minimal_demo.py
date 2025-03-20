"""
Minimal demonstration of the improved CNN architecture for image forgery detection.
This script only requires PyTorch and demonstrates the model structure.
"""
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import os
import sys

# Add the parent directory to sys.path to allow imports from sibling modules
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import the CNN model
from models.minimal_cnn import MinimalImprovedCNN, ResidualBlock, AttentionModule


def main():
    """
    Minimal demonstration of the improved CNN architecture
    """
    print("=== Minimal Demo of Improved Image Forgery Detection CNN ===")
    
    # Create the minimal improved CNN model
    model = MinimalImprovedCNN()
    
    # Print model architecture
    print("\nModel Architecture:")
    print(model)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"\nTotal parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    # Demonstrate forward pass with random input
    print("\nDemonstrating forward pass with random input...")
    
    # Create a random batch of 4 RGB images of size 128x128
    batch_size = 4
    input_tensor = torch.rand(batch_size, 3, 128, 128)
    
    # Set model to evaluation mode
    model.eval()
    
    # Forward pass
    with torch.no_grad():
        features = model(input_tensor)
    
    print(f"Input shape: {input_tensor.shape}")
    print(f"Output feature shape: {features.shape}")
    
    print("\n=== Demonstration Complete ===")
    print("\nThe improved CNN architecture includes:")
    print("1. Residual connections for better gradient flow")
    print("2. Attention mechanisms to focus on important features")
    print("3. Batch normalization for training stability")
    print("4. Deeper network with more capacity")
    print("5. Adaptive pooling for consistent feature dimensions")


if __name__ == "__main__":
    main() 