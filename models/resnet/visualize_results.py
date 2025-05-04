"""
Visualization script for ResNet model results.
"""

import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix, roc_curve, auc
import seaborn as sns
from PIL import Image
import cv2
import random
import sys

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

# Import the ResNet models
from models.resnet import ResNet18Forensics, ResNet34Forensics, ResNet50Forensics, ResNet101Forensics, ResNet152Forensics
from models.resnet.evaluate_resnet import ForensicsDataset, get_model, evaluate_model

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Visualize ResNet model results')
    parser.add_argument('--data_dir', type=str, default='data/processed',
                      help='Directory containing the processed dataset')
    parser.add_argument('--model_path', type=str, required=True,
                      help='Path to the trained model')
    parser.add_argument('--model_type', type=str, default='resnet18',
                      choices=['resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152'],
                      help='Type of ResNet model')
    parser.add_argument('--output_dir', type=str, default='data/output/visualization',
                      help='Directory to save visualization results')
    parser.add_argument('--num_samples', type=int, default=10,
                      help='Number of samples to visualize for each category')
    return parser.parse_args()

def plot_roc_curve(all_labels, all_probs, output_dir):
    """Plot ROC curve."""
    fpr, tpr, _ = roc_curve(all_labels, all_probs)
    roc_auc = auc(fpr, tpr)
    
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc="lower right")
    plt.savefig(os.path.join(output_dir, 'roc_curve.png'))
    plt.close()
    
    return roc_auc

def plot_confusion_matrix(cm, class_names, output_dir):
    """Plot confusion matrix."""
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'confusion_matrix.png'))
    plt.close()

def visualize_misclassified(model, data_loader, device, output_dir, num_samples=10):
    """Visualize misclassified samples."""
    model.eval()
    
    misclassified_authentic = []  # Authentic images classified as tampered
    misclassified_tampered = []  # Tampered images classified as authentic
    
    with torch.no_grad():
        for images, labels in data_loader:
            images = images.to(device)
            labels = labels.to(device)
            
            outputs = model(images)
            _, preds = torch.max(outputs, 1)
            
            # Find misclassified samples
            for i in range(len(preds)):
                if preds[i] != labels[i]:
                    original_image = images[i].cpu().permute(1, 2, 0).numpy()
                    
                    # Denormalize the image
                    mean = np.array([0.485, 0.456, 0.406])
                    std = np.array([0.229, 0.224, 0.225])
                    original_image = std * original_image + mean
                    original_image = np.clip(original_image, 0, 1)
                    
                    if labels[i] == 0:  # Authentic classified as tampered
                        misclassified_authentic.append((original_image, outputs[i].cpu().numpy()))
                    else:  # Tampered classified as authentic
                        misclassified_tampered.append((original_image, outputs[i].cpu().numpy()))
    
    # Visualize misclassified authentic images
    if misclassified_authentic:
        num_samples = min(num_samples, len(misclassified_authentic))
        samples = random.sample(misclassified_authentic, num_samples)
        
        fig, axs = plt.subplots(2, num_samples // 2, figsize=(15, 6))
        axs = axs.flatten()
        
        for i, (img, output) in enumerate(samples):
            confidence = torch.nn.functional.softmax(torch.tensor(output), dim=0)[1].item()
            axs[i].imshow(img)
            axs[i].set_title(f"Authentic (Pred: Tampered)\nConf: {confidence:.2f}")
            axs[i].axis('off')
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'misclassified_authentic.png'))
        plt.close()
    
    # Visualize misclassified tampered images
    if misclassified_tampered:
        num_samples = min(num_samples, len(misclassified_tampered))
        samples = random.sample(misclassified_tampered, num_samples)
        
        fig, axs = plt.subplots(2, num_samples // 2, figsize=(15, 6))
        axs = axs.flatten()
        
        for i, (img, output) in enumerate(samples):
            confidence = torch.nn.functional.softmax(torch.tensor(output), dim=0)[0].item()
            axs[i].imshow(img)
            axs[i].set_title(f"Tampered (Pred: Authentic)\nConf: {confidence:.2f}")
            axs[i].axis('off')
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'misclassified_tampered.png'))
        plt.close()
    
    print(f"Found {len(misclassified_authentic)} misclassified authentic images")
    print(f"Found {len(misclassified_tampered)} misclassified tampered images")

def main():
    """Main function to visualize results."""
    args = parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Define transforms
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Load the test dataset
    test_dataset = ForensicsDataset(args.data_dir, transform=transform, is_train=False)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=4)
    
    print(f"Loaded {len(test_dataset)} test samples")
    
    # Load the model
    model = get_model(args.model_type)
    model.load_state_dict(torch.load(args.model_path, map_location=device))
    model = model.to(device)
    
    # Evaluate the model
    criterion = torch.nn.CrossEntropyLoss()
    accuracy, loss, all_preds, all_labels, all_probs = evaluate_model(
        model, test_loader, device, criterion)
    
    print(f"Test Accuracy: {accuracy:.2f}%")
    if loss:
        print(f"Test Loss: {loss:.4f}")
    
    # Compute confusion matrix
    cm = confusion_matrix(all_labels, all_preds)
    print("Confusion Matrix:")
    print(cm)
    
    # Plot confusion matrix
    class_names = ['Authentic', 'Tampered']
    plot_confusion_matrix(cm, class_names, args.output_dir)
    
    # Plot ROC curve
    roc_auc = plot_roc_curve(all_labels, all_probs, args.output_dir)
    print(f"ROC AUC: {roc_auc:.4f}")
    
    # Visualize misclassified samples
    visualize_misclassified(model, test_loader, device, args.output_dir, args.num_samples)
    
    print(f"Visualization results saved to {args.output_dir}")

if __name__ == "__main__":
    main() 