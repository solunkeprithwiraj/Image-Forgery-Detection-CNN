"""
Evaluation script for trained ResNet models.
"""

import os
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc
import seaborn as sns
from PIL import Image
import sys

# Add the project root to the path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

# Import the ResNet models
from models.resnet import ResNet18Forensics, ResNet34Forensics, ResNet50Forensics, ResNet101Forensics, ResNet152Forensics

class ForensicsDataset(torch.utils.data.Dataset):
    """Dataset for image forgery detection."""
    def __init__(self, data_dir, transform=None, is_train=False, train_split=0.8):
        """Initialize the dataset."""
        self.data_dir = data_dir
        self.transform = transform
        
        # Load authentic images
        authentic_dir = os.path.join(data_dir, 'authentic')
        authentic_images = []
        if os.path.exists(authentic_dir):
            authentic_images = [os.path.join('authentic', f) for f in os.listdir(authentic_dir)
                               if f.lower().endswith(('.png', '.jpg', '.jpeg', '.tif', '.tiff'))]
        
        # Load tampered images
        tampered_dir = os.path.join(data_dir, 'tampered')
        tampered_images = []
        if os.path.exists(tampered_dir):
            tampered_images = [os.path.join('tampered', f) for f in os.listdir(tampered_dir)
                              if f.lower().endswith(('.png', '.jpg', '.jpeg', '.tif', '.tiff'))]
        
        # Create labels: 0 for authentic, 1 for tampered
        self.image_paths = authentic_images + tampered_images
        self.labels = [0] * len(authentic_images) + [1] * len(tampered_images)
        
        # Shuffle the data
        indices = list(range(len(self.image_paths)))
        np.random.shuffle(indices)
        self.image_paths = [self.image_paths[i] for i in indices]
        self.labels = [self.labels[i] for i in indices]
        
        # Split into train/test
        split_idx = int(len(self.image_paths) * train_split)
        if is_train:
            self.image_paths = self.image_paths[:split_idx]
            self.labels = self.labels[:split_idx]
        else:
            self.image_paths = self.image_paths[split_idx:]
            self.labels = self.labels[split_idx:]
    
    def __len__(self):
        """Return the number of images."""
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        """Get an image and its label."""
        img_path = os.path.join(self.data_dir, self.image_paths[idx])
        try:
            image = Image.open(img_path).convert('RGB')
        except Exception as e:
            print(f"Error loading image {img_path}: {e}")
            # Return a placeholder image if there's an error
            image = Image.new('RGB', (224, 224), color=(0, 0, 0))
        
        if self.transform:
            image = self.transform(image)
        
        return image, self.labels[idx]

def get_model(model_type, num_classes=2):
    """Get the model based on the model type."""
    if model_type == 'resnet18':
        return ResNet18Forensics(num_classes=num_classes)
    elif model_type == 'resnet34':
        return ResNet34Forensics(num_classes=num_classes)
    elif model_type == 'resnet50':
        return ResNet50Forensics(num_classes=num_classes)
    elif model_type == 'resnet101':
        return ResNet101Forensics(num_classes=num_classes)
    elif model_type == 'resnet152':
        return ResNet152Forensics(num_classes=num_classes)
    else:
        raise ValueError(f"Unsupported model type: {model_type}")

def evaluate_model(model, data_loader, device, criterion=None):
    """Evaluate the model on the test set."""
    model.eval()
    
    all_preds = []
    all_labels = []
    all_probs = []
    total_loss = 0.0
    total = 0
    
    with torch.no_grad():
        for images, labels in data_loader:
            images = images.to(device)
            labels = labels.to(device)
            total += labels.size(0)
            
            outputs = model(images)
            probs = F.softmax(outputs, dim=1)
            
            if criterion:
                loss = criterion(outputs, labels)
                total_loss += loss.item() * images.size(0)
            
            _, predicted = torch.max(outputs.data, 1)
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs[:, 1].cpu().numpy())  # Probability of class 1 (tampered)
    
    accuracy = 100 * sum(np.array(all_preds) == np.array(all_labels)) / total
    if criterion:
        avg_loss = total_loss / total
        return accuracy, avg_loss, all_preds, all_labels, all_probs
    else:
        return accuracy, None, all_preds, all_labels, all_probs

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

def plot_roc_curve(fpr, tpr, roc_auc, output_dir):
    """Plot ROC curve."""
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'roc_curve.png'))
    plt.close()

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Evaluate trained ResNet model')
    parser.add_argument('--data_dir', type=str, default='data/processed',
                      help='Directory containing the processed dataset')
    parser.add_argument('--model_path', type=str, required=True,
                      help='Path to the trained model')
    parser.add_argument('--model_type', type=str, default='resnet18',
                      choices=['resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152'],
                      help='Type of ResNet model')
    parser.add_argument('--batch_size', type=int, default=32,
                      help='Batch size for evaluation')
    parser.add_argument('--output_dir', type=str, default='data/output',
                      help='Directory to save evaluation results')
    parser.add_argument('--num_classes', type=int, default=2,
                      help='Number of classes')
    return parser.parse_args()

def main():
    """Main function to evaluate the model."""
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
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)
    
    print(f"Loaded {len(test_dataset)} test samples")
    
    # Load the model
    model = get_model(args.model_type, num_classes=args.num_classes)
    model.load_state_dict(torch.load(args.model_path, map_location=device))
    model = model.to(device)
    
    # Define the loss function
    criterion = nn.CrossEntropyLoss()
    
    # Evaluate the model
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
    
    # Compute classification report
    report = classification_report(all_labels, all_preds, target_names=class_names)
    print("Classification Report:")
    print(report)
    
    # Save classification report to file
    with open(os.path.join(args.output_dir, 'classification_report.txt'), 'w') as f:
        f.write(f"Model Type: {args.model_type}\n")
        f.write(f"Model Path: {args.model_path}\n")
        f.write(f"Test Accuracy: {accuracy:.2f}%\n")
        if loss:
            f.write(f"Test Loss: {loss:.4f}\n")
        f.write("\nConfusion Matrix:\n")
        f.write(str(cm))
        f.write("\n\nClassification Report:\n")
        f.write(report)
    
    # Compute ROC curve and AUC
    fpr, tpr, _ = roc_curve(all_labels, all_probs)
    roc_auc = auc(fpr, tpr)
    print(f"ROC AUC: {roc_auc:.4f}")
    
    # Plot ROC curve
    plot_roc_curve(fpr, tpr, roc_auc, args.output_dir)
    
    print(f"Evaluation results saved to {args.output_dir}")

if __name__ == "__main__":
    main() 