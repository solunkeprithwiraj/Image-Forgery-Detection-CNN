"""
Training script for ResNet models.
"""
import os
import sys
import time
import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import cv2
import glob
from datetime import datetime
from PIL import Image
from tqdm import tqdm

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from models.resnet import (
    ResNet18Forensics,
    ResNet34Forensics,
    ResNet50Forensics,
    ResNet101Forensics,
    ResNet152Forensics
)
from utils.common import (
    ensure_dir,
    plot_training_history,
    plot_confusion_matrix,
    plot_roc_curve,
    save_classification_report
)
from configs.model_config import CNN_CONFIG

class ForensicsDataset(Dataset):
    """Dataset for image forgery detection."""
    def __init__(self, data_dir, transform=None, is_train=True, train_split=0.8):
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
        label = self.labels[idx]
        
        try:
            # Always use PIL to load the image
            image = Image.open(img_path).convert('RGB')
            
            # Apply transforms
            if self.transform:
                image = self.transform(image)
                
            return image, label
            
        except Exception as e:
            print(f"Error loading image {img_path}: {e}")
            # Return a placeholder image if there's an error
            if self.transform:
                # Create a black PIL image and apply the transform
                placeholder = Image.new('RGB', (224, 224), color=(0, 0, 0))
                return self.transform(placeholder), label
            else:
                # Return a tensor placeholder
                return torch.zeros((3, 224, 224)), label

def get_transforms():
    """Get the transforms for the datasets."""
    train_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.RandomCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    test_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    return train_transform, test_transform

def get_model(model_type, num_classes=2, srm_filters=True):
    """Get the model based on the model type."""
    if model_type == 'resnet18':
        return ResNet18Forensics(num_classes=num_classes, srm_filters=srm_filters)
    elif model_type == 'resnet34':
        return ResNet34Forensics(num_classes=num_classes, srm_filters=srm_filters)
    elif model_type == 'resnet50':
        return ResNet50Forensics(num_classes=num_classes, srm_filters=srm_filters)
    elif model_type == 'resnet101':
        return ResNet101Forensics(num_classes=num_classes, srm_filters=srm_filters)
    elif model_type == 'resnet152':
        return ResNet152Forensics(num_classes=num_classes, srm_filters=srm_filters)
    else:
        raise ValueError(f"Unsupported model type: {model_type}")

def get_optimizer(optimizer_type, model_parameters, learning_rate):
    """Get the optimizer based on the optimizer type."""
    if optimizer_type.lower() == 'adam':
        return optim.Adam(model_parameters, lr=learning_rate)
    elif optimizer_type.lower() == 'sgd':
        return optim.SGD(model_parameters, lr=learning_rate, momentum=0.9, weight_decay=5e-4)
    elif optimizer_type.lower() == 'rmsprop':
        return optim.RMSprop(model_parameters, lr=learning_rate)
    else:
        raise ValueError(f"Unsupported optimizer type: {optimizer_type}")

def train_model(model, train_loader, test_loader, criterion, optimizer, scheduler, num_epochs, device, output_dir):
    """Train the model."""
    best_acc = 0.0
    history = {'train_loss': [], 'train_acc': [], 'test_loss': [], 'test_acc': []}
    
    # Set model to training mode
    model.train()
    
    print("Starting training...")
    for epoch in range(num_epochs):
        print(f"Epoch {epoch+1}/{num_epochs}")
        print('-' * 10)
        
        # Training phase
        model.train()
        running_loss = 0.0
        running_corrects = 0
        total = 0
        
        pbar = tqdm(train_loader, desc=f"Training {epoch+1}/{num_epochs}")
        for inputs, labels in pbar:
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            # Zero the parameter gradients
            optimizer.zero_grad()
            
            # Forward pass
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            loss = criterion(outputs, labels)
            
            # Backward pass and optimize
            loss.backward()
            optimizer.step()
            
            # Statistics
            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)
            total += labels.size(0)
            
            # Update progress bar
            pbar.set_postfix(loss=loss.item(), acc=f"{(torch.sum(preds == labels.data)/labels.size(0))*100:.2f}%")
        pbar.close()
        
        # Update the learning rate scheduler
        if scheduler:
            scheduler.step()
        
        # Calculate epoch loss and accuracy
        epoch_loss = running_loss / total
        epoch_acc = running_corrects.double() / total * 100.0
        
        print(f"Train Loss: {epoch_loss:.4f} Acc: {epoch_acc:.2f}%")
        
        # Save the epoch statistics
        history['train_loss'].append(epoch_loss)
        history['train_acc'].append(epoch_acc.item())
        
        # Evaluation phase
        model.eval()
        running_loss = 0.0
        running_corrects = 0
        total = 0
        
        with torch.no_grad():
            pbar = tqdm(test_loader, desc=f"Validation {epoch+1}/{num_epochs}")
            for inputs, labels in pbar:
                inputs = inputs.to(device)
                labels = labels.to(device)
                
                # Forward pass
                outputs = model(inputs)
                _, preds = torch.max(outputs, 1)
                loss = criterion(outputs, labels)
                
                # Statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
                total += labels.size(0)
                
                # Update progress bar
                pbar.set_postfix(loss=loss.item(), acc=f"{(torch.sum(preds == labels.data)/labels.size(0))*100:.2f}%")
            pbar.close()
        
        # Calculate epoch loss and accuracy
        epoch_loss = running_loss / total
        epoch_acc = running_corrects.double() / total * 100.0
        
        print(f"Test Loss: {epoch_loss:.4f} Acc: {epoch_acc:.2f}%")
        
        # Save the epoch statistics
        history['test_loss'].append(epoch_loss)
        history['test_acc'].append(epoch_acc.item())
        
        # Save the best model
        if epoch_acc > best_acc:
            best_acc = epoch_acc
            # Save model
            model_path = os.path.join(output_dir, 'models', f"{args.model_type}_best.pth")
            torch.save(model.state_dict(), model_path)
            print(f"New best model saved with accuracy: {best_acc:.2f}%")
        
        print()
    
    # Save the final model
    model_path = os.path.join(output_dir, 'models', f"{args.model_type}_final.pth")
    torch.save(model.state_dict(), model_path)
    print(f"Final model saved with accuracy: {epoch_acc:.2f}%")
    
    # Plot and save the training history
    plot_history(history, output_dir)
    
    return model, history

def plot_history(history, output_dir):
    """Plot and save the training history."""
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(history['train_loss'], label='Train')
    plt.plot(history['test_loss'], label='Test')
    plt.title('Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(history['train_acc'], label='Train')
    plt.plot(history['test_acc'], label='Test')
    plt.title('Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"{args.model_type}_history.png"))
    plt.close()

def evaluate_model(model, data_loader, criterion, device):
    """Evaluate the model on the test set."""
    model.eval()
    
    running_loss = 0.0
    running_corrects = 0
    total = 0
    
    all_preds = []
    all_labels = []
    all_probs = []
    
    with torch.no_grad():
        for inputs, labels in tqdm(data_loader, desc="Evaluating"):
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            outputs = model(inputs)
            probs = torch.nn.functional.softmax(outputs, dim=1)
            _, preds = torch.max(outputs, 1)
            loss = criterion(outputs, labels)
            
            # Statistics
            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)
            total += labels.size(0)
            
            # Store predictions and labels
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs[:, 1].cpu().numpy())  # Probability for positive class
    
    # Calculate metrics
    test_loss = running_loss / total
    test_acc = running_corrects.double() / total * 100.0
    
    return test_loss, test_acc.item(), all_preds, all_labels, all_probs

def save_model(model, model_path, model_info=None):
    """Save the trained model and its information."""
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    
    # Save model state dictionary
    torch.save({
        'model_state_dict': model.state_dict(),
        'info': model_info
    }, model_path)
    
    print(f"Model saved to {model_path}")

def main(args):
    # Set device
    if args.use_gpu and torch.cuda.is_available():
        device = torch.device('cuda')
        print(f"Using cuda device: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device('cpu')
        print(f"Using cpu device")
        
        # Optimize for CPU training
        torch.set_num_threads(4)  # Limit number of threads
    
    # Set random seeds for reproducibility
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    # Create data transformations
    train_transform, test_transform = get_transforms()
    
    # Create datasets
    train_dataset = ForensicsDataset(args.data_dir, transform=train_transform, is_train=True)
    test_dataset = ForensicsDataset(args.data_dir, transform=test_transform, is_train=False)
    
    print(f"Training set: {len(train_dataset)} images")
    print(f"Testing set: {len(test_dataset)} images")
    
    # Create data loaders
    pin_memory = args.pin_memory and torch.cuda.is_available()
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, 
                             num_workers=args.num_workers, pin_memory=pin_memory)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, 
                            num_workers=args.num_workers, pin_memory=pin_memory)
    
    # Create model
    model = get_model(args.model_type, num_classes=2, srm_filters=args.use_srm)
    model = model.to(device)
    
    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = get_optimizer(args.optimizer, model.parameters(), args.learning_rate)
    
    # Define learning rate scheduler
    scheduler = None
    if args.scheduler == 'step':
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)
    elif args.scheduler == 'cosine':
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    
    # Train model
    print("Starting training...")
    model, history = train_model(
        model,
        train_loader,
        test_loader,
        criterion,
        optimizer,
        scheduler,
        args.epochs,
        device,
        args.output_dir
    )
    
    # Create results directory if it doesn't exist
    results_dir = os.path.join(args.output_dir, args.model_type)
    os.makedirs(results_dir, exist_ok=True)
    
    # Plot training history
    plt.figure(figsize=(12, 5))
    
    # Plot training and validation accuracy
    plt.subplot(1, 2, 1)
    plt.plot(history['train_acc'], label='Train')
    plt.plot(history['test_acc'], label='Test')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Training and Test Accuracy')
    plt.legend()
    
    # Plot training and validation loss
    plt.subplot(1, 2, 2)
    plt.plot(history['train_loss'], label='Train')
    plt.plot(history['test_loss'], label='Test')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Test Loss')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, 'training_history.png'))
    plt.close()
    
    # Evaluate model on test set
    print("Evaluating model on test set...")
    test_loss, test_acc, all_preds, all_labels, all_probs = evaluate_model(
        model,
        test_loader,
        criterion,
        device=device
    )
    
    # Plot confusion matrix
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Confusion Matrix')
    plt.colorbar()
    
    classes = ['Authentic', 'Tampered']
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)
    
    # Add text annotations
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, format(cm[i, j], 'd'),
                    horizontalalignment="center",
                    color="white" if cm[i, j] > thresh else "black")
    
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.savefig(os.path.join(results_dir, 'confusion_matrix.png'))
    plt.close()
    
    # Generate classification report
    report = classification_report(all_labels, all_preds, target_names=classes)
    with open(os.path.join(results_dir, 'classification_report.txt'), 'w') as f:
        f.write(report)
    
    # Save model
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_filename = f"{args.model_type}_{timestamp}.pt"
    model_path = os.path.join(args.output_dir, 'models', model_filename)
    
    # Create model info dictionary
    model_info = {
        'model_type': args.model_type,
        'epochs': args.epochs,
        'batch_size': args.batch_size,
        'learning_rate': args.learning_rate,
        'optimizer': args.optimizer,
        'test_accuracy': test_acc,
        'history': history,
        'timestamp': timestamp
    }
    
    # Save model and info
    save_model(model, model_path, model_info)
    
    print(f"Results saved to {results_dir}")
    print(f"Test accuracy: {test_acc:.4f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train ResNet model for image forgery detection')
    
    # Data parameters
    parser.add_argument('--data_dir', type=str, default='data/processed',
                        help='Path to the data directory containing authentic and tampered folders')
    parser.add_argument('--output_dir', type=str, default='data/output',
                        help='Path to the output directory for saving results')
    
    # Model parameters
    parser.add_argument('--model_type', type=str, default='resnet18',
                        choices=['resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152'],
                        help='Type of ResNet model to use')
    
    # Training parameters
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size for training')
    parser.add_argument('--epochs', type=int, default=30,
                        help='Number of epochs to train for')
    parser.add_argument('--learning_rate', type=float, default=0.001,
                        help='Learning rate')
    parser.add_argument('--optimizer', type=str, default='adam',
                        choices=['adam', 'sgd', 'rmsprop'],
                        help='Optimizer to use for training')
    parser.add_argument('--scheduler', type=str, default='step',
                        choices=['step', 'cosine', 'none'],
                        help='Learning rate scheduler')
    
    # Other parameters
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for reproducibility')
    parser.add_argument('--use_srm', action='store_true',
                        help='Use SRM filters')
    parser.add_argument('--pin_memory', action='store_true',
                        help='Pin memory for faster data transfer')
    parser.add_argument('--use_gpu', action='store_true',
                        help='Use GPU for training if available')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='Number of workers for data loading')
    
    args = parser.parse_args()
    
    # Record start time
    start_time = time.time()
    
    # Train the model
    main(args)
    
    # Record end time and print total time
    end_time = time.time()
    total_time = end_time - start_time
    hours, remainder = divmod(total_time, 3600)
    minutes, seconds = divmod(remainder, 60)
    print(f"Total training time: {int(hours)}h {int(minutes)}m {int(seconds)}s") 