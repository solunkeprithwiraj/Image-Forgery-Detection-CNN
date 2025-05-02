import torch.nn.functional as F
import torch.nn as nn
import torch
import numpy as np
from models.SRM_filters import get_filters

class GradientReversalFunction(torch.autograd.Function):
    """
    Gradient Reversal Layer for domain adaptation
    """
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.neg() * ctx.alpha, None

class GradientReversal(nn.Module):
    """
    Gradient Reversal Layer for domain adaptation
    """
    def __init__(self, alpha=1.0):
        super(GradientReversal, self).__init__()
        self.alpha = alpha
        
    def forward(self, x):
        return GradientReversalFunction.apply(x, self.alpha)

class RobustCNN(nn.Module):
    """
    More robust CNN architecture with:
    1. Stronger regularization
    2. Domain adaptation
    3. Feature normalization
    4. Reduced complexity to avoid overfitting
    """
    def __init__(self, dropout_rate=0.5):
        super(RobustCNN, self).__init__()
        
        # Initial SRM filter layer (for noise pattern analysis)
        self.conv0 = nn.Conv2d(3, 3, kernel_size=5, stride=1, padding=2)
        nn.init.xavier_uniform_(self.conv0.weight)
        
        # First conv layer with SRM filters - keeps the same as original to maintain forensic sensitivity
        self.conv1 = nn.Conv2d(3, 30, kernel_size=5, stride=2, padding=2)
        self.conv1.weight = nn.Parameter(get_filters())
        self.bn1 = nn.BatchNorm2d(30)
        
        # Simplified architecture with stronger regularization
        self.conv2 = nn.Conv2d(30, 30, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(30)
        self.dropout2 = nn.Dropout2d(0.2)  # Spatial dropout
        
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        
        self.conv3 = nn.Conv2d(30, 16, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(16)
        self.dropout3 = nn.Dropout2d(0.2)
        
        self.conv4 = nn.Conv2d(16, 16, kernel_size=3, stride=1, padding=1)
        self.bn4 = nn.BatchNorm2d(16)
        
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        
        self.conv5 = nn.Conv2d(16, 16, kernel_size=3, stride=1, padding=1)
        self.bn5 = nn.BatchNorm2d(16)
        
        # Feature normalization layer (to reduce domain differences)
        self.instance_norm = nn.InstanceNorm1d(400, affine=True)
        
        # Domain classifier for domain adaptation
        self.domain_classifier = nn.Sequential(
            GradientReversal(alpha=1.0),
            nn.Linear(400, 100),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(100, 2)  # Binary domain classification
        )
        
        # Final classification layer
        self.fc = nn.Linear(400, 2)
        self.dropout = nn.Dropout(dropout_rate)

    def features(self, x):
        """
        Extract features with enhanced regularization
        """
        # Input normalization
        x = x.float()  # Ensure float type
        
        # Initial layers with SRM filters
        x = F.relu(self.conv0(x))
        x = F.relu(self.bn1(self.conv1(x)))
        
        # Enhanced feature extraction with regularization
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.dropout2(x)
        x = self.pool1(x)
        
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.dropout3(x)
        
        x = F.relu(self.bn4(self.conv4(x)))
        x = self.pool2(x)
        
        x = F.relu(self.bn5(self.conv5(x)))
        
        # Flatten features
        features = x.view(x.size(0), -1)
        
        # Apply instance normalization to reduce domain differences
        features = self.instance_norm(features.unsqueeze(1)).squeeze(1)
        
        return features
    
    def domain_classify(self, features):
        """
        Classify domain (source/target) for domain adaptation
        """
        return self.domain_classifier(features)

    def forward(self, x):
        """
        Forward pass through the network
        """
        features = self.features(x)
        
        # In training phase, apply dropout and classification
        if self.training:
            features = self.dropout(features)
            class_output = self.fc(features)
            domain_output = self.domain_classify(features)
            return class_output, domain_output
        
        # In evaluation phase, just return features
        return features

    def get_domain_adaptation_loss(self, source_images, target_images):
        """
        Calculate domain adaptation loss using source and target images
        """
        self.train()
        
        # Get features and domain predictions for source images
        source_features = self.features(source_images)
        source_domain_preds = self.domain_classify(source_features)
        
        # Get features and domain predictions for target images
        target_features = self.features(target_images)
        target_domain_preds = self.domain_classify(target_features)
        
        # Create domain labels (0 for source, 1 for target)
        source_domain_labels = torch.zeros(source_images.size(0), dtype=torch.long, device=source_images.device)
        target_domain_labels = torch.ones(target_images.size(0), dtype=torch.long, device=target_images.device)
        
        # Calculate domain classification loss
        domain_criterion = nn.CrossEntropyLoss()
        source_domain_loss = domain_criterion(source_domain_preds, source_domain_labels)
        target_domain_loss = domain_criterion(target_domain_preds, target_domain_labels)
        
        return source_domain_loss + target_domain_loss

def load_model_from_original(original_model_path, device='cpu'):
    """
    Load weights from original CNN model into the robust model
    """
    # Create new robust model
    robust_model = RobustCNN()
    
    # Load original model state dict
    original_state_dict = torch.load(original_model_path, map_location=device)
    
    # Copy weights for matching layers
    robust_dict = robust_model.state_dict()
    
    # Copy weights for shared layers
    for name, param in original_state_dict.items():
        if name in robust_dict and robust_dict[name].shape == param.shape:
            robust_dict[name] = param
    
    # Load weights into robust model
    robust_model.load_state_dict(robust_dict, strict=False)
    
    return robust_model 