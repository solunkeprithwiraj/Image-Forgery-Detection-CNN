"""
Minimal version of the improved CNN architecture for image forgery detection.
This version doesn't depend on SRM filters and can be used for demonstration purposes.
"""
import torch.nn.functional as F
import torch.nn as nn
import torch


class ResidualBlock(nn.Module):
    """
    Residual block with skip connections for better gradient flow
    """
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
            
    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class AttentionModule(nn.Module):
    """
    Channel attention module to focus on important features
    """
    def __init__(self, channels, reduction=16):
        super(AttentionModule, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


class MinimalImprovedCNN(nn.Module):
    """
    Minimal version of the improved CNN architecture without SRM filter dependencies
    """
    def __init__(self, feature_dim=400):
        super(MinimalImprovedCNN, self).__init__()
        
        # Initial convolutional layers
        self.conv_init = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        
        # First block of residual layers
        self.res_block1 = nn.Sequential(
            ResidualBlock(32, 64),
            ResidualBlock(64, 64)
        )
        
        self.attention1 = AttentionModule(64)
        
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Second block of residual layers
        self.res_block2 = nn.Sequential(
            ResidualBlock(64, 128, stride=1),
            ResidualBlock(128, 128)
        )
        
        self.attention2 = AttentionModule(128)
        
        # Final convolutional layers
        self.conv_final = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU()
        )
        
        # Feature dimension calculation for adaptive pooling
        self.adaptive_pool = nn.AdaptiveAvgPool2d((5, 5))
        
        # Fully connected layer for classification
        self.fc = nn.Linear(32 * 5 * 5, 2)
        
        # Dropout for regularization
        self.drop = nn.Dropout(p=0.5)
        
        # Feature dimension (for compatibility with existing pipeline)
        self.feature_dim = feature_dim

    def forward(self, x):
        # Initial layers
        x = self.conv_init(x)
        
        # Residual blocks with attention
        x = self.res_block1(x)
        x = self.attention1(x)
        x = self.pool1(x)
        
        x = self.res_block2(x)
        x = self.attention2(x)
        
        # Final convolutions
        x = self.conv_final(x)
        
        # Adaptive pooling to ensure consistent feature size
        x = self.adaptive_pool(x)
        
        # Flatten features
        features = x.view(-1, 32 * 5 * 5)
        
        # In the training phase we also need the fully connected layer with softmax
        if self.training:
            x = self.drop(features)
            x = self.fc(x)
            x = F.log_softmax(x, dim=1)
            return x
        
        return features 