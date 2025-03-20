import torch.nn.functional as f
import torch.nn as nn
import torch
import os
import sys

from models.SRM_filters import get_filters


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
        out = f.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = f.relu(out)
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


class ImprovedCNN(nn.Module):
    """
    The improved convolutional neural network (CNN) class with residual connections,
    batch normalization, and attention mechanisms
    """
    def __init__(self, feature_dim=400):
        """
        Initialization of all the layers in the network.
        """
        super(ImprovedCNN, self).__init__()
        
        # Initial SRM filter layer (keeping this from original implementation)
        self.conv0 = nn.Conv2d(3, 3, kernel_size=5, stride=1, padding=2)
        nn.init.xavier_uniform_(self.conv0.weight)
        
        self.conv1 = nn.Conv2d(3, 30, kernel_size=5, stride=2, padding=2)
        self.conv1.weight = nn.Parameter(get_filters())
        self.bn1 = nn.BatchNorm2d(30)
        
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        
        # First block of residual layers
        self.res_block1 = nn.Sequential(
            ResidualBlock(30, 32),
            ResidualBlock(32, 32)
        )
        
        self.attention1 = AttentionModule(32)
        
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        
        # Second block of residual layers
        self.res_block2 = nn.Sequential(
            ResidualBlock(32, 64, stride=1),
            ResidualBlock(64, 64)
        )
        
        self.attention2 = AttentionModule(64)
        
        # Final convolutional layers
        self.conv_final = nn.Sequential(
            nn.Conv2d(64, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 16, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU()
        )
        
        # Feature dimension calculation for adaptive pooling
        self.adaptive_pool = nn.AdaptiveAvgPool2d((5, 5))
        
        # Fully connected layer for classification
        self.fc = nn.Linear(16 * 5 * 5, 2)
        
        # Dropout for regularization
        self.drop = nn.Dropout(p=0.5)
        
        # Feature dimension (for compatibility with existing pipeline)
        self.feature_dim = feature_dim

    def forward(self, x):
        """
        The forward step of the network that consumes an image patch and either uses a fully connected layer in the
        training phase with a softmax or just returns the feature map after the final convolutional layer.
        :returns: Either the output of the softmax during training or the feature representation at testing
        """
        # Initial layers
        x = f.relu(self.conv0(x))
        x = f.relu(self.bn1(self.conv1(x)))
        x = self.pool1(x)
        
        # Residual blocks with attention
        x = self.res_block1(x)
        x = self.attention1(x)
        x = self.pool2(x)
        
        x = self.res_block2(x)
        x = self.attention2(x)
        
        # Final convolutions
        x = self.conv_final(x)
        
        # Adaptive pooling to ensure consistent feature size
        x = self.adaptive_pool(x)
        
        # Flatten features
        features = x.view(-1, 16 * 5 * 5)
        
        # In the training phase we also need the fully connected layer with softmax
        if self.training:
            x = self.drop(features)
            x = self.fc(x)
            x = f.log_softmax(x, dim=1)
            return x
        
        return features


# Keep the original CNN class for backward compatibility
class CNN(nn.Module):
    """
    The convolutional neural network (CNN) class
    """
    def __init__(self):
        """
        Initialization of all the layers in the network.
        """
        super(CNN, self).__init__()

        self.conv0 = nn.Conv2d(3, 3, kernel_size=5, stride=1, padding=0)
        nn.init.xavier_uniform_(self.conv0.weight)

        self.conv1 = nn.Conv2d(3, 30, kernel_size=5, stride=2, padding=0)
        self.conv1.weight = nn.Parameter(get_filters())

        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)

        self.conv2 = nn.Conv2d(30, 16, kernel_size=3, stride=1, padding=0)
        nn.init.xavier_uniform_(self.conv2.weight)

        self.conv3 = nn.Conv2d(16, 16, kernel_size=3, stride=1, padding=0)
        nn.init.xavier_uniform_(self.conv3.weight)

        self.conv4 = nn.Conv2d(16, 16, kernel_size=3, stride=1, padding=0)
        nn.init.xavier_uniform_(self.conv4.weight)

        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)

        self.conv5 = nn.Conv2d(16, 16, kernel_size=3, stride=1, padding=0)
        nn.init.xavier_uniform_(self.conv5.weight)

        self.conv6 = nn.Conv2d(16, 16, kernel_size=3, stride=1, padding=0)
        nn.init.xavier_uniform_(self.conv6.weight)

        self.conv7 = nn.Conv2d(16, 16, kernel_size=3, stride=1, padding=0)
        nn.init.xavier_uniform_(self.conv7.weight)

        self.conv8 = nn.Conv2d(16, 16, kernel_size=3, stride=1, padding=0)
        nn.init.xavier_uniform_(self.conv8.weight)

        self.fc = nn.Linear(16 * 5 * 5, 2)

        self.drop1 = nn.Dropout(p=0.5)  # used only for the NC dataset

    def features(self, x):
        """
        Extract features from the model without the final classification layer.
        :param x: Input tensor
        :returns: Feature representation (400-D)
        """
        x = f.relu(self.conv0(x))
        x = f.relu(self.conv1(x))
        lrn = nn.LocalResponseNorm(3)
        x = lrn(x)
        x = self.pool1(x)
        x = f.relu(self.conv2(x))
        x = f.relu(self.conv3(x))
        x = f.relu(self.conv4(x))
        x = f.relu(self.conv5(x))
        x = lrn(x)
        x = self.pool2(x)
        x = f.relu(self.conv6(x))
        x = f.relu(self.conv7(x))
        x = f.relu(self.conv8(x))
        x = x.view(-1, 16 * 5 * 5)
        return x

    def forward(self, x):
        """
        The forward step of the network that consumes an image patch and either uses a fully connected layer in the
        training phase with a softmax or just returns the feature map after the final convolutional layer.
        :returns: Either the output of the softmax during training or the 400-D feature representation at testing
        """
        x = self.features(x)

        # In the training phase we also need the fully connected layer with softmax
        if self.training:
            # x = self.drop1(x) # used only for the NC dataset
            x = f.relu(self.fc(x))
            x = f.softmax(x, dim=1)

        return x
