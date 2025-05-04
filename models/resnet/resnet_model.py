"""
ResNet implementation for image forgery detection.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from models.SRM_filters import get_filters

class BasicBlock(nn.Module):
    """
    Basic residual block with two 3x3 convolutions and a skip connection
    """
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out

class Bottleneck(nn.Module):
    """
    Bottleneck block with 1x1, 3x3, 1x1 convolutions
    """
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion * planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion * planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out

class AttentionModule(nn.Module):
    """
    Channel and spatial attention module to focus on important features
    """
    def __init__(self, in_channels, reduction=16):
        super(AttentionModule, self).__init__()
        # Channel attention
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc1 = nn.Conv2d(in_channels, in_channels // reduction, kernel_size=1, bias=False)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Conv2d(in_channels // reduction, in_channels, kernel_size=1, bias=False)
        self.sigmoid = nn.Sigmoid()
        
        # Spatial attention
        self.conv = nn.Conv2d(2, 1, kernel_size=7, padding=3, bias=False)

    def forward(self, x):
        # Channel attention
        avg_out = self.fc2(self.relu(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu(self.fc1(self.max_pool(x))))
        channel_out = self.sigmoid(avg_out + max_out)
        x = x * channel_out
        
        # Spatial attention
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        spatial_out = torch.cat([avg_out, max_out], dim=1)
        spatial_out = self.sigmoid(self.conv(spatial_out))
        
        return x * spatial_out

class ResNetForensics(nn.Module):
    """
    ResNet model with SRM filters for image forgery detection
    """
    def __init__(self, block, num_blocks, num_classes=2, srm_filters=True):
        super(ResNetForensics, self).__init__()
        self.in_planes = 64
        self.srm_filters = srm_filters
        
        # SRM filter layer (fixed weights for noise analysis)
        if self.srm_filters:
            # First get the SRM filters (30x3x5x5)
            srm_weights = torch.tensor(get_filters()).float()
            
            # Create a separate convolutional layer for SRM filters
            # This will extract noise features from the input image
            self.srm_conv = nn.Conv2d(3, 30, kernel_size=5, padding=2, bias=False)
            self.srm_conv.weight = nn.Parameter(srm_weights, requires_grad=False)
            
            # Add a 1x1 convolution to reduce from 30 channels back to 3
            self.srm_reduce = nn.Conv2d(30, 3, kernel_size=1, bias=False)
        
        # Initial convolutional layer
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        
        # Residual blocks
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        
        # Attention module
        self.attention = AttentionModule(512 * block.expansion)
        
        # Classification layer
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)
        
        # Initialize weights
        self._initialize_weights()

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d) and m is not self.srm_conv:
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        # Apply SRM filters if enabled
        if self.srm_filters:
            # Extract noise features with SRM filters
            noise_features = self.srm_conv(x)
            # Reduce back to 3 channels
            noise_features = self.srm_reduce(noise_features)
            # Add noise features to original image
            x = x + noise_features
        
        # Initial convolutional layer
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.max_pool2d(x, kernel_size=3, stride=2, padding=1)
        
        # Residual blocks
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        
        # Apply attention
        x = self.attention(x)
        
        # Classification
        x = self.avg_pool(x)
        x_flat = x.view(x.size(0), -1)
        x = self.fc(x_flat)
        
        return x
    
    def features(self, x):
        """
        Extract features from the model for feature fusion
        """
        # Apply SRM filters if enabled
        if self.srm_filters:
            # Extract noise features with SRM filters
            noise_features = self.srm_conv(x)
            # Reduce back to 3 channels
            noise_features = self.srm_reduce(noise_features)
            # Add noise features to original image
            x = x + noise_features
        
        # Initial convolutional layer
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.max_pool2d(x, kernel_size=3, stride=2, padding=1)
        
        # Residual blocks
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        
        # Apply attention
        x = self.attention(x)
        
        # Global average pooling
        x = self.avg_pool(x)
        x_flat = x.view(x.size(0), -1)
        
        return x_flat

# Model configurations
def ResNet18Forensics(num_classes=2, srm_filters=True):
    return ResNetForensics(BasicBlock, [2, 2, 2, 2], num_classes, srm_filters)

def ResNet34Forensics(num_classes=2, srm_filters=True):
    return ResNetForensics(BasicBlock, [3, 4, 6, 3], num_classes, srm_filters)

def ResNet50Forensics(num_classes=2, srm_filters=True):
    return ResNetForensics(Bottleneck, [3, 4, 6, 3], num_classes, srm_filters)

def ResNet101Forensics(num_classes=2, srm_filters=True):
    return ResNetForensics(Bottleneck, [3, 4, 23, 3], num_classes, srm_filters)

def ResNet152Forensics(num_classes=2, srm_filters=True):
    return ResNetForensics(Bottleneck, [3, 8, 36, 3], num_classes, srm_filters) 