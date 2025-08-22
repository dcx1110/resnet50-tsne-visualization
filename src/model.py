"""
ResNet50 Model Definition
========================
This module contains the ResNet50 model architecture definition.
"""

import torch
import torch.nn as nn
from torchvision import models


def resnet50(num_classes=10, pretrained=False):
    """
    Create a ResNet50 model with custom number of output classes.
    
    Args:
        num_classes (int): Number of output classes
        pretrained (bool): Whether to load ImageNet pretrained weights
        
    Returns:
        torch.nn.Module: ResNet50 model
    """
    if pretrained:
        # Load pretrained model
        model = models.resnet50(pretrained=True)
        # Replace the final fully connected layer
        num_features = model.fc.in_features
        model.fc = nn.Linear(num_features, num_classes)
    else:
        # Create model from scratch
        model = models.resnet50(pretrained=False, num_classes=num_classes)
    
    return model


class ResNet50FeatureExtractor(nn.Module):
    """
    ResNet50 model modified to extract features from intermediate layers.
    """
    
    def __init__(self, original_model, layer_name='avgpool'):
        """
        Initialize the feature extractor.
        
        Args:
            original_model: Pre-trained ResNet50 model
            layer_name (str): Name of the layer to extract features from
                            Options: 'layer1', 'layer2', 'layer3', 'layer4', 'avgpool'
        """
        super(ResNet50FeatureExtractor, self).__init__()
        
        self.layer_name = layer_name
        
        # Copy all layers up to the specified layer
        if layer_name == 'layer1':
            self.features = nn.Sequential(
                original_model.conv1,
                original_model.bn1,
                original_model.relu,
                original_model.maxpool,
                original_model.layer1
            )
        elif layer_name == 'layer2':
            self.features = nn.Sequential(
                original_model.conv1,
                original_model.bn1,
                original_model.relu,
                original_model.maxpool,
                original_model.layer1,
                original_model.layer2
            )
        elif layer_name == 'layer3':
            self.features = nn.Sequential(
                original_model.conv1,
                original_model.bn1,
                original_model.relu,
                original_model.maxpool,
                original_model.layer1,
                original_model.layer2,
                original_model.layer3
            )
        elif layer_name == 'layer4':
            self.features = nn.Sequential(
                original_model.conv1,
                original_model.bn1,
                original_model.relu,
                original_model.maxpool,
                original_model.layer1,
                original_model.layer2,
                original_model.layer3,
                original_model.layer4
            )
        elif layer_name == 'avgpool':
            self.features = nn.Sequential(
                original_model.conv1,
                original_model.bn1,
                original_model.relu,
                original_model.maxpool,
                original_model.layer1,
                original_model.layer2,
                original_model.layer3,
                original_model.layer4,
                original_model.avgpool,
                nn.Flatten()
            )
        else:
            raise ValueError(f"Unknown layer name: {layer_name}")
    
    def forward(self, x):
        """Forward pass to extract features."""
        return self.features(x)