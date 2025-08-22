"""
ResNet50 Feature Visualization Package
======================================
A package for extracting and visualizing features from ResNet50 using t-SNE.
"""

from .feature_extractor import FeatureExtractor
from .utils import load_config, save_features, load_features

__version__ = "1.0.0"
__author__ = "Your Name"
__email__ = "your.email@example.com"

__all__ = [
    "FeatureExtractor",
    "load_config",
    "save_features",
    "load_features"
]