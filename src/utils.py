"""
Utility Functions
================
Helper functions for configuration, data handling, and visualization.
"""

import os
import json
import yaml
import torch
import numpy as np
from typing import Dict, Any, Tuple, Optional


def load_config(config_path: str) -> Dict[str, Any]:
    """
    Load configuration from a YAML file.
    
    Args:
        config_path (str): Path to the configuration file
        
    Returns:
        dict: Configuration dictionary
    """
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def save_features(features: np.ndarray, labels: np.ndarray, save_path: str):
    """
    Save extracted features and labels to disk.
    
    Args:
        features (np.ndarray): Feature array
        labels (np.ndarray): Label array
        save_path (str): Path to save the features
    """
    save_dict = {
        'features': features,
        'labels': labels
    }
    
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    # Save as .npz file
    np.savez_compressed(save_path, **save_dict)
    print(f"Features saved to {save_path}")


def load_features(load_path: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load previously saved features and labels.
    
    Args:
        load_path (str): Path to the saved features
        
    Returns:
        tuple: (features, labels) arrays
    """
    data = np.load(load_path)
    features = data['features']
    labels = data['labels']
    print(f"Features loaded from {load_path}")
    return features, labels


def create_class_mapping(class_to_idx: Dict[str, int], save_path: Optional[str] = None) -> Dict[int, str]:
    """
    Create and optionally save a mapping from class indices to class names.
    
    Args:
        class_to_idx (dict): Dictionary mapping class names to indices
        save_path (str, optional): Path to save the mapping as JSON
        
    Returns:
        dict: Dictionary mapping indices to class names
    """
    idx_to_class = {v: k for k, v in class_to_idx.items()}
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        with open(save_path, 'w') as f:
            json.dump(idx_to_class, f, indent=4)
        print(f"Class mapping saved to {save_path}")
    
    return idx_to_class


def set_random_seed(seed: int = 42):
    """
    Set random seed for reproducibility.
    
    Args:
        seed (int): Random seed value
    """
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def get_device(device: Optional[str] = None) -> torch.device:
    """
    Get the appropriate device for computation.
    
    Args:
        device (str, optional): Specified device ('cuda', 'cpu', or None for auto)
        
    Returns:
        torch.device: Device object
    """
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    if device == 'cuda' and not torch.cuda.is_available():
        print("CUDA not available, falling back to CPU")
        device = 'cpu'
    
    return torch.device(device)


def calculate_dataset_stats(dataloader) -> Tuple[list, list]:
    """
    Calculate mean and standard deviation of a dataset.
    
    Args:
        dataloader: PyTorch DataLoader
        
    Returns:
        tuple: (mean, std) for each channel
    """
    mean = torch.zeros(3)
    std = torch.zeros(3)
    total_samples = 0
    
    for data, _ in dataloader:
        batch_samples = data.size(0)
        data = data.view(batch_samples, data.size(1), -1)
        mean += data.mean(2).sum(0)
        std += data.std(2).sum(0)
        total_samples += batch_samples
    
    mean /= total_samples
    std /= total_samples
    
    return mean.tolist(), std.tolist()


def print_model_summary(model: torch.nn.Module):
    """
    Print a summary of the model architecture.
    
    Args:
        model (torch.nn.Module): PyTorch model
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"{'='*50}")
    print(f"Model Summary")
    print(f"{'='*50}")
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print(f"Non-trainable parameters: {total_params - trainable_params:,}")
    print(f"{'='*50}")