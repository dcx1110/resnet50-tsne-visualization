#!/usr/bin/env python
"""
Download Pre-trained ResNet50 Weights
=====================================
This script downloads pre-trained ResNet50 weights for different datasets.
"""

import os
import argparse
import urllib.request
from pathlib import Path
import torch
from torchvision import models


def download_imagenet_pretrained(save_path):
    """
    Download ImageNet pre-trained ResNet50 weights.
    
    Args:
        save_path (str): Path to save the model
    """
    print("Downloading ImageNet pre-trained ResNet50...")
    model = models.resnet50(pretrained=True)
    
    # Save the state dict
    torch.save(model.state_dict(), save_path)
    print(f"Model saved to {save_path}")


def download_from_url(url, save_path):
    """
    Download model from a specific URL.
    
    Args:
        url (str): URL to download from
        save_path (str): Path to save the model
    """
    print(f"Downloading from {url}...")
    
    def download_progress(block_num, block_size, total_size):
        downloaded = block_num * block_size
        percent = min(downloaded * 100 / total_size, 100)
        print(f"Download progress: {percent:.1f}%", end='\r')
    
    urllib.request.urlretrieve(url, save_path, reporthook=download_progress)
    print(f"\nModel saved to {save_path}")


def main():
    parser = argparse.ArgumentParser(description='Download pre-trained ResNet50 weights')
    parser.add_argument(
        '--source',
        type=str,
        default='imagenet',
        choices=['imagenet', 'url'],
        help='Source of pre-trained weights'
    )
    parser.add_argument(
        '--url',
        type=str,
        default=None,
        help='URL to download weights from (if source=url)'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='models/resnet50_pretrained.pth',
        help='Output path for the model'
    )
    
    args = parser.parse_args()
    
    # Create output directory
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Download based on source
    if args.source == 'imagenet':
        download_imagenet_pretrained(str(output_path))
    elif args.source == 'url':
        if not args.url:
            print("Error: URL must be provided when source=url")
            return
        download_from_url(args.url, str(output_path))
    
    print("\nDownload completed successfully!")
    
    # Verify the downloaded file
    if output_path.exists():
        file_size = output_path.stat().st_size / (1024 * 1024)  # Convert to MB
        print(f"File size: {file_size:.2f} MB")
        
        # Try to load the model to verify integrity
        try:
            state_dict = torch.load(str(output_path), map_location='cpu')
            print(f"Model verified: {len(state_dict)} parameters loaded")
        except Exception as e:
            print(f"Warning: Could not verify model: {e}")


if __name__ == '__main__':
    main()