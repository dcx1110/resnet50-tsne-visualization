#!/usr/bin/env python
"""
Prepare Dataset for Feature Extraction
======================================
This script downloads and prepares datasets for feature extraction.
"""

import os
import argparse
from pathlib import Path
import tarfile
import urllib.request
from torchvision import datasets


def download_cifar10(data_dir):
    """
    Download and extract CIFAR-10 dataset.
    
    Args:
        data_dir (str): Directory to save the dataset
    """
    print("Downloading CIFAR-10 dataset...")
    
    # Create directory
    Path(data_dir).mkdir(parents=True, exist_ok=True)
    
    # Download using torchvision
    train_dataset = datasets.CIFAR10(
        root=data_dir,
        train=True,
        download=True
    )
    
    test_dataset = datasets.CIFAR10(
        root=data_dir,
        train=False,
        download=True
    )
    
    print(f"CIFAR-10 downloaded to {data_dir}")
    print(f"Training samples: {len(train_dataset)}")
    print(f"Test samples: {len(test_dataset)}")


def download_cifar100(data_dir):
    """
    Download and extract CIFAR-100 dataset.
    
    Args:
        data_dir (str): Directory to save the dataset
    """
    print("Downloading CIFAR-100 dataset...")
    
    # Create directory
    Path(data_dir).mkdir(parents=True, exist_ok=True)
    
    # Download using torchvision
    train_dataset = datasets.CIFAR100(
        root=data_dir,
        train=True,
        download=True
    )
    
    test_dataset = datasets.CIFAR100(
        root=data_dir,
        train=False,
        download=True
    )
    
    print(f"CIFAR-100 downloaded to {data_dir}")
    print(f"Training samples: {len(train_dataset)}")
    print(f"Test samples: {len(test_dataset)}")


def organize_dataset(data_dir, dataset_type='cifar10'):
    """
    Organize dataset into folder structure expected by ImageFolder.
    
    Args:
        data_dir (str): Directory containing the dataset
        dataset_type (str): Type of dataset
    """
    print(f"Organizing {dataset_type} dataset...")
    
    # This would typically involve:
    # 1. Creating train/val/test directories
    # 2. Creating subdirectories for each class
    # 3. Moving images to appropriate directories
    
    # For CIFAR datasets, torchvision handles this internally
    # For custom datasets, you would implement the organization logic here
    
    print("Dataset organization complete!")


def verify_dataset(data_dir):
    """
    Verify that the dataset is properly structured.
    
    Args:
        data_dir (str): Directory containing the dataset
    """
    data_path = Path(data_dir)
    
    if not data_path.exists():
        print(f"Error: {data_dir} does not exist!")
        return False
    
    # Check for expected structure
    expected_dirs = ['train', 'val', 'test']
    found_dirs = [d.name for d in data_path.iterdir() if d.is_dir()]
    
    print(f"Found directories: {found_dirs}")
    
    # Count images in each directory
    for dir_name in found_dirs:
        dir_path = data_path / dir_name
        if dir_path.is_dir():
            # Count subdirectories (classes)
            classes = [d for d in dir_path.iterdir() if d.is_dir()]
            total_images = 0
            
            for class_dir in classes:
                images = list(class_dir.glob('*.jpg')) + \
                        list(class_dir.glob('*.png')) + \
                        list(class_dir.glob('*.jpeg'))
                total_images += len(images)
            
            print(f"  {dir_name}: {len(classes)} classes, {total_images} images")
    
    return True


def main():
    parser = argparse.ArgumentParser(description='Prepare dataset for feature extraction')
    parser.add_argument(
        '--dataset',
        type=str,
        default='cifar10',
        choices=['cifar10', 'cifar100', 'custom'],
        help='Dataset to download and prepare'
    )
    parser.add_argument(
        '--data-dir',
        type=str,
        default='data',
        help='Directory to save the dataset'
    )
    parser.add_argument(
        '--verify-only',
        action='store_true',
        help='Only verify existing dataset structure'
    )
    
    args = parser.parse_args()
    
    if args.verify_only:
        # Only verify the dataset
        print(f"Verifying dataset in {args.data_dir}...")
        if verify_dataset(args.data_dir):
            print("Dataset verification successful!")
        else:
            print("Dataset verification failed!")
    else:
        # Download and prepare dataset
        data_path = Path(args.data_dir) / args.dataset
        
        if args.dataset == 'cifar10':
            download_cifar10(str(data_path))
        elif args.dataset == 'cifar100':
            download_cifar100(str(data_path))
        elif args.dataset == 'custom':
            print("For custom datasets, please manually organize your data into:")
            print(f"  {data_path}/train/class_name/images...")
            print(f"  {data_path}/val/class_name/images...")
            print(f"  {data_path}/test/class_name/images...")
        
        # Organize the dataset
        if args.dataset in ['cifar10', 'cifar100']:
            organize_dataset(str(data_path), args.dataset)
        
        print("\nDataset preparation completed!")
        print(f"Dataset location: {data_path}")


if __name__ == '__main__':
    main()