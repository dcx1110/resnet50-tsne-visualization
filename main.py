#!/usr/bin/env python
"""
Main Script for ResNet50 Feature Extraction and t-SNE Visualization
===================================================================
Run this script to extract features and create t-SNE visualizations.
"""

import argparse
import os
import sys
from pathlib import Path

# Add src directory to path
sys.path.append(str(Path(__file__).parent))

from src.feature_extractor import FeatureExtractor
from src.utils import load_config, set_random_seed


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Extract ResNet50 features and visualize with t-SNE'
    )
    
    parser.add_argument(
        '--config',
        type=str,
        default='configs/default_config.yaml',
        help='Path to configuration file'
    )
    
    parser.add_argument(
        '--model-path',
        type=str,
        default=None,
        help='Path to pre-trained model (overrides config)'
    )
    
    parser.add_argument(
        '--data-path',
        type=str,
        default=None,
        help='Path to dataset (overrides config)'
    )
    
    parser.add_argument(
        '--output-path',
        type=str,
        default=None,
        help='Path to save visualization (overrides config)'
    )
    
    parser.add_argument(
        '--batch-size',
        type=int,
        default=None,
        help='Batch size for processing (overrides config)'
    )
    
    parser.add_argument(
        '--device',
        type=str,
        choices=['cuda', 'cpu', 'auto'],
        default=None,
        help='Device to use for computation (overrides config)'
    )
    
    parser.add_argument(
        '--seed',
        type=int,
        default=None,
        help='Random seed for reproducibility (overrides config)'
    )
    
    parser.add_argument(
        '--no-visualization',
        action='store_true',
        help='Skip visualization, only extract features'
    )
    
    return parser.parse_args()


def main():
    """Main execution function."""
    
    # Parse arguments
    args = parse_arguments()
    
    # Load configuration
    if os.path.exists(args.config):
        config = load_config(args.config)
        print(f"Configuration loaded from {args.config}")
    else:
        print(f"Config file {args.config} not found, using defaults")
        config = {}
    
    # Override config with command line arguments
    if args.model_path:
        if 'model' not in config:
            config['model'] = {}
        config['model']['pretrained_path'] = args.model_path
    
    if args.data_path:
        if 'data' not in config:
            config['data'] = {}
        config['data']['data_root'] = args.data_path
    
    if args.batch_size:
        if 'data' not in config:
            config['data'] = {}
        config['data']['batch_size'] = args.batch_size
    
    if args.device:
        if 'system' not in config:
            config['system'] = {}
        config['system']['device'] = args.device
    
    if args.seed:
        if 'system' not in config:
            config['system'] = {}
        config['system']['seed'] = args.seed
    
    # Set random seed for reproducibility
    seed = config.get('system', {}).get('seed', 42)
    set_random_seed(seed)
    print(f"Random seed set to {seed}")
    
    # Extract configuration parameters
    model_config = config.get('model', {})
    data_config = config.get('data', {})
    system_config = config.get('system', {})
    output_config = config.get('output', {})
    
    # Initialize feature extractor
    print("\nInitializing Feature Extractor...")
    extractor = FeatureExtractor(
        model_path=model_config.get('pretrained_path', './models/resNet50.pth'),
        data_path=data_config.get('data_root', './data/cifar10'),
        batch_size=data_config.get('batch_size', 128),
        num_classes=model_config.get('num_classes', 10),
        device=system_config.get('device', None)
    )
    
    # Set output path
    if args.output_path:
        save_path = args.output_path
    else:
        save_path = output_config.get('visualization_path', './results/tsne_visualization.jpg')
    
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    # Run the pipeline
    print("\nStarting feature extraction and visualization pipeline...")
    print("="*60)
    
    try:
        if args.no_visualization:
            # Only extract features
            print("Extracting features only (no visualization)...")
            dataloader = extractor.load_data(split=data_config.get('split', 'val'))
            model = extractor.load_model()
            features, labels = extractor.extract_features(model, dataloader)
            
            # Save features if configured
            if output_config.get('save_features', True):
                from src.utils import save_features
                features_path = output_config.get('features_path', './results/features.npz')
                save_features(features, labels, features_path)
                print(f"\nFeatures saved to {features_path}")
        else:
            # Run complete pipeline
            tsne_features, labels = extractor.run(save_path=save_path)
            print(f"\nVisualization saved to {save_path}")
            
            # Save features if configured
            if output_config.get('save_features', True):
                from src.utils import save_features
                features_path = output_config.get('features_path', './results/features.npz')
                save_features(tsne_features, labels, features_path)
                print(f"t-SNE features saved to {features_path}")
        
        print("\n" + "="*60)
        print("Pipeline completed successfully!")
        print("="*60)
        
    except Exception as e:
        print(f"\nError occurred: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()