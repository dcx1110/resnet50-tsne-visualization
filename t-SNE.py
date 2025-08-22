"""
ResNet50 Feature Visualization using t-SNE
==========================================
This script extracts features from a pre-trained ResNet50 model and visualizes them
using t-SNE dimensionality reduction technique.

Author: [Chenxun Deng]
License: MIT
"""

import os
import json
import torch
import numpy as np
from PIL import Image, ImageFile
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.manifold import TSNE
from matplotlib import cm
from torchvision import transforms, datasets

# Import your custom model
from model import resnet50

# Allow loading of truncated images
ImageFile.LOAD_TRUNCATED_IMAGES = True


class FeatureExtractor:
    """
    A class to extract features from a pre-trained ResNet50 model
    and visualize them using t-SNE.
    """
    
    def __init__(self, model_path='./resNet50(if=100).pth', 
                 data_path='../../data_set/cifar10',
                 batch_size=128,
                 num_classes=10,
                 device=None):
        """
        Initialize the feature extractor.
        
        Args:
            model_path (str): Path to the pre-trained model weights
            data_path (str): Path to the dataset
            batch_size (int): Batch size for data loading
            num_classes (int): Number of classes in the dataset
            device (str): Device to run the model on
        """
        self.model_path = model_path
        self.batch_size = batch_size
        self.num_classes = num_classes
        
        # Set device
        if device is None:
            self.device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
        print(f"Using {self.device} device.")
        
        # Set data path
        self.data_root = os.path.abspath(os.path.join(os.getcwd(), "../.."))
        self.image_path = os.path.join(self.data_root, "data_set", "cifar10")
        
        # Data transformations
        self.data_transform = self._get_transforms()
        
    def _get_transforms(self):
        """
        Define data transformations for training and validation.
        
        Returns:
            dict: Dictionary containing transforms for 'train' and 'val'
        """
        return {
            "train": transforms.Compose([
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], 
                                   [0.229, 0.224, 0.225])
            ]),
            "val": transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], 
                                   [0.229, 0.224, 0.225])
            ])
        }
    
    def load_data(self, split='val'):
        """
        Load and prepare the dataset.
        
        Args:
            split (str): Dataset split to load ('train' or 'val')
            
        Returns:
            DataLoader: PyTorch DataLoader object
        """
        # Check if path exists
        assert os.path.exists(self.image_path), \
            f"{self.image_path} path does not exist."
        
        # Load dataset
        dataset = datasets.ImageFolder(
            root=os.path.join(self.image_path, split),
            transform=self.data_transform["train"]
        )
        
        print(f"Loaded {len(dataset)} images from {split} set.")
        
        # Save class indices
        self._save_class_indices(dataset.class_to_idx)
        
        # Create dataloader
        nw = min([os.cpu_count(), self.batch_size if self.batch_size > 1 else 0, 8])
        print(f'Using {nw} dataloader workers per process')
        
        dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=self.batch_size, 
            shuffle=True,
            num_workers=nw
        )
        
        return dataloader
    
    def _save_class_indices(self, class_to_idx):
        """
        Save class indices to a JSON file.
        
        Args:
            class_to_idx (dict): Dictionary mapping class names to indices
        """
        cla_dict = dict((val, key) for key, val in class_to_idx.items())
        json_str = json.dumps(cla_dict, indent=4)
        
        with open('class_indices_cifar10.json', 'w') as json_file:
            json_file.write(json_str)
        print("Class indices saved to 'class_indices_cifar10.json'")
    
    def load_model(self):
        """
        Load the pre-trained ResNet50 model.
        
        Returns:
            torch.nn.Module: Loaded model
        """
        # Initialize model
        model = resnet50(num_classes=self.num_classes)
        
        # Load pre-trained weights
        if os.path.exists(self.model_path):
            model.load_state_dict(torch.load(self.model_path))
            print(f"Model loaded from {self.model_path}")
        else:
            raise FileNotFoundError(f"Model file not found: {self.model_path}")
        
        model.to(self.device)
        model.eval()
        
        return model
    
    def extract_features(self, model, dataloader):
        """
        Extract features from the model for all data in the dataloader.
        
        Args:
            model (torch.nn.Module): The model to extract features from
            dataloader (DataLoader): DataLoader containing the data
            
        Returns:
            tuple: (features, labels) as numpy arrays
        """
        all_features = []
        all_labels = []
        
        with torch.no_grad():
            for data, labels in tqdm(dataloader, desc="Extracting features"):
                inputs = data.to(self.device)
                features = model(inputs)
                
                all_features.append(features.cpu())
                all_labels.append(labels)
        
        # Concatenate all batches
        features = torch.cat(all_features, dim=0).numpy()
        labels = torch.cat(all_labels, dim=0).numpy()
        
        print(f"Extracted features shape: {features.shape}")
        print(f"Labels shape: {labels.shape}")
        
        return features, labels
    
    def apply_tsne(self, features, n_components=2, random_state=33):
        """
        Apply t-SNE dimensionality reduction to features.
        
        Args:
            features (np.ndarray): High-dimensional features
            n_components (int): Number of dimensions to reduce to
            random_state (int): Random seed for reproducibility
            
        Returns:
            np.ndarray: Reduced features
        """
        print("Applying t-SNE...")
        tsne = TSNE(n_components=n_components, random_state=random_state)
        tsne_features = tsne.fit_transform(features)
        
        # Normalize to [0, 1]
        x_min, x_max = np.min(tsne_features, 0), np.max(tsne_features, 0)
        tsne_norm = (tsne_features - x_min) / (x_max - x_min)
        
        return tsne_norm
    
    def visualize_tsne(self, tsne_features, labels, save_path='tsne_visualization.jpg'):
        """
        Visualize t-SNE features.
        
        Args:
            tsne_features (np.ndarray): 2D t-SNE features
            labels (np.ndarray): Class labels
            save_path (str): Path to save the visualization
        """
        # Set up the plot
        plt.figure(figsize=(20, 20))
        
        # Remove axes and borders
        plt.xticks([])
        plt.yticks([])
        plt.gca().spines['right'].set_visible(False)
        plt.gca().spines['top'].set_visible(False)
        plt.gca().spines['left'].set_visible(False)
        plt.gca().spines['bottom'].set_visible(False)
        
        # Define colors for each class
        class_num = len(np.unique(labels))
        colors = cm.jet(np.linspace(0, 1, class_num))
        
        # Plot each point
        for i in range(tsne_features.shape[0]):
            if labels[i] < class_num:
                plt.text(tsne_features[i, 0], tsne_features[i, 1], '.', 
                        color=colors[labels[i]],
                        fontdict={'weight': 'bold', 'size': 20})
            else:
                print(f"Warning: Label {labels[i]} at index {i} is out of range "
                      f"(max: {class_num - 1})")
        
        # Save and show
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Visualization saved to {save_path}")
        plt.show()
    
    def run(self, save_path='tsne_visualization.jpg'):
        """
        Run the complete feature extraction and visualization pipeline.
        
        Args:
            save_path (str): Path to save the visualization
        """
        # Load data
        dataloader = self.load_data(split='val')
        
        # Load model
        model = self.load_model()
        
        # Extract features
        features, labels = self.extract_features(model, dataloader)
        
        # Apply t-SNE
        tsne_features = self.apply_tsne(features)
        
        # Visualize
        self.visualize_tsne(tsne_features, labels, save_path)
        
        return tsne_features, labels


def main():
    """
    Main function to run the feature extraction and visualization.
    """
    # Configuration
    config = {
        'model_path': './resNet50(if=100).pth',
        'batch_size': 128,
        'num_classes': 10,
        'device': None  # Will auto-select GPU if available
    }
    
    # Create feature extractor
    extractor = FeatureExtractor(**config)
    
    # Run the pipeline
    tsne_features, labels = extractor.run(save_path='if=100.jpg')
    
    return tsne_features, labels


if __name__ == '__main__':
    main()