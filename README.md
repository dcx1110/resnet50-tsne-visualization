# ResNet50 Feature Visualization with t-SNE

A PyTorch implementation for extracting deep features from pre-trained ResNet50 and visualizing them using t-SNE dimensionality reduction technique. This project is particularly useful for understanding learned representations and analyzing feature distributions across different classes.

## 📁 Project Structure

```
resnet50-tsne-visualization/
│
├── README.md                    # Project documentation (this file)
├── requirements.txt             # Python dependencies
├── LICENSE                      # MIT License
├── .gitignore                  # Git ignore rules
├── setup.py                    # Package setup file
│
├── src/                        # Source code directory
│   ├── __init__.py            # Package initializer
│   ├── feature_extractor.py   # Main feature extraction class
│   ├── model.py               # ResNet50 model definition
│   └── utils.py               # Utility functions
│
├── configs/                    # Configuration files
│   └── default_config.yaml    # Default configuration
│
├── models/                     # Pre-trained model weights
│   ├── .gitkeep               # Placeholder for directory
│   └── README.md              # Instructions for model files
│
├── data/                       # Dataset directory
│   ├── .gitkeep               # Placeholder for directory
│   └── README.md              # Data preparation instructions
│
├── results/                    # Output visualizations
│   ├── .gitkeep               # Placeholder for directory
│   └── sample_results/        # Sample visualization results
│       └── tsne_cifar10.jpg   # Example t-SNE visualization
│
├── notebooks/                  # Jupyter notebooks
│   ├── demo.ipynb             # Interactive demo
│   └── analysis.ipynb         # Detailed analysis notebook
│
└── scripts/                    # Utility scripts
    ├── download_pretrained.py  # Download pre-trained weights
    └── prepare_data.py         # Data preparation script
```

## ✨ Features

- 🚀 **Feature Extraction**: Extract high-dimensional features from any layer of ResNet50
- 📊 **t-SNE Visualization**: Reduce dimensions and visualize feature distributions
- 🎨 **Customizable Plots**: Beautiful, publication-ready visualizations with color-coded classes
- 📦 **Modular Design**: Clean, reusable code structure with OOP design
- 🔧 **Configurable**: Easy to adjust parameters via configuration files
- 📈 **Batch Processing**: Efficient processing of large datasets
- 💾 **Checkpoint Support**: Save and load extracted features for faster experimentation

## 🚀 Quick Start

### Installation

1. **Clone the repository**
```bash
git clone https://github.com/yourusername/resnet50-tsne-visualization.git
cd resnet50-tsne-visualization
```

2. **Create a virtual environment** (recommended)
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

4. **Download pre-trained weights** (optional)
```bash
python scripts/download_pretrained.py
```

### Basic Usage

```python
from src.feature_extractor import FeatureExtractor

# Initialize the feature extractor
extractor = FeatureExtractor(
    model_path='./models/resNet50.pth',  # Path to pre-trained weights
    data_path='./data/cifar10',          # Path to your dataset
    batch_size=128,                       # Batch size for processing
    num_classes=10,                       # Number of classes
    device='cuda'                         # Use 'cpu' if GPU not available
)

# Run the complete pipeline
tsne_features, labels = extractor.run(
    save_path='./results/my_visualization.jpg'
)

# The extracted features and labels are also returned for further analysis
print(f"t-SNE features shape: {tsne_features.shape}")
print(f"Labels shape: {labels.shape}")
```

### Advanced Usage

```python
from src.feature_extractor import FeatureExtractor

# Create extractor with custom configuration
extractor = FeatureExtractor(
    model_path='./models/resNet50_custom.pth',
    batch_size=256,
    num_classes=100  # For CIFAR-100 or custom dataset
)

# Step-by-step execution for more control
dataloader = extractor.load_data(split='val')
model = extractor.load_model()

# Extract features from a specific layer (optional)
features, labels = extractor.extract_features(model, dataloader)

# Apply t-SNE with custom parameters
tsne_features = extractor.apply_tsne(
    features, 
    n_components=2,
    perplexity=30,
    random_state=42
)

# Create custom visualization
extractor.visualize_tsne(
    tsne_features, 
    labels,
    save_path='./results/custom_tsne.jpg',
    figsize=(15, 15),
    point_size=10
)
```

## 📊 Results

### Sample t-SNE Visualization
The following image shows a t-SNE visualization of ResNet50 features extracted from CIFAR-10 dataset:

![t-SNE Visualization](results/sample_results/tsne_cifar10.jpg)

Each color represents a different class, and the spatial proximity indicates feature similarity learned by the model.

## 🛠️ Configuration

You can modify the default configuration in `configs/default_config.yaml`:

```yaml
model:
  architecture: resnet50
  num_classes: 10
  pretrained_path: ./models/resNet50.pth

data:
  dataset: cifar10
  data_root: ./data
  batch_size: 128
  num_workers: 4

tsne:
  n_components: 2
  perplexity: 30
  random_state: 42
  n_iter: 1000

visualization:
  figsize: [20, 20]
  dpi: 300
  colormap: jet
```

## 📝 Code Explanation

### Main Components

1. **`FeatureExtractor` Class**: The core class that handles the entire pipeline
   - `load_data()`: Loads and preprocesses the dataset
   - `load_model()`: Initializes and loads the pre-trained ResNet50
   - `extract_features()`: Extracts features from the model
   - `apply_tsne()`: Applies t-SNE dimensionality reduction
   - `visualize_tsne()`: Creates and saves the visualization

2. **Data Processing**: 
   - Automatic data augmentation for training
   - Normalization using ImageNet statistics
   - Efficient batch processing with DataLoader

3. **Feature Extraction**:
   - Extracts features from the final layer before classification
   - Supports extraction from intermediate layers
   - Handles GPU/CPU computation automatically

4. **Visualization**:
   - Color-coded points for different classes
   - Customizable plot parameters
   - High-resolution output for publications

## 📋 Requirements

- Python 3.7+
- PyTorch 1.9.0+
- torchvision 0.10.0+
- NumPy 1.19.0+
- Matplotlib 3.3.0+
- scikit-learn 0.24.0+
- tqdm 4.62.0+
- Pillow 8.0.0+
- PyYAML 5.4.0+ (for configuration files)

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request. For major changes, please open an issue first to discuss what you would like to change.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- ResNet50 architecture from [Deep Residual Learning for Image Recognition](https://arxiv.org/abs/1512.03385)
- t-SNE algorithm from [Visualizing Data using t-SNE](https://www.jmlr.org/papers/v9/vandermaaten08a.html)
- CIFAR-10 dataset from [Learning Multiple Layers of Features from Tiny Images](https://www.cs.toronto.edu/~kriz/cifar.html)

## 🌟 Star History

If you find this project useful, please consider giving it a star ⭐️

---
<p align="center">Made with ❤️ by [Your Name]</p>
