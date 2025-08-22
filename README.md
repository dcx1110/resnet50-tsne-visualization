# ResNet50 Feature Visualization with t-SNE
In this work, we take the case of ResNet-50 on CIFAR-10 as an example. Moreover, based on this repository, one can perform t-SNE visualizations across different DNN architectures and datasets to further analyze the feature representations.


Extract features from pre-trained ResNet50 and visualize using t-SNE dimensionality reduction.

## Features
- Feature extraction from ResNet50
- t-SNE visualization
- Support for CIFAR-10 dataset
- Customizable parameters


resnet50-tsne-visualization/
│
├── README.md                 # 项目说明
├── requirements.txt          # 依赖包列表
├── LICENSE                   # 许可证
├── .gitignore               # 忽略文件配置
│
├── src/                     # 源代码目录
│   ├── feature_extractor.py # 主要代码
│   ├── model.py            # ResNet50模型定义
│   └── utils.py            # 工具函数
│
├── models/                  # 预训练模型
│   └── .gitkeep            # 空文件，保持文件夹
│
├── data/                    # 数据集目录
│   └── .gitkeep
│
├── results/                 # 结果图片
│   └── tsne_visualization.jpg
│
└── notebooks/              # Jupyter notebooks（可选）
    └── demo.ipynb


## Installation
```bash
git clone https://github.com/yourusername/resnet50-tsne-visualization.git
cd resnet50-tsne-visualization
pip install -r requirements.txt
