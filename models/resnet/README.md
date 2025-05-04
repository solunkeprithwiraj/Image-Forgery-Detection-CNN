# ResNet for Image Forgery Detection

This directory contains the implementation of ResNet models for image forgery detection.

## Overview

The ResNet (Residual Network) architecture is a powerful deep learning model that has shown excellent performance in various computer vision tasks. In this implementation, we've adapted ResNet for image forgery detection by incorporating:

- SRM (Spatial Rich Model) filters for capturing noise residuals
- Attention mechanisms to focus on informative features
- Residual blocks to enable deep network training

## Model Variants

The following ResNet variants are implemented:

- **ResNet18Forensics**: Lightweight model with 18 layers
- **ResNet34Forensics**: Medium-sized model with 34 layers
- **ResNet50Forensics**: Deeper model with 50 layers and bottleneck blocks
- **ResNet101Forensics**: Very deep model with 101 layers
- **ResNet152Forensics**: Extremely deep model with 152 layers

## Usage

### Data Preparation

Before training or evaluating the models, you need to prepare your dataset:

#### Using CASIA2 Dataset

1. Make sure your CASIA2 dataset is organized in the following structure:
```
data/casia2/
├── authentic/
│   ├── image1.jpg
│   ├── image2.jpg
│   └── ...
└── tampered/
    ├── image1.jpg
    ├── image2.jpg
    └── ...
```

2. Preprocess the CASIA2 dataset using the provided script:
```
# On Windows
models\resnet\process_casia2.bat

# On Unix-like systems
bash models/resnet/preprocess_data.sh
```

#### Using Other Datasets

1. Organize your raw data in the following structure:
```
data/raw/
├── authentic/
│   ├── image1.png
│   ├── image2.jpg
│   └── ...
└── tampered/
    ├── image1.png
    ├── image2.jpg
    └── ...
```

2. Preprocess the data using the provided script:
```
# On Windows
powershell -ExecutionPolicy Bypass -File models\resnet\preprocess_data.ps1

# On Unix-like systems
bash models/resnet/preprocess_data.sh
```

### Training

#### Training on CASIA2 Dataset

To train a ResNet model on the CASIA2 dataset:

```
# On Windows
models\resnet\train_casia2.bat
```

#### Training on Other Datasets

To train a ResNet model on other datasets:

```
# On Windows
models\resnet\train_resnet.bat

# On Unix-like systems
bash models/resnet/train_resnet.sh
```

You can modify the scripts to change hyperparameters or train different ResNet variants.

### Evaluation

To evaluate a trained model:

```
# On Windows
models\resnet\evaluate_casia2.bat  # For CASIA2 dataset
models\resnet\evaluate_resnet.bat  # For other datasets

# On Unix-like systems
bash models/resnet/evaluate_resnet.sh
```

### Visualization

To visualize the model performance and misclassified samples:

```
# On Windows
models\resnet\visualize_casia2.bat
```

## Files

- `resnet_model.py`: ResNet model architecture implementation
- `train_resnet.py`: Training script for ResNet models
- `evaluate_resnet.py`: Evaluation script for trained models
- `preprocess_data.py`: Script to prepare dataset for training
- `process_casia2.py`: Script to prepare CASIA2 dataset for training
- `visualize_results.py`: Script to visualize model performance and misclassified samples
- `*.bat` and `*.sh`: Batch and shell scripts for running the processes

## Requirements

- PyTorch >= 1.7.0
- torchvision
- NumPy
- Matplotlib
- scikit-learn
- PIL (Pillow)
- tqdm
- seaborn
- OpenCV

## References

- He, K., Zhang, X., Ren, S., & Sun, J. (2016). Deep residual learning for image recognition. CVPR 2016.
- Fridrich, J., & Kodovsky, J. (2012). Rich models for steganalysis of digital images. IEEE TIFS.
- Woo, S., Park, J., Lee, J. Y., & Kweon, I. S. (2018). CBAM: Convolutional block attention module. ECCV 2018.
