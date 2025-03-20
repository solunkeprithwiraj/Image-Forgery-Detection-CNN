# Source Code Directory

This directory contains the source code for the improved image forgery detection model.

## Demo Scripts

- `improved_model_demo.py`: Main demo script for the improved model with all enhancements
- `minimal_demo.py`: Minimal demo of the improved CNN architecture
- `xgboost_demo.py`: Demo of XGBoost classifier with proper configuration

## Directory Structure

- `cnn/`: CNN implementation including the improved architecture

  - `cnn.py`: Original and improved CNN implementations
  - `minimal_cnn.py`: Minimal version of the improved CNN
  - `SRM_filters.py`: SRM filter implementation
  - `train_cnn.py`: Enhanced training code

- `feature_fusion/`: Feature fusion techniques

  - `feature_fusion.py`: Enhanced feature fusion methods

- `classification/`: Classification methods

  - `advanced_classifiers.py`: Advanced classification techniques including XGBoost

- `patch_extraction/`: Code for extracting patches from images

- `plots/`: Visualization utilities

## Running the Pipeline

1. **Extract CNN training patches**: Use the patch extraction code in `patch_extraction/`

2. **Train CNN**: Configure and run `cnn/train_cnn.py` with the extracted patches

3. **Compute image features**: Use `feature_fusion/feature_fusion.py` with the trained CNN

4. **Run classification**: Use `classification/advanced_classifiers.py` with the extracted features

## Quick Demo

For a quick demonstration of the improved model, run one of the demo scripts:

```bash
python improved_model_demo.py  # Full demo
python minimal_demo.py         # Minimal architecture demo
python xgboost_demo.py         # XGBoost classifier demo
```
