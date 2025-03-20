import os
import sys
import pandas as pd
import numpy as np
import torch
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
import seaborn as sns
import time
import joblib
from datetime import datetime

# Add the parent directory to sys.path to allow imports from sibling modules
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from models.cnn import ImprovedCNN
from utils.feature_fusion import get_y_hat, get_spatial_pyramid_features, ensemble_fusion
from models.advanced_classifiers import (
    train_xgboost_model,
    train_svm_model,
    train_ensemble_model,
    evaluate_model
)
from utils.common import plot_confusion_matrix, plot_roc_curve
from configs.model_config import CNN_CONFIG, XGBOOST_CONFIG, SVM_CONFIG


def main():
    """
    Demonstration of the improved image forgery detection model
    """
    print("=== Image Forgery Detection with Advanced CNN ===")
    
    # Get the current directory
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(current_dir)
    
    # Paths
    features_path = os.path.join(project_root, 'data', 'output', 'features', 'CASIA2_WithRot_LR001_b128_nodrop.csv')
    model_save_path = os.path.join(project_root, 'data', 'output', 'pre_trained_cnn', 'improved_model.pt')
    classifier_save_path = os.path.join(project_root, 'data', 'output', 'classifiers', 'ensemble_classifier.joblib')
    
    # Create directories if they don't exist
    os.makedirs(os.path.dirname(classifier_save_path), exist_ok=True)
    
    # Load features and labels
    print("\n1. Loading feature data...")
    df = pd.read_csv(filepath_or_buffer=features_path)
    X = df.loc[:, ~df.columns.isin(['labels', 'image_names'])]
    y = df['labels']
    img_ids = df['image_names']

    # Ensure labels are integers starting from 0 (for XGBoost compatibility)
    if not all(isinstance(label, (int, np.integer)) for label in y):
        print("Converting labels to integers...")
        label_encoder = LabelEncoder()
        y = label_encoder.fit_transform(y)
        print(f"Labels encoded as: {np.unique(y)}")

    print(f"Loaded {len(X)} samples with {X.shape[1]} features")
    print(f"Class distribution: {pd.Series(y).value_counts().to_dict()}")
    
    # Preprocess data
    print("\n2. Preprocessing data...")
    scaler = StandardScaler()
    X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)
    
    # Handle imbalanced data if needed
    if y.value_counts().iloc[0] / y.value_counts().iloc[1] > 1.5:
        print("\n3. Handling imbalanced data...")
        X_resampled, y_resampled = handle_imbalanced_data(X_scaled, y, method='smote')
        X_train, X_test, y_train, y_test = train_test_split(
            X_resampled, y_resampled, test_size=0.2, random_state=42, stratify=y_resampled
        )
    else:
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y, test_size=0.2, random_state=42, stratify=y
        )
    
    # Train ensemble classifier
    print("\n4. Training ensemble classifier...")
    ensemble_model = train_ensemble_classifier(X_train, y_train)
    
    # Evaluate ensemble classifier
    print("\n5. Evaluating ensemble classifier...")
    ensemble_metrics = evaluate_classifier(ensemble_model, X_test, y_test)
    
    # Save the ensemble model
    save_model(ensemble_model, classifier_save_path)
    
    # Optimize SVM classifier (optional)
    print("\n6. Optimizing SVM classifier...")
    best_svm, best_params = optimize_classifier(X_train, y_train, classifier_type='svm')
    print(f"Best SVM parameters: {best_params}")
    
    # Train deep neural network classifier
    print("\n7. Training deep neural network classifier...")
    input_size = X_train.shape[1]
    deep_model = train_deep_classifier(
        X_train, y_train, 
        input_size=input_size,
        hidden_sizes=[256, 128, 64],
        batch_size=32,
        learning_rate=0.001,
        num_epochs=50
    )
    
    # Save the deep model
    save_model(deep_model, model_save_path.replace('.pt', '_deep_classifier.pt'))
    
    print("\n=== Demonstration of Advanced Feature Fusion ===")
    
    # This part would typically be done with actual image patches
    # For demonstration, we'll simulate patch features
    print("\n8. Demonstrating advanced feature fusion techniques...")
    
    # Simulate 9 patches from an image
    num_patches = 9
    patch_features = []
    for i in range(num_patches):
        # Simulate features from a patch (using random subset of our features)
        random_indices = np.random.choice(len(X_test), 1)
        patch_features.append(X_test.iloc[random_indices].values[0])
    
    # Apply different fusion techniques
    print("\nFusion results:")
    max_fusion = get_y_hat(patch_features, "max")
    mean_fusion = get_y_hat(patch_features, "mean")
    attention_fusion = get_y_hat(patch_features, "attention")
    
    # Compare fusion results (just showing first 5 features)
    fusion_comparison = pd.DataFrame({
        'Max Fusion': max_fusion[:5],
        'Mean Fusion': mean_fusion[:5],
        'Attention Fusion': attention_fusion[:5]
    })
    print(fusion_comparison)
    
    print("\n=== Demonstration Complete ===")
    print("\nThe improved model includes:")
    print("1. Advanced CNN architecture with residual connections and attention mechanisms")
    print("2. Enhanced feature fusion techniques (attention-based, PCA-based, etc.)")
    print("3. Ensemble classification methods")
    print("4. Deep neural network classifier option")
    print("5. Handling of imbalanced data")
    print("6. Comprehensive model evaluation")


if __name__ == "__main__":
    main() 