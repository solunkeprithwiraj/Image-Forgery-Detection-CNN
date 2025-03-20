"""
Simplified XGBoost demo to test the fixes for the warnings.
"""
import os
import sys
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from xgboost import XGBClassifier

# Add the parent directory to sys.path to allow imports from sibling modules
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


def main():
    """
    Simplified XGBoost demo
    """
    print("=== XGBoost Demo with Warning Fixes ===")
    
    # Get the current directory
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(current_dir)
    
    # Paths
    features_path = os.path.join(project_root, 'data', 'output', 'features', 'CASIA2_WithRot_LR001_b128_nodrop.csv')
    
    # Load features and labels
    print("\n1. Loading feature data...")
    try:
        df = pd.read_csv(filepath_or_buffer=features_path)
        X = df.loc[:, ~df.columns.isin(['labels', 'image_names'])]
        y = df['labels']
        
        # Ensure labels are integers starting from 0 (for XGBoost compatibility)
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
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Train XGBoost classifier with fixes for warnings
        print("\n3. Training XGBoost classifier with fixes for warnings...")
        xgb_model = XGBClassifier(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=5,
            random_state=42,
            use_label_encoder=False,  # Fix for label encoder warning
            eval_metric='logloss'     # Fix for evaluation metric warning
        )
        
        # Train the model
        xgb_model.fit(X_train, y_train)
        
        # Evaluate the model
        accuracy = xgb_model.score(X_test, y_test)
        print(f"\nAccuracy: {accuracy:.4f}")
        
        print("\n=== Demo Complete ===")
        print("The XGBoost warnings have been fixed by:")
        print("1. Setting use_label_encoder=False")
        print("2. Explicitly setting eval_metric='logloss'")
        print("3. Ensuring labels are integers starting from 0")
        
    except FileNotFoundError:
        print(f"Error: Could not find the features file at {features_path}")
        print("This is a demo file. You can use any CSV file with features and labels.")
        print("For testing purposes, you can create a small random dataset:")
        
        # Create a small random dataset for demonstration
        print("\nCreating a random dataset for demonstration...")
        X = np.random.rand(100, 10)
        y = np.random.randint(0, 2, 100)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Train XGBoost classifier with fixes for warnings
        print("\n3. Training XGBoost classifier with fixes for warnings...")
        xgb_model = XGBClassifier(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=5,
            random_state=42,
            use_label_encoder=False,  # Fix for label encoder warning
            eval_metric='logloss'     # Fix for evaluation metric warning
        )
        
        # Train the model
        xgb_model.fit(X_train, y_train)
        
        # Evaluate the model
        accuracy = xgb_model.score(X_test, y_test)
        print(f"\nAccuracy: {accuracy:.4f}")
        
        print("\n=== Demo Complete ===")
        print("The XGBoost warnings have been fixed by:")
        print("1. Setting use_label_encoder=False")
        print("2. Explicitly setting eval_metric='logloss'")
        print("3. Ensuring labels are integers starting from 0")


if __name__ == "__main__":
    main() 