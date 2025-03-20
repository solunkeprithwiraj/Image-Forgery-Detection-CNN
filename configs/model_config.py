"""
Configuration file for Image Forgery Detection CNN models.
This file centralizes all model parameters and configurations.
"""

# CNN Model Parameters
CNN_CONFIG = {
    'input_shape': (64, 64, 3),
    'batch_size': 32,
    'epochs': 50,
    'learning_rate': 0.001,
    'validation_split': 0.2,
    'test_split': 0.1,
    'early_stopping_patience': 10,
    'dropout_rate': 0.5,
}

# XGBoost Parameters
XGBOOST_CONFIG = {
    'n_estimators': 100,
    'max_depth': 3,
    'learning_rate': 0.1,
    'objective': 'binary:logistic',
    'eval_metric': 'logloss'
}

# SVM Parameters
SVM_CONFIG = {
    'C': 1.0,
    'kernel': 'rbf',
    'gamma': 'scale'
}

# Feature Extraction Parameters
FEATURE_EXTRACTION = {
    'patch_size': 64,
    'stride': 32,
    'min_patches': 50,
}

# Data Paths
DATA_PATHS = {
    'raw_data': '../data/raw',
    'processed_data': '../data/processed',
    'interim_data': '../data/interim',
    'external_data': '../data/external',
    'model_weights': '../models/weights',
} 