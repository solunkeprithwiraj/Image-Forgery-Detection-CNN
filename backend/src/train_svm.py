import pandas as pd
import os
from joblib import dump
from sklearn import svm
from src.classification.SVM import optimize_hyperparams

# Get the absolute path to the project root directory
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

# Read features and labels from CSV
features_path = os.path.join(project_root, 'data', 'output', 'features', 'CASIA2_WithRot_LR001_b128_nodrop.csv')
print(f"Loading features from: {features_path}")
df = pd.read_csv(filepath_or_buffer=features_path)
X = df.loc[:, ~df.columns.isin(['labels', 'image_names'])]
y = df['labels']

img_ids = df['image_names']

print('Has NaN:', df.isnull().values.any())

# Define hyperparameter search space
hyper_params = [{'kernel': ['rbf'], 'gamma': [1e-3, 1e-4], 'C': [0.001, 0.01, 0.1, 1, 10, 100, 1000]}]

# Find optimal hyperparameters
print("Optimizing hyperparameters...")
opt_params = optimize_hyperparams(X, y, params=hyper_params)

# Train the final model with optimal parameters
print("Training final model with optimal parameters...")
final_model = svm.SVC(kernel='rbf', gamma=opt_params['gamma'], C=opt_params['C'], probability=True)
final_model.fit(X.values, y.values)

# Save the trained model
output_dir = os.path.join(project_root, 'data', 'output', 'pre_trained_svm')
os.makedirs(output_dir, exist_ok=True)
model_path = os.path.join(output_dir, 'CASIA2_WithRot_LR001_b128_nodrop.pt')
print(f"Saving model to: {model_path}")
dump(final_model, model_path)
print("Model saved successfully!")