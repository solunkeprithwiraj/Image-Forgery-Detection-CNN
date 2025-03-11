import pandas as pd
from src.classification.SVM import optimize_hyperparams, classify, print_confusion_matrix, find_misclassified
import os

# Get the current directory
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)

# Read features and labels from CSV
features_path = os.path.join(project_root, 'data', 'output', 'features', 'CASIA2_WithRot_LR001_b128_nodrop.csv')
df = pd.read_csv(filepath_or_buffer=features_path)
X = df.loc[:, ~df.columns.isin(['labels', 'image_names'])]
y = df['labels']

img_ids = df['image_names']

print('Has NaN:', df.isnull().values.any())

hyper_params = [{'kernel': ['rbf'], 'gamma': [1e-3, 1e-4], 'C': [0.001, 0.01, 0.1, 1, 10, 100, 1000]}]

opt_params = optimize_hyperparams(X, y, params=hyper_params)
classify(X, y, opt_params)
print_confusion_matrix(X, y, opt_params)
find_misclassified(X, y, opt_params, img_ids)
