import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import sys
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold, cross_val_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, precision_recall_curve
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import joblib
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from collections import Counter
import copy

# Add the parent directory to sys.path to allow imports from sibling modules
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))


class DeepClassifier(nn.Module):
    """
    Deep neural network classifier for feature vectors
    """
    def __init__(self, input_size, hidden_sizes=[256, 128, 64], num_classes=2, dropout_rate=0.5):
        super(DeepClassifier, self).__init__()
        
        layers = []
        # Input layer
        layers.append(nn.Linear(input_size, hidden_sizes[0]))
        layers.append(nn.ReLU())
        layers.append(nn.BatchNorm1d(hidden_sizes[0]))
        layers.append(nn.Dropout(dropout_rate))
        
        # Hidden layers
        for i in range(len(hidden_sizes) - 1):
            layers.append(nn.Linear(hidden_sizes[i], hidden_sizes[i+1]))
            layers.append(nn.ReLU())
            layers.append(nn.BatchNorm1d(hidden_sizes[i+1]))
            layers.append(nn.Dropout(dropout_rate))
        
        # Output layer
        layers.append(nn.Linear(hidden_sizes[-1], num_classes))
        
        self.model = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.model(x)


def train_deep_classifier(X, y, input_size, hidden_sizes=[256, 128, 64], num_classes=2, 
                          batch_size=32, learning_rate=0.001, num_epochs=100, 
                          early_stopping=True, patience=10, device=None):
    """
    Train a deep neural network classifier
    :param X: Feature vectors
    :param y: Labels
    :param input_size: Input dimension
    :param hidden_sizes: List of hidden layer sizes
    :param num_classes: Number of output classes
    :param batch_size: Batch size for training
    :param learning_rate: Learning rate
    :param num_epochs: Maximum number of epochs
    :param early_stopping: Whether to use early stopping
    :param patience: Patience for early stopping
    :param device: Device to use (cpu or cuda)
    :return: Trained model
    """
    # Determine device
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Convert data to PyTorch tensors
    X_tensor = torch.FloatTensor(X.values)
    y_tensor = torch.LongTensor(y.values)
    
    # Create dataset and dataloader
    dataset = TensorDataset(X_tensor, y_tensor)
    
    # Split into train and validation sets
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    # Create model
    model = DeepClassifier(input_size, hidden_sizes, num_classes)
    model.to(device)
    
    # Loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    # Training loop
    best_val_loss = float('inf')
    best_model = None
    no_improve_epochs = 0
    
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            
            # Zero the parameter gradients
            optimizer.zero_grad()
            
            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            # Backward pass and optimize
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item() * inputs.size(0)
        
        train_loss = train_loss / len(train_loader.dataset)
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        val_correct = 0
        
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item() * inputs.size(0)
                
                _, preds = torch.max(outputs, 1)
                val_correct += torch.sum(preds == labels.data)
        
        val_loss = val_loss / len(val_loader.dataset)
        val_acc = val_correct.double() / len(val_loader.dataset)
        
        print(f'Epoch {epoch+1}/{num_epochs} - Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}')
        
        # Early stopping
        if early_stopping:
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_model = copy.deepcopy(model.state_dict())
                no_improve_epochs = 0
            else:
                no_improve_epochs += 1
            
            if no_improve_epochs >= patience:
                print(f'Early stopping triggered after {epoch+1} epochs')
                model.load_state_dict(best_model)
                break
    
    # Load best model if using early stopping
    if early_stopping and best_model is not None:
        model.load_state_dict(best_model)
    
    return model


def handle_imbalanced_data(X, y, method='smote', sampling_strategy='auto'):
    """
    Handle imbalanced datasets using various techniques
    :param X: Feature vectors
    :param y: Labels
    :param method: Method to use ('smote', 'undersample', or 'combined')
    :param sampling_strategy: Sampling strategy
    :return: Resampled X and y
    """
    print(f"Original class distribution: {Counter(y)}")
    
    if method == 'smote':
        # Oversample minority class
        smote = SMOTE(sampling_strategy=sampling_strategy, random_state=42)
        X_resampled, y_resampled = smote.fit_resample(X, y)
    elif method == 'undersample':
        # Undersample majority class
        rus = RandomUnderSampler(sampling_strategy=sampling_strategy, random_state=42)
        X_resampled, y_resampled = rus.fit_resample(X, y)
    elif method == 'combined':
        # Combine oversampling and undersampling
        smote = SMOTE(sampling_strategy=0.5, random_state=42)  # Increase minority to 50% of majority
        X_temp, y_temp = smote.fit_resample(X, y)
        
        rus = RandomUnderSampler(sampling_strategy=0.8, random_state=42)  # Reduce majority to 80% of its size
        X_resampled, y_resampled = rus.fit_resample(X_temp, y_temp)
    else:
        raise ValueError("Method must be 'smote', 'undersample', or 'combined'")
    
    print(f"Resampled class distribution: {Counter(y_resampled)}")
    return X_resampled, y_resampled


def train_ensemble_classifier(X, y, models=None, voting='soft'):
    """
    Train an ensemble of classifiers
    :param X: Feature vectors
    :param y: Labels
    :param models: List of (name, model) tuples or None to use default
    :param voting: Voting strategy ('hard' or 'soft')
    :return: Trained ensemble model
    """
    if models is None:
        models = [
            ('svm', SVC(probability=True, kernel='rbf', gamma='scale', C=1.0)),
            ('rf', RandomForestClassifier(n_estimators=100, random_state=42)),
            ('gb', GradientBoostingClassifier(n_estimators=100, random_state=42)),
            ('xgb', XGBClassifier(
                n_estimators=100, 
                random_state=42, 
                use_label_encoder=False,
                eval_metric='logloss'
            )),
            ('mlp', MLPClassifier(hidden_layer_sizes=(100, 50), max_iter=300, random_state=42))
        ]
    
    # Create and train the ensemble
    ensemble = VotingClassifier(estimators=models, voting=voting)
    
    # Train with cross-validation
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = cross_val_score(ensemble, X, y, cv=cv, scoring='accuracy')
    
    print(f"Cross-validation scores: {cv_scores}")
    print(f"Mean CV accuracy: {np.mean(cv_scores):.4f} ± {np.std(cv_scores):.4f}")
    
    # Train on the full dataset
    ensemble.fit(X, y)
    
    return ensemble


def evaluate_classifier(model, X, y, model_type='sklearn'):
    """
    Comprehensive evaluation of a classifier
    :param model: Trained classifier model
    :param X: Feature vectors
    :param y: Labels
    :param model_type: Type of model ('sklearn' or 'pytorch')
    :return: Dictionary of evaluation metrics
    """
    # Split data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
    
    # Make predictions
    if model_type == 'sklearn':
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else None
    elif model_type == 'pytorch':
        model.eval()
        with torch.no_grad():
            X_tensor = torch.FloatTensor(X_test.values)
            outputs = model(X_tensor)
            _, y_pred = torch.max(outputs, 1)
            y_pred = y_pred.numpy()
            y_prob = torch.softmax(outputs, dim=1)[:, 1].numpy() if outputs.shape[1] > 1 else None
    else:
        raise ValueError("model_type must be 'sklearn' or 'pytorch'")
    
    # Calculate metrics
    metrics = {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred, average='binary'),
        'recall': recall_score(y_test, y_pred, average='binary'),
        'f1': f1_score(y_test, y_pred, average='binary')
    }
    
    if y_prob is not None:
        metrics['auc'] = roc_auc_score(y_test, y_prob)
    
    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    
    # Print results
    print(f"Classification Report:")
    print(classification_report(y_test, y_pred))
    
    print(f"Confusion Matrix:")
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    plt.show()
    
    # ROC curve if probabilities are available
    if y_prob is not None:
        plt.figure(figsize=(8, 6))
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        plt.plot(fpr, tpr, label=f'ROC Curve (AUC = {metrics["auc"]:.4f})')
        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curve')
        plt.legend()
        plt.show()
    
    return metrics


def optimize_classifier(X, y, classifier_type='svm', param_grid=None):
    """
    Optimize hyperparameters for a classifier
    :param X: Feature vectors
    :param y: Labels
    :param classifier_type: Type of classifier ('svm', 'rf', 'gb', 'xgb', 'mlp')
    :param param_grid: Parameter grid for optimization or None to use default
    :return: Best model and parameters
    """
    # Define classifier and default parameter grid
    if classifier_type == 'svm':
        classifier = SVC(probability=True)
        if param_grid is None:
            param_grid = {
                'kernel': ['rbf', 'poly'],
                'gamma': ['scale', 'auto', 0.001, 0.01, 0.1],
                'C': [0.1, 1, 10, 100, 1000]
            }
    elif classifier_type == 'rf':
        classifier = RandomForestClassifier(random_state=42)
        if param_grid is None:
            param_grid = {
                'n_estimators': [50, 100, 200],
                'max_depth': [None, 10, 20, 30],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4]
            }
    elif classifier_type == 'gb':
        classifier = GradientBoostingClassifier(random_state=42)
        if param_grid is None:
            param_grid = {
                'n_estimators': [50, 100, 200],
                'learning_rate': [0.01, 0.1, 0.2],
                'max_depth': [3, 5, 7],
                'subsample': [0.8, 1.0]
            }
    elif classifier_type == 'xgb':
        classifier = XGBClassifier(
            random_state=42,
            use_label_encoder=False,
            eval_metric='logloss'
        )
        if param_grid is None:
            param_grid = {
                'n_estimators': [50, 100, 200],
                'learning_rate': [0.01, 0.1, 0.2],
                'max_depth': [3, 5, 7],
                'subsample': [0.8, 1.0],
                'colsample_bytree': [0.8, 1.0]
            }
    elif classifier_type == 'mlp':
        classifier = MLPClassifier(random_state=42, max_iter=300)
        if param_grid is None:
            param_grid = {
                'hidden_layer_sizes': [(50,), (100,), (50, 50), (100, 50)],
                'activation': ['relu', 'tanh'],
                'alpha': [0.0001, 0.001, 0.01],
                'learning_rate': ['constant', 'adaptive']
            }
    else:
        raise ValueError("classifier_type must be 'svm', 'rf', 'gb', 'xgb', or 'mlp'")
    
    # Perform grid search
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    grid_search = GridSearchCV(classifier, param_grid, cv=cv, scoring='accuracy', n_jobs=-1, verbose=1)
    grid_search.fit(X, y)
    
    print(f"Best parameters: {grid_search.best_params_}")
    print(f"Best cross-validation score: {grid_search.best_score_:.4f}")
    
    return grid_search.best_estimator_, grid_search.best_params_


def save_model(model, filename):
    """
    Save a trained model to disk
    :param model: Trained model
    :param filename: Filename to save to
    """
    if isinstance(model, torch.nn.Module):
        torch.save(model.state_dict(), filename)
    else:
        joblib.dump(model, filename)
    print(f"Model saved to {filename}")


def load_model(filename, model_type='sklearn', model_class=None, model_params=None):
    """
    Load a trained model from disk
    :param filename: Filename to load from
    :param model_type: Type of model ('sklearn' or 'pytorch')
    :param model_class: Class of PyTorch model (required if model_type is 'pytorch')
    :param model_params: Parameters for PyTorch model initialization
    :return: Loaded model
    """
    if model_type == 'sklearn':
        model = joblib.load(filename)
    elif model_type == 'pytorch':
        if model_class is None:
            raise ValueError("model_class must be provided for PyTorch models")
        
        model = model_class(**model_params)
        model.load_state_dict(torch.load(filename))
        model.eval()
    else:
        raise ValueError("model_type must be 'sklearn' or 'pytorch'")
    
    print(f"Model loaded from {filename}")
    return model 