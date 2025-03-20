import torch.nn as nn
import torch.optim as optim
import torch.utils.data
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau
from torch.autograd import Variable
import time
import numpy as np
import torch
import os
import copy


def create_loss_and_optimizer(net, learning_rate=0.01, optimizer_type='sgd'):
    """
    Creates the loss function and optimizer of the network.
    :param net: The network object
    :param learning_rate: The initial learning rate
    :param optimizer_type: Type of optimizer to use ('sgd' or 'adam')
    :returns: The loss function and the optimizer
    """
    # Use weighted cross entropy loss for imbalanced datasets
    loss = nn.CrossEntropyLoss()
    
    if optimizer_type.lower() == 'adam':
        optimizer = optim.Adam(net.parameters(), lr=learning_rate, weight_decay=1e-4)
    else:  # default to SGD
        optimizer = optim.SGD(net.parameters(), lr=learning_rate, momentum=0.9, weight_decay=5e-4, nesterov=True)
    
    return loss, optimizer


def train_net(net, train_set, n_epochs, learning_rate, batch_size, val_set=None, 
              optimizer_type='sgd', scheduler_type='cosine', early_stopping=True, 
              patience=10, use_mixed_precision=True, save_path='model_checkpoint.pt'):
    """
    Training of the CNN with advanced techniques
    :param net: The CNN object
    :param train_set: The training part of the dataset
    :param n_epochs: The number of epochs of the experiment
    :param learning_rate: The initial learning rate
    :param batch_size: The batch size of the SGD
    :param val_set: Validation dataset (optional)
    :param optimizer_type: Type of optimizer ('sgd' or 'adam')
    :param scheduler_type: Type of learning rate scheduler ('cosine', 'plateau', or 'step')
    :param early_stopping: Whether to use early stopping
    :param patience: Number of epochs to wait for improvement before early stopping
    :param use_mixed_precision: Whether to use mixed precision training
    :param save_path: Path to save the best model
    :returns: The epoch loss (vector) and the epoch accuracy (vector)
    """
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True, pin_memory=True)
    
    if val_set:
        val_loader = torch.utils.data.DataLoader(val_set, batch_size=batch_size, shuffle=False, pin_memory=True)
    
    criterion, optimizer = create_loss_and_optimizer(net, learning_rate, optimizer_type)
    
    # Setup learning rate scheduler
    if scheduler_type == 'cosine':
        scheduler = CosineAnnealingLR(optimizer, T_max=n_epochs, eta_min=1e-6)
    elif scheduler_type == 'plateau' and val_set:
        scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5, verbose=True)
    else:  # default to step scheduler
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
    
    # Setup mixed precision training if available
    scaler = torch.cuda.amp.GradScaler() if use_mixed_precision and torch.cuda.is_available() else None
    
    n_batches = len(train_loader)
    epoch_loss = []
    epoch_accuracy = []
    val_epoch_loss = []
    val_epoch_accuracy = []
    
    # Early stopping variables
    best_val_loss = float('inf')
    best_model_wts = copy.deepcopy(net.state_dict())
    no_improve_epochs = 0
    
    for epoch in range(n_epochs):
        # Training phase
        net.train()
        total_running_loss = 0.0
        print_every = n_batches // 5
        training_start_time = time.time()
        c = 0
        total_predicted = []
        total_labels = []

        for i, (inputs, labels) in enumerate(train_loader):
            # get the inputs
            if torch.cuda.is_available():
                inputs = inputs.cuda()
                labels = labels.cuda().long()
            
            # zero the parameter gradients
            optimizer.zero_grad()
            
            # Mixed precision training
            if scaler:
                with torch.cuda.amp.autocast():
                    outputs = net(inputs)
                    loss = criterion(outputs, labels)
                
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                # Standard training
                outputs = net(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

            _, predicted = torch.max(outputs.data, 1)

            total_labels.extend(labels.cpu().numpy())
            total_predicted.extend(predicted.cpu().numpy())

            if (i + 1) % (print_every + 1) == 0:
                total_running_loss += loss.item()
                c += 1
        
        # Calculate training metrics
        epoch_predictions = (np.array(total_predicted) == np.array(total_labels)).sum().item()
        train_accuracy = epoch_predictions / len(total_predicted)
        train_loss = total_running_loss / max(c, 1)
        
        epoch_accuracy.append(train_accuracy)
        epoch_loss.append(train_loss)
        
        # Validation phase
        if val_set:
            val_loss, val_accuracy = validate(net, val_loader, criterion)
            val_epoch_loss.append(val_loss)
            val_epoch_accuracy.append(val_accuracy)
            
            print('Epoch %d/%d - Train Loss: %.4f, Train Acc: %.4f, Val Loss: %.4f, Val Acc: %.4f, Time: %.2fs' % 
                  (epoch + 1, n_epochs, train_loss, train_accuracy, val_loss, val_accuracy, 
                   time.time() - training_start_time))
            
            # Update learning rate scheduler if using plateau scheduler
            if scheduler_type == 'plateau':
                scheduler.step(val_loss)
            else:
                scheduler.step()
            
            # Early stopping check
            if early_stopping:
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    best_model_wts = copy.deepcopy(net.state_dict())
                    no_improve_epochs = 0
                    
                    # Save the best model
                    if save_path:
                        torch.save({
                            'epoch': epoch,
                            'model_state_dict': net.state_dict(),
                            'optimizer_state_dict': optimizer.state_dict(),
                            'loss': val_loss,
                            'accuracy': val_accuracy
                        }, save_path)
                else:
                    no_improve_epochs += 1
                
                if no_improve_epochs >= patience:
                    print(f'Early stopping triggered after {epoch + 1} epochs')
                    # Load the best model weights
                    net.load_state_dict(best_model_wts)
                    break
        else:
            print('Epoch %d/%d - Train Loss: %.4f, Train Acc: %.4f, Time: %.2fs' % 
                  (epoch + 1, n_epochs, train_loss, train_accuracy, time.time() - training_start_time))
            scheduler.step()

    print('Finished Training')
    
    # If using early stopping, load the best model weights
    if early_stopping and val_set:
        net.load_state_dict(best_model_wts)
    
    return epoch_loss, epoch_accuracy, val_epoch_loss, val_epoch_accuracy


def validate(net, val_loader, criterion):
    """
    Validate the model on a validation set
    :param net: The CNN object
    :param val_loader: The validation data loader
    :param criterion: The loss function
    :returns: Validation loss and accuracy
    """
    net.eval()
    val_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for inputs, labels in val_loader:
            if torch.cuda.is_available():
                inputs = inputs.cuda()
                labels = labels.cuda().long()
            
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            val_loss += loss.item()
            
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    return val_loss / len(val_loader), correct / total
