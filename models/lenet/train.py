import os
import torch
from torch.utils.data import DataLoader
from torch import nn, optim
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import seaborn as sns
from model import LeNet
from dataset import BreastHistopathologyDataset
import numpy as np
from utils import get_device
import json
import datetime

def count_parameters(model):
    """Count the number of trainable parameters in the model"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def evaluate_batch(model, dataloader, criterion, device):
    """Evaluate the model on a batch of data during training"""
    model.eval()
    total_loss = 0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            total_loss += loss.item()
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    # Calculate metrics
    accuracy = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, zero_division=0)
    recall = recall_score(all_labels, all_preds, zero_division=0)
    f1 = f1_score(all_labels, all_preds, zero_division=0)
    cm = confusion_matrix(all_labels, all_preds)
    tn, fp, fn, tp = cm.ravel()
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    balanced_accuracy = (recall + specificity) / 2
    
    return {
        'loss': total_loss / len(dataloader),
        'accuracy': accuracy,
        'balanced_accuracy': balanced_accuracy,
        'precision': precision,
        'recall': recall,
        'specificity': specificity,
        'f1': f1,
        'confusion_matrix': cm
    }

def train_model(train_dataset, val_dataset, epochs=20, batch_size=1024, lr=1e-4, 
                device=None, logger=None, patience=5, eval_freq=1, warmup_epochs=3):
    """
    Train the LeNet model
    Args:
        train_dataset: Training dataset
        val_dataset: Validation dataset
        epochs: Number of training epochs
        batch_size: Batch size for training
        lr: Initial learning rate
        device: Device to train on (if None, will be automatically selected)
        logger: Logger instance
        patience: Number of epochs to wait before early stopping
        eval_freq: Frequency of validation evaluation (in epochs)
        warmup_epochs: Number of epochs for learning rate warmup
    Returns:
        Trained LeNet model
    """
    if device is None:
        device = get_device()
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    
    # Initialize model
    model = LeNet().to(device)
    
    # Print model parameters
    num_params = count_parameters(model)
    if logger:
        logger.info(f"Model has {num_params:,} trainable parameters")
    
    # Calculate class weights for balanced loss
    all_labels = torch.tensor([label for _, label in train_dataset])
    class_counts = torch.bincount(all_labels)
    total_samples = len(all_labels)
    class_weights = total_samples / (len(class_counts) * class_counts)
    class_weights = class_weights.to(device)
    
    # Initialize optimizer and loss function
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    
    # Create learning rate scheduler with warmup and cosine decay
    def get_lr(epoch):
        if epoch < warmup_epochs:
            return lr * (epoch + 1) / warmup_epochs
        else:
            progress = (epoch - warmup_epochs) / (epochs - warmup_epochs)
            return lr * 0.5 * (1 + np.cos(np.pi * progress))
    
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    
    if logger:
        logger.info(f"Training with batch size: {batch_size}, initial learning rate: {lr}, epochs: {epochs}")
        logger.info(f"Class weights: {class_weights.tolist()}")
        logger.info(f"Warmup epochs: {warmup_epochs}")
    
    # Training loop
    train_losses = []
    val_losses = []
    best_val_loss = float('inf')
    patience_counter = 0
    best_epoch = 0
    best_model_state = None
    
    for epoch in range(epochs):
        # Update learning rate
        current_lr = get_lr(epoch)
        for param_group in optimizer.param_groups:
            param_group['lr'] = current_lr
        
        model.train()
        total_loss = 0
        all_preds = []
        all_labels = []
        total_grad_norm = 0
        num_batches = 0
        
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            # Calculate gradient norm
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=float('inf'))
            total_grad_norm += grad_norm.item()
            num_batches += 1
            
            optimizer.step()
            
            total_loss += loss.item()
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
        
        # Calculate training metrics
        train_metrics = {
            'loss': total_loss / len(train_loader),
            'grad_norm': total_grad_norm / num_batches,
            'accuracy': accuracy_score(all_labels, all_preds),
            'precision': precision_score(all_labels, all_preds, zero_division=0),
            'recall': recall_score(all_labels, all_preds, zero_division=0),
            'f1': f1_score(all_labels, all_preds, zero_division=0)
        }
        
        # Evaluate on validation set if needed
        val_metrics = None
        if (epoch + 1) % eval_freq == 0:
            val_metrics = evaluate_batch(model, val_loader, criterion, device)
            val_losses.append(val_metrics['loss'])
            
            # Early stopping check
            if val_metrics['loss'] < best_val_loss:
                best_val_loss = val_metrics['loss']
                best_epoch = epoch
                patience_counter = 0
                # Save best model state
                best_model_state = model.state_dict().copy()
            else:
                patience_counter += 1
        
        train_losses.append(train_metrics['loss'])
        
        # Log metrics
        if logger:
            log_msg = f"[Epoch {epoch+1}/{epochs}]"
            log_msg += f"\n  Training - Loss: {train_metrics['loss']:.4f}, Grad Norm: {train_metrics['grad_norm']:.4f}"
            log_msg += f"\n  Training - Accuracy: {train_metrics['accuracy']:.4f}, Precision: {train_metrics['precision']:.4f}"
            log_msg += f"\n  Training - Recall: {train_metrics['recall']:.4f}, F1: {train_metrics['f1']:.4f}"
            log_msg += f"\n  Learning Rate: {current_lr:.2e}"
            
            if val_metrics:
                log_msg += f"\n  Validation - Loss: {val_metrics['loss']:.4f}"
                log_msg += f"\n  Validation - Accuracy: {val_metrics['accuracy']:.4f}"
                log_msg += f"\n  Validation - Balanced Accuracy: {val_metrics['balanced_accuracy']:.4f}"
                log_msg += f"\n  Validation - Precision: {val_metrics['precision']:.4f}"
                log_msg += f"\n  Validation - Recall: {val_metrics['recall']:.4f}"
                log_msg += f"\n  Validation - Specificity: {val_metrics['specificity']:.4f}"
                log_msg += f"\n  Validation - F1: {val_metrics['f1']:.4f}"
            
            logger.info(log_msg)
        
        # Early stopping check
        if patience_counter >= patience:
            if logger:
                logger.info(f"Early stopping triggered after {epoch+1} epochs")
            break
    
    # Restore best model
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
        if logger:
            logger.info(f"Restored best model from epoch {best_epoch + 1}")
    
    # Save training history
    current_dir = os.path.dirname(os.path.abspath(__file__))
    results_dir = os.path.join(current_dir, 'results')
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
    
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Save training history
    history = {
        'train_losses': train_losses,
        'val_losses': val_losses,
        'best_epoch': best_epoch,
        'best_val_loss': best_val_loss,
        'num_parameters': num_params,
        'final_learning_rate': current_lr
    }
    
    with open(os.path.join(results_dir, f'training_history_{timestamp}.json'), 'w') as f:
        json.dump(history, f, indent=4)
    
    # Plot training and validation loss
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(train_losses) + 1), train_losses, marker='o', label='Training Loss')
    plt.plot(range(1, len(val_losses) + 1), val_losses, marker='o', label='Validation Loss')
    plt.title('Training and Validation Loss Over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid(True)
    plt.legend()
    
    plt.savefig(os.path.join(results_dir, f'loss_curves_{timestamp}.png'))
    plt.close()
    
    if logger:
        logger.info(f"Saved training history and plots to {results_dir}")
    
    return model

def plot_training_results(model, dataset, device=None, logger=None):
    """
    Plot training results and save them
    Args:
        model: Trained LeNet model
        dataset: BreastHistopathologyDataset instance
        device: Device to use for inference (if None, will be automatically selected)
        logger: Logger instance
    """
    if device is None:
        device = get_device()
    
    model.eval()
    dataloader = DataLoader(dataset, batch_size=64, shuffle=False, num_workers=4)
    
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device)
            outputs = model(images)
            _, preds = torch.max(outputs, 1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())
    
    # Plot confusion matrix
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Confusion Matrix')
    plt.colorbar()
    classes = ['Non-IDC', 'IDC']
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)
    
    # Add text annotations to confusion matrix
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, format(cm[i, j], 'd'),
                    horizontalalignment="center",
                    color="white" if cm[i, j] > thresh else "black")
    
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    
    current_dir = os.path.dirname(os.path.abspath(__file__))
    results_dir = os.path.join(current_dir, 'results')
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
    
    plt.savefig(os.path.join(results_dir, 'confusion_matrix.png'))
    plt.close()
    
    if logger:
        logger.info(f"Saved confusion matrix plot to {os.path.join(results_dir, 'confusion_matrix.png')}") 