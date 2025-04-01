import torch
from torch.utils.data import DataLoader
from torch import nn, optim
from torch.amp import GradScaler, autocast
import os
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

def evaluate_batch(model, dataloader, criterion, device):
    """Evaluate the model on a batch of data during training"""
    model.eval()
    total_loss = 0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)
            
            # Use automatic mixed precision during evaluation
            with autocast('cuda', enabled=torch.cuda.is_available()):
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

class EarlyStopping:
    """Early stopping to prevent overfitting"""
    def __init__(self, patience=7, min_delta=0, verbose=False):
        """
        Args:
            patience (int): How many epochs to wait after last time validation loss improved.
            min_delta (float): Minimum change in the monitored quantity to qualify as an improvement.
            verbose (bool): If True, prints a message for each validation loss improvement.
        """
        self.patience = patience
        self.min_delta = min_delta
        self.verbose = verbose
        self.counter = 0
        self.best_loss = None
        self.early_stop = False
        self.best_model = None
    
    def __call__(self, val_loss, model):
        if self.best_loss is None:
            self.best_loss = val_loss
            return False
        
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
            self.best_model = model.state_dict().copy()
            return False
        else:
            self.counter += 1
            if self.verbose:
                print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
                return True
            return False

def train_model(model, train_dataset, val_dataset, epochs=10, batch_size=64, lr=1e-4, device='cpu', logger=None):
    # Optimize data loading: use more workers and pin memory for GPU transfers
    num_workers = min(8, os.cpu_count() or 1)  # Ensure we have at least 1 worker
    pin_memory = device != 'cpu'  # Only pin memory if using GPU
    
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, 
                                num_workers=num_workers, pin_memory=pin_memory)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, 
                              num_workers=num_workers, pin_memory=pin_memory)
    # Use torch.compile if using PyTorch 2.0+ and CUDA is available
    if hasattr(torch, 'compile') and torch.cuda.is_available():
        logger.info("Using torch.compile for model acceleration")
        model = torch.compile(model)
    
    model.to(device)
    
    # Calculate class weights for balanced loss
    all_labels = torch.tensor([label for _, label in train_dataset])
    class_counts = torch.bincount(all_labels)
    total_samples = len(all_labels)
    class_weights = total_samples / (len(class_counts) * class_counts)
    class_weights = class_weights.to(device)
    
    # Log parameter count
    num_params = model.count_parameters()
    logger.info(f"Model parameters: {num_params:,} ({num_params/(1000000):.2f}M)")
    
    # Use Adam optimizer with weight decay (L2 regularization)
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    
    # Use weighted cross entropy loss for class imbalance
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    
    # Initialize gradient scaler for mixed precision training
    scaler = GradScaler('cuda', enabled=torch.cuda.is_available())
    
    # Learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=3, factor=0.5, verbose=True)

    # Initialize early stopping
    early_stopping = EarlyStopping(patience=5, verbose=True)

    logger.info(f"Training with batch size: {batch_size}, initial learning rate: {lr}, epochs: {epochs}")
    logger.info(f"Class weights: {class_weights.tolist()}")
    logger.info(f"Early stopping enabled with patience: {early_stopping.patience}")
    
    # Track metrics for plotting
    train_losses = []
    val_losses = []
    
    model.train()
    for epoch in range(epochs):
        # Training phase
        total_loss = 0
        for images, labels in train_dataloader:
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()
            
            # Use automatic mixed precision for forward and backward pass
            with autocast('cuda', enabled=torch.cuda.is_available()):
                outputs = model(images)
                loss = criterion(outputs, labels)
            
            # Scale gradients and optimize with mixed precision
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            total_loss += loss.item()
        
        # Record training metrics
        train_loss = total_loss / len(train_dataloader)
        train_losses.append(train_loss)
        
        # Validation phase
        val_metrics = evaluate_batch(model, val_dataloader, criterion, device)
        val_losses.append(val_metrics['loss'])
        
        # Update learning rate scheduler
        scheduler.step(val_metrics['loss'])
        
        logger.info(f"[Epoch {epoch+1}/{epochs}] Train Loss: {train_loss:.4f}, Val Loss: {val_metrics['loss']:.4f}, Val Accuracy: {val_metrics['accuracy']:.4f}")
        
        # Check early stopping
        if early_stopping(val_metrics['loss'], model):
            logger.info(f"Early stopping triggered at epoch {epoch+1}")
            # Load the best model
            model.load_state_dict(early_stopping.best_model)
            break
    
    # Create timestamp for filename
    import datetime
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Save training plot
    current_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    results_dir = os.path.join(current_dir, 'results/training_evaluation')
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
    
    # Plot training and validation loss
    plt.figure(figsize=(8, 6))
    epochs_trained = len(train_losses)
    plt.plot(range(1, epochs_trained + 1), train_losses, marker='o', label='Training Loss')
    plt.plot(range(1, epochs_trained + 1), val_losses, marker='o', label='Validation Loss')
    plt.title('Training and Validation Loss Over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    
    plots_path = os.path.join(results_dir, 'training_validation_loss.png')
    plt.savefig(plots_path)
    logger.info(f"Saved training and validation loss plot to {plots_path}")
    
    # Save training history
    import json
    
    history = {
        'epochs': epochs_trained,
        'train_losses': [float(loss) for loss in train_losses],
        'val_losses': [float(loss) for loss in val_losses],
    }
    
    history_path = os.path.join(results_dir, f'training_history_{timestamp}.json')
    with open(history_path, 'w') as f:
        json.dump(history, f, indent=4)
    
    return model