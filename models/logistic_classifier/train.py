import torch
from torch.utils.data import DataLoader
from torch import nn, optim
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

def train_model(model, train_dataset, val_dataset, epochs=10, batch_size=64, lr=1e-5, device='cpu', logger=None):
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=min(4, os.cpu_count()))
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=min(4, os.cpu_count()))
    model.to(device)
    
    # Calculate class weights for balanced loss
    all_labels = torch.tensor([label for _, label in train_dataset])
    class_counts = torch.bincount(all_labels)
    total_samples = len(all_labels)
    class_weights = total_samples / (len(class_counts) * class_counts)
    class_weights = class_weights.to(device)
    
    # Use Adam optimizer with small learning rate and weight decay
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    
    # Use weighted cross entropy loss for class imbalance
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    logger.info(f"Training with batch size: {batch_size}, initial learning rate: {lr}, epochs: {epochs}")
    logger.info(f"Class weights: {class_weights.tolist()}")
    
    # Track metrics for plotting
    train_losses = []
    val_losses = []
    prev_loss = float('inf')
    
    model.train()
    for epoch in range(epochs):
        # Training phase
        total_loss = 0
        for images, labels in train_dataloader:
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        # Record training metrics
        train_loss = total_loss / len(train_dataloader)
        train_losses.append(train_loss)
        
        # Validation phase
        val_metrics = evaluate_batch(model, val_dataloader, criterion, device)
        val_losses.append(val_metrics['loss'])
        
        # Adjust learning rate based on validation loss improvement
        if epoch > 0:
            improvement = (prev_loss - val_metrics['loss']) / prev_loss
            if improvement < 0.01:  # If improvement is less than 1%
                for param_group in optimizer.param_groups:
                    param_group['lr'] *= 0.8  # Reduce learning rate by 20%
                logger.info(f"Reducing learning rate to {optimizer.param_groups[0]['lr']:.2e}")
        
        prev_loss = val_metrics['loss']
        logger.info(f"[Epoch {epoch+1}/{epochs}] Train Loss: {train_loss:.4f}, Val Loss: {val_metrics['loss']:.4f}, Val Accuracy: {val_metrics['accuracy']:.4f}")
    
    # Save training plot
    current_dir = os.path.dirname(os.path.abspath(__file__))
    results_dir = os.path.join(current_dir, 'results/training_evaluation')
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
    
    # Plot training and validation loss
    plt.figure(figsize=(8, 6))
    plt.plot(range(1, epochs + 1), train_losses, marker='o', label='Training Loss')
    plt.plot(range(1, epochs + 1), val_losses, marker='o', label='Validation Loss')
    plt.title('Training and Validation Loss Over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    
    plots_path = os.path.join(results_dir, 'training_validation_loss.png')
    plt.savefig(plots_path)
    logger.info(f"Saved training and validation loss plot to {plots_path}")
    
    return model