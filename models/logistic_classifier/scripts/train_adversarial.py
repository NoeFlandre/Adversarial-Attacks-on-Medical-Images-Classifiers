import torch
from torch.utils.data import DataLoader
from torch import nn, optim
import os
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from ..src.attacks import FGSM

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

def train_model_adversarial(model, train_dataset, val_dataset, epochs=10, batch_size=64, 
                           lr=1e-5, device='cpu', logger=None, epsilon=0.05, mix_ratio=0.5):
    """
    Train a model with adversarial training using FGSM.
    
    Args:
        model: Model to train
        train_dataset: Training dataset
        val_dataset: Validation dataset
        epochs: Number of training epochs
        batch_size: Batch size for training
        lr: Learning rate
        device: Device to train on
        logger: Logger for recording info
        epsilon: Perturbation size for FGSM attack
        mix_ratio: Ratio of adversarial examples in each batch (0.0 to 1.0)
    """
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, 
                                 num_workers=min(4, os.cpu_count()))
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, 
                               num_workers=min(4, os.cpu_count()))
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
    
    # Create FGSM attack for adversarial training
    fgsm_attack = FGSM(model, criterion=criterion, epsilon=epsilon)

    logger.info(f"Training with adversarial examples (FGSM, epsilon={epsilon}, mix_ratio={mix_ratio})")
    logger.info(f"Batch size: {batch_size}, initial learning rate: {lr}, epochs: {epochs}")
    logger.info(f"Class weights: {class_weights.tolist()}")
    
    # Track metrics for plotting
    train_losses = []
    val_losses = []
    clean_val_accuracies = []
    adv_val_accuracies = []
    prev_loss = float('inf')
    
    model.train()
    for epoch in range(epochs):
        # Training phase
        total_loss = 0
        for images, labels in train_dataloader:
            images, labels = images.to(device), labels.to(device)
            
            # Split batch into clean and adversarial samples
            split_idx = int(images.size(0) * (1 - mix_ratio))
            
            # Clean sample training
            optimizer.zero_grad()
            clean_outputs = model(images[:split_idx])
            clean_loss = criterion(clean_outputs, labels[:split_idx])
            
            # Create adversarial examples for the rest of the batch
            if mix_ratio > 0 and split_idx < images.size(0):
                # Detach to avoid gradient through gradient calculation (second order derivatives)
                adv_images = fgsm_attack.generate(images[split_idx:].clone().detach(), 
                                                 labels[split_idx:])
                adv_outputs = model(adv_images)
                adv_loss = criterion(adv_outputs, labels[split_idx:])
                
                # Combine losses based on mix ratio
                if split_idx > 0:
                    loss = (1 - mix_ratio) * clean_loss + mix_ratio * adv_loss
                else:
                    loss = adv_loss
            else:
                loss = clean_loss
                
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        # Record training metrics
        train_loss = total_loss / len(train_dataloader)
        train_losses.append(train_loss)
        
        # Validation phase (clean data)
        val_metrics_clean = evaluate_batch(model, val_dataloader, criterion, device)
        val_losses.append(val_metrics_clean['loss'])
        clean_val_accuracies.append(val_metrics_clean['accuracy'])
        
        # Evaluate on adversarial examples from validation set
        adv_val_accuracy = evaluate_adversarial_accuracy(model, val_dataset, epsilon, 
                                                        batch_size, device)
        adv_val_accuracies.append(adv_val_accuracy)
        
        # Adjust learning rate based on validation loss improvement
        if epoch > 0:
            improvement = (prev_loss - val_metrics_clean['loss']) / prev_loss
            if improvement < 0.01:  # If improvement is less than 1%
                for param_group in optimizer.param_groups:
                    param_group['lr'] *= 0.8  # Reduce learning rate by 20%
                logger.info(f"Reducing learning rate to {optimizer.param_groups[0]['lr']:.2e}")
        
        prev_loss = val_metrics_clean['loss']
        logger.info(f"[Epoch {epoch+1}/{epochs}] Train Loss: {train_loss:.4f}, "
                   f"Clean Val Acc: {val_metrics_clean['accuracy']:.4f}, "
                   f"Adv Val Acc: {adv_val_accuracy:.4f}")
    
    # Save training plot in a dedicated adversarial training directory
    current_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    results_dir = os.path.join(current_dir, 'results/adversarial_training')
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
    
    # Add timestamp to make filenames unique
    import datetime
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Plot training and validation loss
    plt.figure(figsize=(8, 6))
    plt.plot(range(1, epochs + 1), train_losses, marker='o', label='Training Loss')
    plt.plot(range(1, epochs + 1), val_losses, marker='o', label='Validation Loss')
    plt.title('Training and Validation Loss Over Epochs (Adversarial Training)')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    
    plots_path = os.path.join(results_dir, f'training_validation_loss_eps{epsilon}_mix{mix_ratio}_{timestamp}.png')
    plt.savefig(plots_path)
    plt.close()
    
    # Plot clean vs. adversarial accuracy
    plt.figure(figsize=(8, 6))
    plt.plot(range(1, epochs + 1), clean_val_accuracies, marker='o', label='Clean Validation Accuracy')
    plt.plot(range(1, epochs + 1), adv_val_accuracies, marker='o', label='Adversarial Validation Accuracy')
    plt.title(f'Clean vs. Adversarial Accuracy (ε={epsilon}, mix={mix_ratio})')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)
    
    acc_plots_path = os.path.join(results_dir, f'clean_vs_adversarial_accuracy_eps{epsilon}_mix{mix_ratio}_{timestamp}.png')
    plt.savefig(acc_plots_path)
    plt.close()
    
    # Save training history as JSON for later analysis
    import json
    history = {
        'epochs': epochs,
        'epsilon': float(epsilon),
        'mix_ratio': float(mix_ratio),
        'train_losses': [float(loss) for loss in train_losses],
        'val_losses': [float(loss) for loss in val_losses],
        'clean_val_accuracies': [float(acc) for acc in clean_val_accuracies],
        'adv_val_accuracies': [float(acc) for acc in adv_val_accuracies],
    }
    
    history_path = os.path.join(results_dir, f'training_history_eps{epsilon}_mix{mix_ratio}_{timestamp}.json')
    with open(history_path, 'w') as f:
        json.dump(history, f, indent=4)
    
    logger.info(f"Saved training plots to {plots_path} and {acc_plots_path}")
    logger.info(f"Saved training history to {history_path}")
    
    return model

def evaluate_adversarial_accuracy(model, dataset, epsilon, batch_size=64, device='cpu'):
    """
    Evaluate model accuracy against adversarial examples.
    
    Args:
        model: Model to evaluate
        dataset: Dataset to use for evaluation
        epsilon: Perturbation size for FGSM attack
        batch_size: Batch size for evaluation
        device: Device to run evaluation on
        
    Returns:
        Accuracy on adversarial examples
    """
    model.eval()
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    
    criterion = nn.CrossEntropyLoss()
    fgsm_attack = FGSM(model, criterion=criterion, epsilon=epsilon)
    
    correct = 0
    total = 0
    
    for images, labels in dataloader:
        images, labels = images.to(device), labels.to(device)
        
        # Generate adversarial examples
        adv_images = fgsm_attack.generate(images, labels)
        
        with torch.no_grad():
            outputs = model(adv_images)
            _, predicted = torch.max(outputs.data, 1)
            
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
    
    return correct / total