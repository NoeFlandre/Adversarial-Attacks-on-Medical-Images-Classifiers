import os
import torch
from torch.utils.data import DataLoader
from torch import nn, optim
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import numpy as np
import json
import datetime

from ..src.model import LeNet
from ..src.logger import setup_logger
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

def train_model_adversarial(data_dir, epochs=20, batch_size=1024, learning_rate=1e-4, 
                          val_ratio=0.2, seed=42, epsilon=0.05, mix_ratio=0.5, 
                          checkpoint_dir=None, log_dir=None):
    """
    Train the LeNet model with adversarial training.
    
    Args:
        data_dir: Path to the dataset directory
        epochs: Number of training epochs
        batch_size: Batch size for training
        learning_rate: Initial learning rate
        val_ratio: Validation ratio
        seed: Random seed for reproducibility
        epsilon: Perturbation size for FGSM attack
        mix_ratio: Ratio of adversarial examples in each batch (0.0 to 1.0)
        checkpoint_dir: Directory to save checkpoints
        log_dir: Directory to save logs
        
    Returns:
        Trained LeNet model and training history
    """
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 
                         'mps' if torch.backends.mps.is_available() else 
                         'cpu')
    
    # Setup directories
    current_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    if checkpoint_dir is None:
        checkpoint_dir = os.path.join(current_dir, 'checkpoints')
    
    if log_dir is None:
        log_dir = os.path.join(current_dir, 'logs')
    
    # Create a separate directory for adversarial training results
    results_dir = os.path.join(current_dir, 'results/adversarial_training')
    
    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)
    
    # Setup logger
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    logger = setup_logger(name=f'lenet_{timestamp}', log_type='adversarial_training')
    
    logger.info(f"Using device: {device}")
    logger.info(f"Training with adversarial examples (FGSM, epsilon={epsilon}, mix_ratio={mix_ratio})")
    logger.info(f"Training with batch size: {batch_size}, learning rate: {learning_rate}, epochs: {epochs}")
    logger.info(f"Logs will be saved to: {log_dir}")
    logger.info(f"Results will be saved to: {results_dir}")
    logger.info(f"Checkpoints will be saved to: {checkpoint_dir}")
    
    # Create dataloaders
    from ..src.dataset import get_dataloaders
    train_loader, val_loader = get_dataloaders(
        data_dir=data_dir,
        batch_size=batch_size,
        val_ratio=val_ratio,
        seed=seed
    )
    
    # Get validation dataset for adversarial evaluation
    from ..src.dataset import BreastTumorDataset
    val_dataset = BreastTumorDataset(
        root_dir=data_dir,
        train=False,
        val_ratio=val_ratio,
        seed=seed
    )
    
    # Initialize model
    model = LeNet().to(device)
    
    # Calculate class weights for balanced loss
    train_labels = []
    for _, labels in train_loader:
        train_labels.extend(labels.numpy())
    
    class_counts = np.bincount(train_labels)
    total_samples = len(train_labels)
    class_weights = torch.tensor(total_samples / (len(class_counts) * class_counts), dtype=torch.float32).to(device)
    
    # Initialize optimizer and loss function
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    
    # Create learning rate scheduler with warmup and cosine decay
    warmup_epochs = 3
    def get_lr(epoch):
        if epoch < warmup_epochs:
            return learning_rate * (epoch + 1) / warmup_epochs
        else:
            progress = (epoch - warmup_epochs) / (epochs - warmup_epochs)
            return learning_rate * 0.5 * (1 + np.cos(np.pi * progress))
    
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    
    # Create FGSM attack for adversarial training
    fgsm_attack = FGSM(model, criterion=criterion, epsilon=epsilon)
    
    logger.info(f"Class weights: {class_weights.tolist()}")
    logger.info(f"Warmup epochs: {warmup_epochs}")
    
    # Training loop
    train_losses = []
    val_losses = []
    best_val_loss = float('inf')
    patience_counter = 0
    patience = 5
    best_epoch = 0
    best_model_state = None
    eval_freq = 1
    
    # Track metrics for adversarial training
    clean_val_accuracies = []
    adv_val_accuracies = []
    
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
            
            # Split batch into clean and adversarial samples based on mix_ratio
            split_idx = int(images.size(0) * (1 - mix_ratio))
            
            # Clean sample training
            optimizer.zero_grad()
            
            if split_idx > 0:
                clean_outputs = model(images[:split_idx])
                clean_loss = criterion(clean_outputs, labels[:split_idx])
            else:
                clean_loss = 0
            
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
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            # Calculate gradient norm
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=float('inf'))
            total_grad_norm += grad_norm.item()
            num_batches += 1
            
            optimizer.step()
            
            total_loss += loss.item()
            
            # For metrics tracking, only use clean part predictions
            if split_idx > 0:
                _, preds = torch.max(clean_outputs, 1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels[:split_idx].cpu().numpy())
        
        # Calculate training metrics
        train_metrics = {
            'loss': total_loss / len(train_loader),
            'grad_norm': total_grad_norm / num_batches,
        }
        
        if all_preds:  # Only calculate accuracy metrics if we have predictions
            train_metrics.update({
                'accuracy': accuracy_score(all_labels, all_preds),
                'precision': precision_score(all_labels, all_preds, zero_division=0),
                'recall': recall_score(all_labels, all_preds, zero_division=0),
                'f1': f1_score(all_labels, all_preds, zero_division=0)
            })
        
        # Evaluate on validation set if needed
        val_metrics = None
        if (epoch + 1) % eval_freq == 0:
            val_metrics = evaluate_batch(model, val_loader, criterion, device)
            val_losses.append(val_metrics['loss'])
            clean_val_accuracies.append(val_metrics['accuracy'])
            
            # Evaluate on adversarial examples from validation set
            adv_val_accuracy = evaluate_adversarial_accuracy(model, val_dataset, epsilon, 
                                                          batch_size, device)
            adv_val_accuracies.append(adv_val_accuracy)
            
            # Early stopping check
            if val_metrics['loss'] < best_val_loss:
                best_val_loss = val_metrics['loss']
                best_epoch = epoch
                patience_counter = 0
                # Save best model state
                best_model_state = model.state_dict().copy()
                
                # Save checkpoint
                checkpoint_path = os.path.join(checkpoint_dir, f'lenet_adv_model_best.pth')
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'val_loss': val_metrics['loss'],
                    'val_accuracy': val_metrics['accuracy'],
                    'adversarial_training': {
                        'epsilon': epsilon,
                        'mix_ratio': mix_ratio
                    }
                }, checkpoint_path)
                logger.info(f"Saved best model checkpoint to {checkpoint_path}")
            else:
                patience_counter += 1
        
        train_losses.append(train_metrics['loss'])
        
        # Log metrics
        log_msg = f"[Epoch {epoch+1}/{epochs}]"
        log_msg += f"\n  Training - Loss: {train_metrics['loss']:.4f}, Grad Norm: {train_metrics['grad_norm']:.4f}"
        if 'accuracy' in train_metrics:
            log_msg += f"\n  Training - Accuracy: {train_metrics['accuracy']:.4f}, Precision: {train_metrics['precision']:.4f}"
            log_msg += f"\n  Training - Recall: {train_metrics['recall']:.4f}, F1: {train_metrics['f1']:.4f}"
        log_msg += f"\n  Learning Rate: {current_lr:.2e}"
        
        if val_metrics:
            log_msg += f"\n  Validation - Loss: {val_metrics['loss']:.4f}"
            log_msg += f"\n  Validation - Clean Accuracy: {val_metrics['accuracy']:.4f}"
            log_msg += f"\n  Validation - Adversarial Accuracy: {adv_val_accuracy:.4f}"
            log_msg += f"\n  Validation - Balanced Accuracy: {val_metrics['balanced_accuracy']:.4f}"
            log_msg += f"\n  Validation - Precision: {val_metrics['precision']:.4f}"
            log_msg += f"\n  Validation - Recall: {val_metrics['recall']:.4f}"
        
        logger.info(log_msg)
        
        # Early stopping check
        if patience_counter >= patience:
            logger.info(f"Early stopping triggered after {epoch+1} epochs")
            break
    
    # Restore best model
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
        logger.info(f"Restored best model from epoch {best_epoch+1}")
    
    # Save final checkpoint
    checkpoint_path = os.path.join(checkpoint_dir, f'lenet_adv_model_eps{epsilon}_mix{mix_ratio}_{timestamp}.pth')
    torch.save({
        'epoch': epochs,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'val_loss': val_losses[-1] if val_losses else None,
        'adversarial_training': {
            'epsilon': epsilon,
            'mix_ratio': mix_ratio
        }
    }, checkpoint_path)
    logger.info(f"Saved final model checkpoint to {checkpoint_path}")
    
    # Plot training/validation loss
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='Training Loss')
    if val_losses:
        plt.plot(list(range(0, len(val_losses) * eval_freq, eval_freq)), val_losses, label='Validation Loss')
    plt.title('Training and Validation Loss (Adversarial Training)')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    
    # Save plot to results directory
    loss_plot_path = os.path.join(results_dir, f'training_validation_loss_eps{epsilon}_mix{mix_ratio}_{timestamp}.png')
    plt.savefig(loss_plot_path)
    logger.info(f"Saved loss plot to {loss_plot_path}")
    plt.close()
    
    # Plot clean vs. adversarial accuracy
    plt.figure(figsize=(10, 6))
    plt.plot(list(range(0, len(clean_val_accuracies) * eval_freq, eval_freq)), clean_val_accuracies, marker='o', label='Clean Validation Accuracy')
    plt.plot(list(range(0, len(adv_val_accuracies) * eval_freq, eval_freq)), adv_val_accuracies, marker='o', label='Adversarial Validation Accuracy')
    plt.title(f'Clean vs. Adversarial Accuracy (ε={epsilon}, mix={mix_ratio})')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    
    acc_plots_path = os.path.join(results_dir, f'clean_vs_adversarial_accuracy_eps{epsilon}_mix{mix_ratio}_{timestamp}.png')
    plt.savefig(acc_plots_path)
    plt.close()
    
    # Save training history as JSON for later analysis
    history = {
        'epochs': epochs,
        'epsilon': float(epsilon),
        'mix_ratio': float(mix_ratio),
        'train_losses': [float(loss) for loss in train_losses],
        'val_losses': [float(loss) for loss in val_losses if val_losses],
        'clean_val_accuracies': [float(acc) for acc in clean_val_accuracies],
        'adv_val_accuracies': [float(acc) for acc in adv_val_accuracies],
    }
    
    history_path = os.path.join(results_dir, f'training_history_eps{epsilon}_mix{mix_ratio}_{timestamp}.json')
    with open(history_path, 'w') as f:
        json.dump(history, f, indent=4)
    
    logger.info(f"Saved training plots to {loss_plot_path} and {acc_plots_path}")
    logger.info(f"Saved training history to {history_path}")
    
    return model, history