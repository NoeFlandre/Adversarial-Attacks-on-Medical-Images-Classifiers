import torch
from torch.utils.data import DataLoader
from torch import nn, optim
import os
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from src.attacks import DeepFool


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


def train_model_adversarial_deepfool(model, train_dataset, val_dataset,
                                     epochs=10, batch_size=64,
                                     lr=1e-5, device='cpu', logger=None,
                                     max_iter=50, overshoot=0.02, mix_ratio=0.5):
    """
    Train a model with adversarial training using DeepFool.
    Args:
        model: Model to train
        train_dataset: Training dataset
        val_dataset: Validation dataset
        epochs: Number of training epochs
        batch_size: Batch size for training
        lr: Learning rate
        device: Device to train on
        logger: Logger for recording info
        max_iter: Max iterations for DeepFool attack
        overshoot: Overshoot factor for DeepFool
        mix_ratio: Ratio of adversarial examples in each batch (0.0 to 1.0)
    """
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                              num_workers=min(4, os.cpu_count()))
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False,
                            num_workers=min(4, os.cpu_count()))
    model.to(device)
    
    # Compute class weights
    all_labels = torch.tensor([label for _, label in train_dataset])
    class_counts = torch.bincount(all_labels)
    total_samples = len(all_labels)
    class_weights = total_samples / (len(class_counts) * class_counts)
    class_weights = class_weights.to(device)
    
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    
    # DeepFool attack
    deepfool_attack = DeepFool(model, criterion=criterion,
                                max_iter=max_iter, overshoot=overshoot)

    logger.info(f"Training with DeepFool adversarial examples (max_iter={max_iter}, overshoot={overshoot}, mix_ratio={mix_ratio})")
    train_losses, val_losses, clean_accs, adv_accs = [], [], [], []
    prev_loss = float('inf')

    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            split = int(images.size(0) * (1 - mix_ratio))
            optimizer.zero_grad()
            # clean
            out_clean = model(images[:split])
            loss_clean = criterion(out_clean, labels[:split])
            # adversarial
            if mix_ratio > 0 and split < images.size(0):
                adv_imgs = deepfool_attack.generate(images[split:].clone().detach(), labels[split:])
                out_adv = model(adv_imgs.to(device))
                loss_adv = criterion(out_adv, labels[split:])
                loss = (1 - mix_ratio) * loss_clean + mix_ratio * loss_adv if split > 0 else loss_adv
            else:
                loss = loss_clean
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        # record
        train_loss = total_loss / len(train_loader)
        train_losses.append(train_loss)
        # validation clean
        val_metrics = evaluate_batch(model, val_loader, criterion, device)
        val_losses.append(val_metrics['loss'])
        clean_accs.append(val_metrics['accuracy'])
        # validation adversarial
        adv_acc = evaluate_adversarial_accuracy_deepfool(model, val_dataset,
                                                         max_iter, overshoot,
                                                         batch_size, device)
        adv_accs.append(adv_acc)
        # lr scheduling
        if epoch > 0:
            improvement = (prev_loss - val_metrics['loss']) / prev_loss
            if improvement < 0.01:
                for g in optimizer.param_groups:
                    g['lr'] *= 0.8
                logger.info(f"Reducing LR to {optimizer.param_groups[0]['lr']:.2e}")
        prev_loss = val_metrics['loss']
        logger.info(f"[Epoch {epoch+1}/{epochs}] Loss: {train_loss:.4f}, Clean Acc: {val_metrics['accuracy']:.4f}, Adv Acc: {adv_acc:.4f}")

    # save plots
    current = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    results_dir = os.path.join(current, 'results/adversarial_training_deepfool')
    os.makedirs(results_dir, exist_ok=True)
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    # loss plot
    plt.figure(figsize=(8,6))
    plt.plot(range(1, epochs+1), train_losses, 'o-', label='Train Loss')
    plt.plot(range(1, epochs+1), val_losses, 'o-', label='Val Loss')
    plt.title('Training and Validation Loss (DeepFool)')
    plt.xlabel('Epoch'); plt.ylabel('Loss'); plt.legend(); plt.grid()
    loss_path = os.path.join(results_dir, f'deepfool_loss_max{max_iter}_overshoot{overshoot}_mix{mix_ratio}_{ts}.png')
    plt.savefig(loss_path); plt.close()
    # accuracy plot
    plt.figure(figsize=(8,6))
    plt.plot(range(1, epochs+1), clean_accs, 'o-', label='Clean Acc')
    plt.plot(range(1, epochs+1), adv_accs, 'o-', label='Adv Acc')
    plt.title(f'Clean vs Adversarial Accuracy (DeepFool)')
    plt.xlabel('Epoch'); plt.ylabel('Accuracy'); plt.legend(); plt.grid()
    acc_path = os.path.join(results_dir, f'deepfool_acc_max{max_iter}_overshoot{overshoot}_mix{mix_ratio}_{ts}.png')
    plt.savefig(acc_path); plt.close()
    # history
    import json, datetime
    history = {
        'epochs': epochs,
        'mix_ratio': mix_ratio,
        'max_iter': max_iter,
        'overshoot': overshoot,
        'train_losses': [float(x) for x in train_losses],
        'val_losses': [float(x) for x in val_losses],
        'clean_accuracies': [float(x) for x in clean_accs],
        'adv_accuracies': [float(x) for x in adv_accs]
    }
    hist_path = os.path.join(results_dir, f'deepfool_history_max{max_iter}_overshoot{overshoot}_mix{mix_ratio}_{ts}.json')
    with open(hist_path, 'w') as f:
        json.dump(history, f, indent=4)
    logger.info(f"Saved plots to {loss_path} and {acc_path}, history to {hist_path}")
    return model


def evaluate_adversarial_accuracy_deepfool(model, dataset, max_iter, overshoot,
                                           batch_size=64, device='cpu'):
    """
    Evaluate model accuracy against DeepFool adversarial examples.
    """
    model.eval()
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    criterion = nn.CrossEntropyLoss()
    deepfool_attack = DeepFool(model, criterion=criterion,
                                max_iter=max_iter, overshoot=overshoot)
    correct, total = 0, 0
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            adv_imgs = deepfool_attack.generate(images.clone().detach(), labels)
            outputs = model(adv_imgs.to(device))
            _, pred = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (pred == labels).sum().item()
    return correct / total
