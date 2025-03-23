import torch
from torch.utils.data import DataLoader
from torch import nn, optim
import os
import matplotlib.pyplot as plt
import numpy as np

def train_model(model, dataset, epochs=10, batch_size=64, lr=1e-5, device='cpu', logger=None):
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=min(4, os.cpu_count()))
    model.to(device)
    
    # Calculate class weights for balanced loss
    all_labels = torch.tensor([label for _, label in dataset])
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
    epoch_losses = []
    prev_loss = float('inf')
    
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        # Record epoch metrics
        epoch_loss = total_loss / len(dataloader)
        epoch_losses.append(epoch_loss)
        
        # Adjust learning rate based on loss improvement
        if epoch > 0:
            improvement = (prev_loss - epoch_loss) / prev_loss
            if improvement < 0.01:  # If improvement is less than 1%
                for param_group in optimizer.param_groups:
                    param_group['lr'] *= 0.8  # Reduce learning rate by 20%
                logger.info(f"Reducing learning rate to {optimizer.param_groups[0]['lr']:.2e}")
        
        prev_loss = epoch_loss
        logger.info(f"[Epoch {epoch+1}/{epochs}] Loss: {epoch_loss:.4f}")
    
    # Save training plot
    current_dir = os.path.dirname(os.path.abspath(__file__))
    results_dir = os.path.join(current_dir, 'results')
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
    
    # Plot training loss
    plt.figure(figsize=(8, 6))
    plt.plot(range(1, epochs + 1), epoch_losses, marker='o')
    plt.title('Training Loss Over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid(True)
    
    plots_path = os.path.join(results_dir, 'training_loss.png')
    plt.savefig(plots_path)
    logger.info(f"Saved training loss plot to {plots_path}")
    
    return model
