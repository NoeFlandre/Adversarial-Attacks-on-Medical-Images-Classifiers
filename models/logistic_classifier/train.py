import torch
from torch.utils.data import DataLoader
from torch import nn, optim
import os
import matplotlib.pyplot as plt
import numpy as np

def train_model(model, dataset, epochs=5, batch_size=64, lr=1e-3, device='cpu', logger=None):
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    model.to(device)
    optimizer = optim.SGD(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    logger.info(f"Training with batch size: {batch_size}, learning rate: {lr}, epochs: {epochs}")
    
    # Track metrics for plotting
    epoch_losses = []
    
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
        
        # Record epoch loss
        epoch_loss = total_loss / len(dataloader)
        epoch_losses.append(epoch_loss)
        
        logger.info(f"[Epoch {epoch+1}/{epochs}] Loss: {epoch_loss:.4f}")
    
    # Save training loss plot
    current_dir = os.path.dirname(os.path.abspath(__file__))
    results_dir = os.path.join(current_dir, 'results')
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
    
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, epochs + 1), epoch_losses, marker='o')
    plt.title('Training Loss Over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid(True)
    
    loss_plot_path = os.path.join(results_dir, 'training_loss.png')
    plt.savefig(loss_plot_path)
    logger.info(f"Saved training loss plot to {loss_plot_path}")
    
    return model
