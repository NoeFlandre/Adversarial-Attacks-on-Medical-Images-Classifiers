import torch
import torch.nn as nn
import os
import matplotlib.pyplot as plt
import numpy as np
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms
from PIL import Image
import json
import datetime


class FGSM:
    """
    Fast Gradient Sign Method (FGSM) adversarial attack implementation.
    """
    def __init__(self, model, criterion=None, epsilon=0.1):
        """
        Initialize FGSM attack.
        
        Args:
            model: The model to attack
            criterion: Loss function (default: CrossEntropyLoss)
            epsilon: Attack strength parameter (default: 0.1)
        """
        self.model = model
        self.criterion = criterion if criterion is not None else nn.CrossEntropyLoss()
        self.epsilon = epsilon
        
    def generate(self, images, labels, targeted=False):
        """
        Generate adversarial examples using FGSM.
        
        Args:
            images: Input images tensor
            labels: Ground truth labels
            targeted: If True, perform targeted attack (default: False). In our case we are dealing with binary classification, so targeted and untargeted attacks are the same.
            
        Returns:
            Adversarial examples
        """
        # Set requires_grad
        images.requires_grad = True
        
        # Forward pass
        outputs = self.model(images)
        
        # Calculate loss
        if targeted:
            # For targeted attack, minimize loss w.r.t. target
            cost = -self.criterion(outputs, labels)
        else:
            # For untargeted attack, maximize loss w.r.t. true label
            cost = self.criterion(outputs, labels)
        
        # Backward pass
        self.model.zero_grad()
        cost.backward()
        
        # Create adversarial examples
        grad_sign = images.grad.data.sign()
        perturbed_images = images + self.epsilon * grad_sign
        
        # Clamp to [0,1] range
        perturbed_images = torch.clamp(perturbed_images, 0, 1)
        
        return perturbed_images
    

def evaluate_attack(model, dataset, attack, batch_size=64, device='cpu', logger=None, save_results=True):
    """
    Evaluate a model's robustness against an adversarial attack.
    
    Args:
        model: The model to evaluate
        dataset: The dataset to use for evaluation
        attack: The attack to use (must have a generate method)
        batch_size: Batch size for evaluation
        device: Device to run evaluation on
        logger: Logger for recording results
        save_results: Whether to save results to disk
        
    Returns:
        Dictionary of evaluation metrics
    """
    model.eval()
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    
    # Initialize lists to store predictions and ground truth
    clean_preds = []
    adversarial_preds = []
    all_labels = []
    
    # Sample images for visualization
    vis_images = []
    vis_adversarial = []
    vis_labels = []
    vis_clean_preds = []
    vis_adv_preds = []
    num_samples = min(5, len(dataset))
    sample_indices = np.random.choice(len(dataset), num_samples, replace=False)
    
    logger.info(f"Evaluating {attack.__class__.__name__} attack with epsilon={attack.epsilon}")
    
    for i, (images, labels) in enumerate(dataloader):
        images, labels = images.to(device), labels.to(device)
        
        # Generate adversarial examples
        adversarial_images = attack.generate(images, labels)
        
        # Get predictions for clean and adversarial images
        with torch.no_grad():
            # Clean predictions
            clean_outputs = model(images)
            clean_batch_preds = clean_outputs.argmax(dim=1)
            
            # Adversarial predictions
            adv_outputs = model(adversarial_images)
            adv_batch_preds = adv_outputs.argmax(dim=1)
        
        # Store batch results
        clean_preds.extend(clean_batch_preds.cpu().numpy())
        adversarial_preds.extend(adv_batch_preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
        
        # Store some examples for visualization
        if i == 0:
            for idx in range(min(num_samples, len(images))):
                vis_images.append(images[idx].cpu())
                vis_adversarial.append(adversarial_images[idx].detach().cpu())
                vis_labels.append(labels[idx].item())
                vis_clean_preds.append(clean_batch_preds[idx].item())
                vis_adv_preds.append(adv_batch_preds[idx].item())
    
    # Convert to numpy arrays for easier manipulation
    clean_preds = np.array(clean_preds)
    adversarial_preds = np.array(adversarial_preds)
    all_labels = np.array(all_labels)
    
    # Calculate metrics
    clean_accuracy = np.mean(clean_preds == all_labels)
    adversarial_accuracy = np.mean(adversarial_preds == all_labels)
    attack_success_rate = np.mean(clean_preds != adversarial_preds)
    attack_success_rate_correct = np.mean((clean_preds == all_labels) & (adversarial_preds != all_labels))
    
    # Log results
    logger.info(f"Attack evaluation results:")
    logger.info(f"  - Clean Accuracy: {clean_accuracy:.4f}")
    logger.info(f"  - Adversarial Accuracy: {adversarial_accuracy:.4f}")
    logger.info(f"  - Attack Success Rate: {attack_success_rate:.4f}")
    logger.info(f"  - Attack Success Rate (on correctly classified): {attack_success_rate_correct:.4f}")
    
    if save_results:
        # Create results directory if it doesn't exist
        current_dir = os.path.dirname(os.path.abspath(__file__))
        results_dir = os.path.join(current_dir, 'results', 'adversarial')
        os.makedirs(results_dir, exist_ok=True)
        
        # Create timestamp for unique filenames
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        attack_name = attack.__class__.__name__
        
        # Save metrics as JSON
        metrics = {
            'attack': attack_name,
            'epsilon': attack.epsilon,
            'clean_accuracy': float(clean_accuracy),
            'adversarial_accuracy': float(adversarial_accuracy),
            'attack_success_rate': float(attack_success_rate),
            'attack_success_rate_correct': float(attack_success_rate_correct)
        }
        
        metrics_file = os.path.join(results_dir, f"{attack_name}_metrics_eps{attack.epsilon}_{timestamp}.json")
        with open(metrics_file, 'w') as f:
            json.dump(metrics, f, indent=4)
        logger.info(f"Saved metrics to {metrics_file}")
        
        # Visualize examples
        fig, axs = plt.subplots(num_samples, 2, figsize=(8, 2*num_samples))
        
        class_names = ['Non-IDC', 'IDC']
        
        for i in range(num_samples):
            # Display clean image
            img = vis_images[i].detach().permute(1, 2, 0).numpy()
            axs[i, 0].imshow(img)
            clean_title = f"Clean: {class_names[vis_clean_preds[i]]}"
            if vis_clean_preds[i] == vis_labels[i]:
                clean_title += " ✓"
            else:
                clean_title += " ✗"
            axs[i, 0].set_title(clean_title)
            axs[i, 0].axis('off')
            
            # Display adversarial image
            adv_img = vis_adversarial[i].permute(1, 2, 0).numpy()
            axs[i, 1].imshow(adv_img)
            adv_title = f"Adv: {class_names[vis_adv_preds[i]]}"
            if vis_adv_preds[i] == vis_labels[i]:
                adv_title += " ✓"
            else:
                adv_title += " ✗"
            axs[i, 1].set_title(adv_title)
            axs[i, 1].axis('off')
        
        plt.tight_layout()
        vis_file = os.path.join(results_dir, f"{attack_name}_examples_eps{attack.epsilon}_{timestamp}.png")
        plt.savefig(vis_file)
        logger.info(f"Saved adversarial examples visualization to {vis_file}")
    
    return {
        'clean_accuracy': clean_accuracy,
        'adversarial_accuracy': adversarial_accuracy,
        'attack_success_rate': attack_success_rate,
        'attack_success_rate_correct': attack_success_rate_correct
    }


class AdversarialDataset(Dataset):
    """
    Dataset class for storing adversarial examples.
    """
    def __init__(self, original_dataset, attack, device='cpu'):
        """
        Generate and store adversarial examples for a dataset.
        
        Args:
            original_dataset: Original dataset
            attack: Attack to use for generating adversarial examples
            device: Device to run attack on
        """
        self.original_dataset = original_dataset
        self.adversarial_images = []
        self.labels = []
        self.transform = transforms.ToTensor()
        
        # Generate adversarial examples in batches
        dataloader = DataLoader(original_dataset, batch_size=64, shuffle=False)
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)
            adversarial_batch = attack.generate(images, labels)
            self.adversarial_images.append(adversarial_batch.detach().cpu())
            self.labels.append(labels.cpu())
        
        self.adversarial_images = torch.cat(self.adversarial_images)
        self.labels = torch.cat(self.labels)
    
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        return self.adversarial_images[idx], self.labels[idx]