import argparse
import torch
import os
import matplotlib.pyplot as plt
import numpy as np
from model import LogisticRegressionModel
from logger import setup_logger
import json

def plot_epsilon_vs_accuracy(results_dir, output_path, logger):
    """
    Plot accuracy vs epsilon values for FGSM attack.
    """
    # Find all FGSM metrics files
    metrics_files = [f for f in os.listdir(results_dir) if f.startswith('FGSM_metrics_eps') and f.endswith('.json')]
    
    if not metrics_files:
        logger.error(f"No FGSM metrics files found in {results_dir}")
        return
    
    # Extract epsilon values and corresponding metrics
    epsilons = []
    clean_accuracies = []
    adv_accuracies = []
    success_rates = []
    
    for file in metrics_files:
        with open(os.path.join(results_dir, file), 'r') as f:
            metrics = json.load(f)
            
            epsilons.append(metrics['epsilon'])
            clean_accuracies.append(metrics['clean_accuracy'])
            adv_accuracies.append(metrics['adversarial_accuracy'])
            success_rates.append(metrics['attack_success_rate_correct'])
    
    # Sort by epsilon value
    sorted_indices = np.argsort(epsilons)
    epsilons = [epsilons[i] for i in sorted_indices]
    clean_accuracies = [clean_accuracies[i] for i in sorted_indices]
    adv_accuracies = [adv_accuracies[i] for i in sorted_indices]
    success_rates = [success_rates[i] for i in sorted_indices]
    
    # Plot accuracy vs epsilon
    plt.figure(figsize=(10, 6))
    plt.plot(epsilons, clean_accuracies, 'o-', label='Clean Accuracy')
    plt.plot(epsilons, adv_accuracies, 'o-', label='Adversarial Accuracy')
    plt.plot(epsilons, success_rates, 'o-', label='Attack Success Rate')
    plt.xlabel('Epsilon (Attack Strength)')
    plt.ylabel('Rate')
    plt.title('FGSM Attack Impact vs Epsilon Value')
    plt.legend()
    plt.grid(True)
    plt.savefig(output_path)
    logger.info(f"Saved epsilon vs accuracy plot to {output_path}")

def visualize_adversarial_perturbations(adv_dataset_path, output_path, num_samples=5, logger=None):
    """
    Visualize adversarial examples and the perturbations added to them.
    """
    # Load adversarial dataset
    adv_data = torch.load(adv_dataset_path)
    adversarial_images = adv_data['adversarial_images']
    labels = adv_data['labels']
    
    # Create directory for visualizations if it doesn't exist
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Get a random sample of images
    indices = np.random.choice(len(adversarial_images), min(num_samples, len(adversarial_images)), replace=False)
    
    # Create visualization
    plt.figure(figsize=(12, 3 * num_samples))
    
    class_names = ['Non-IDC', 'IDC']
    
    for i, idx in enumerate(indices):
        adv_img = adversarial_images[idx].permute(1, 2, 0).numpy()
        label = labels[idx].item()
        
        plt.subplot(num_samples, 1, i + 1)
        plt.imshow(adv_img)
        plt.title(f"Adversarial Example (True class: {class_names[label]})")
        plt.axis('off')
    
    plt.tight_layout()
    plt.savefig(output_path)
    if logger:
        logger.info(f"Saved adversarial examples visualization to {output_path}")

def main():
    parser = argparse.ArgumentParser(description='Visualize adversarial attacks results')
    parser.add_argument('--results_dir', type=str, default=None, 
                        help='Path to adversarial results directory')
    parser.add_argument('--adv_dataset', type=str, default=None,
                        help='Path to saved adversarial dataset')
    parser.add_argument('--num_samples', type=int, default=5,
                        help='Number of samples to visualize')
    args = parser.parse_args()
    
    # Setup paths
    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    if not args.results_dir:
        args.results_dir = os.path.join(current_dir, 'results', 'adversarial')
    
    # Setup logger
    log_dir = os.path.join(current_dir, 'logs')
    os.makedirs(log_dir, exist_ok=True)
    logger = setup_logger(os.path.join(log_dir, 'visualize_attacks.log'))
    
    logger.info(f"Visualizing adversarial attack results from {args.results_dir}")
    
    # Create output directory
    visualization_dir = os.path.join(current_dir, 'results', 'visualizations')
    os.makedirs(visualization_dir, exist_ok=True)
    
    # Plot epsilon vs accuracy
    if os.path.exists(args.results_dir):
        plot_epsilon_vs_accuracy(
            results_dir=args.results_dir,
            output_path=os.path.join(visualization_dir, 'epsilon_vs_accuracy.png'),
            logger=logger
        )
    
    # Visualize adversarial examples
    if args.adv_dataset and os.path.exists(args.adv_dataset):
        visualize_adversarial_perturbations(
            adv_dataset_path=args.adv_dataset,
            output_path=os.path.join(visualization_dir, 'adversarial_examples.png'),
            num_samples=args.num_samples,
            logger=logger
        )
    elif os.path.exists(args.results_dir):
        # Look for adversarial datasets in results directory
        for root, dirs, files in os.walk(args.results_dir):
            for file in files:
                if file == 'adversarial_dataset.pt':
                    adv_dataset_path = os.path.join(root, file)
                    output_path = os.path.join(visualization_dir, f"{os.path.basename(root)}_examples.png")
                    visualize_adversarial_perturbations(
                        adv_dataset_path=adv_dataset_path,
                        output_path=output_path,
                        num_samples=args.num_samples,
                        logger=logger
                    )

if __name__ == '__main__':
    main()