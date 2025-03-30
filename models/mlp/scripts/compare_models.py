"""
Script to compare regular and adversarially trained models.
"""
import os
import argparse
import torch
import numpy as np
import matplotlib.pyplot as plt
import json
import datetime
from ..src.model import MLPModel
from ..src.dataset import BreastHistopathologyDataset
from ..src.attacks import FGSM, evaluate_attack
from ..src.logger import setup_logger
from torch.utils.data import DataLoader

def load_model(checkpoint_path, device, logger=None):
    """Load a model from a checkpoint."""
    model = MLPModel()
    
    # Log parameter count
    if logger:
        num_params = model.count_parameters()
        logger.info(f"Model parameters: {num_params:,} ({num_params/(1000000):.2f}M)")
    
    # Set weights_only=False to avoid the unpickling error in PyTorch 2.6+
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    
    # Get model metadata
    is_adversarial = 'adversarial_training' in checkpoint
    adv_params = checkpoint.get('adversarial_training', {})
    
    return model, is_adversarial, adv_params

def compare_models(standard_model_path, adv_model_path, data_path, 
                  epsilons=[0.01, 0.05, 0.1, 0.2], batch_size=64):
    """
    Compare standard and adversarially trained models against various attack strengths.
    
    Args:
        standard_model_path: Path to standard trained model checkpoint
        adv_model_path: Path to adversarially trained model checkpoint
        data_path: Path to dataset
        epsilons: List of attack strengths to evaluate
        batch_size: Batch size for evaluation
    """
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 
                         'mps' if torch.backends.mps.is_available() else 
                         'cpu')
    
    # Setup logger
    current_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    log_dir = os.path.join(current_dir, 'logs')
    os.makedirs(log_dir, exist_ok=True)
    logger = setup_logger(os.path.join(log_dir, 'model_comparison.log'))
    
    logger.info(f"Comparing models on device: {device}")
    logger.info(f"Standard model path: {standard_model_path}")
    logger.info(f"Adversarial model path: {adv_model_path}")
    
    # Load models
    standard_model, is_std_adv, std_adv_params = load_model(standard_model_path, device, logger)
    adv_model, is_adv, adv_params = load_model(adv_model_path, device, logger)
    
    if not is_adv:
        logger.warning("The provided adversarial model does not have adversarial training metadata.")
    
    logger.info(f"Standard model: {'Adversarially trained' if is_std_adv else 'Standard trained'}")
    logger.info(f"Adversarial model: {'Adversarially trained' if is_adv else 'Standard trained'}")
    if is_adv:
        logger.info(f"Adversarial training parameters: epsilon={adv_params.get('epsilon')}, mix_ratio={adv_params.get('mix_ratio')}")
    
    # Load test dataset
    test_dataset = BreastHistopathologyDataset(
        root_dir=data_path,
        train=False,
        val_ratio=0.2
    )
    logger.info(f"Test dataset size: {len(test_dataset)}")
    
    # Create criterion for attack
    criterion = torch.nn.CrossEntropyLoss()
    
    # Initialize results dictionary
    results = {
        'standard_model': {
            'clean_accuracy': None,
            'adversarial_accuracy': {}
        },
        'adversarial_model': {
            'clean_accuracy': None,
            'adversarial_accuracy': {}
        }
    }
    
    # Evaluate clean accuracy for both models
    clean_accuracy_std = evaluate_clean_accuracy(standard_model, test_dataset, batch_size, device)
    clean_accuracy_adv = evaluate_clean_accuracy(adv_model, test_dataset, batch_size, device)
    
    # Convert to Python native float for JSON serialization
    results['standard_model']['clean_accuracy'] = float(clean_accuracy_std)
    results['adversarial_model']['clean_accuracy'] = float(clean_accuracy_adv)
    
    logger.info(f"Clean accuracy - Standard model: {clean_accuracy_std:.4f}")
    logger.info(f"Clean accuracy - Adversarial model: {clean_accuracy_adv:.4f}")
    
    # Evaluate adversarial accuracy for various epsilon values
    std_model_adv_accuracies = []
    adv_model_adv_accuracies = []
    
    for epsilon in epsilons:
        logger.info(f"Evaluating FGSM attack with epsilon={epsilon}")
        
        # Create FGSM attacks
        fgsm_attack_std = FGSM(standard_model, criterion=criterion, epsilon=epsilon)
        fgsm_attack_adv = FGSM(adv_model, criterion=criterion, epsilon=epsilon)
        
        # Evaluate attacks on standard model
        logger.info(f"Evaluating standard model against FGSM (ε={epsilon})")
        std_metrics = evaluate_attack(
            model=standard_model,
            dataset=test_dataset,
            attack=fgsm_attack_std,
            batch_size=batch_size,
            device=device,
            logger=logger,
            save_results=False
        )
        # Convert to native Python types for JSON serialization
        adv_acc_std = float(std_metrics['adversarial_accuracy'])
        std_model_adv_accuracies.append(adv_acc_std)
        results['standard_model']['adversarial_accuracy'][str(epsilon)] = adv_acc_std
        
        # Evaluate attacks on adversarial model
        logger.info(f"Evaluating adversarial model against FGSM (ε={epsilon})")
        adv_metrics = evaluate_attack(
            model=adv_model,
            dataset=test_dataset,
            attack=fgsm_attack_adv,
            batch_size=batch_size,
            device=device,
            logger=logger,
            save_results=False
        )
        # Convert to native Python types for JSON serialization
        adv_acc_adv = float(adv_metrics['adversarial_accuracy'])
        adv_model_adv_accuracies.append(adv_acc_adv)
        results['adversarial_model']['adversarial_accuracy'][str(epsilon)] = adv_acc_adv
    
    # Save results to JSON
    results_dir = os.path.join(current_dir, 'results/comparisons')
    os.makedirs(results_dir, exist_ok=True)
    
    import datetime
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = os.path.join(results_dir, f'model_comparison_{timestamp}.json')
    
    # Ensure all values are JSON serializable
    def convert_to_serializable(obj):
        if isinstance(obj, (np.integer, np.floating, np.bool_)):
            return obj.item()
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {k: convert_to_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_to_serializable(item) for item in obj]
        else:
            return obj
    
    # Convert all values to JSON serializable types
    results_serializable = convert_to_serializable(results)
    
    with open(results_file, 'w') as f:
        json.dump(results_serializable, f, indent=4)
    
    logger.info(f"Results saved to {results_file}")
    
    # Plot comparison
    plt.figure(figsize=(10, 6))
    plt.plot(epsilons, std_model_adv_accuracies, 'o-', label='Standard Model')
    plt.plot(epsilons, adv_model_adv_accuracies, 's-', label='Adversarially Trained Model')
    plt.axhline(y=clean_accuracy_std, color='b', linestyle='--', alpha=0.3, label='Standard Model (Clean)')
    plt.axhline(y=clean_accuracy_adv, color='orange', linestyle='--', alpha=0.3, label='Adversarial Model (Clean)')
    
    plt.xlabel('Epsilon (Attack Strength)')
    plt.ylabel('Accuracy')
    plt.title('Model Robustness Comparison')
    plt.legend()
    plt.grid(True)
    
    plot_path = os.path.join(results_dir, f'model_comparison_plot_{timestamp}.png')
    plt.savefig(plot_path)
    logger.info(f"Comparison plot saved to {plot_path}")
    
    return results

def evaluate_clean_accuracy(model, dataset, batch_size=64, device='cpu'):
    """Evaluate model accuracy on clean data."""
    model.eval()
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    
    correct = 0
    total = 0
    
    with torch.no_grad():
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    return correct / total

def main():
    parser = argparse.ArgumentParser(description='Compare standard and adversarially trained models')
    parser.add_argument('--standard_model', type=str, required=True, 
                        help='Path to standard trained model checkpoint')
    parser.add_argument('--adversarial_model', type=str, required=True, 
                        help='Path to adversarially trained model checkpoint')
    parser.add_argument('--data_path', type=str, default='data/', 
                        help='Path to the dataset directory')
    parser.add_argument('--epsilons', type=float, nargs='+', default=[0.01, 0.05, 0.1, 0.2], 
                        help='Epsilon values for FGSM attack (attack strength)')
    parser.add_argument('--batch_size', type=int, default=64, 
                        help='Batch size for evaluation')
    
    args = parser.parse_args()
    
    compare_models(
        standard_model_path=args.standard_model,
        adv_model_path=args.adversarial_model,
        data_path=args.data_path,
        epsilons=args.epsilons,
        batch_size=args.batch_size
    )

if __name__ == '__main__':
    main()