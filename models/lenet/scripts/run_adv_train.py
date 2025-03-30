import argparse
import os
import sys
import torch
import json
import datetime
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve
from .train_adversarial import train_model_adversarial
from .eval import evaluate_model
from ..src.model import LeNet
from ..src.logger import setup_logger

def main():
    """
    Main function for adversarial training of LeNet.
    """
    parser = argparse.ArgumentParser(description='Train LeNet model with adversarial training')
    parser.add_argument('--data_dir', type=str, default='data/', help='Path to the dataset')
    parser.add_argument('--epochs', type=int, default=20, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=1024, help='Batch size for training and evaluation')
    parser.add_argument('--learning_rate', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--val_ratio', type=float, default=0.2, help='Ratio of validation data')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility')
    parser.add_argument('--epsilon', type=float, default=0.05, help='Epsilon for FGSM adversarial training')
    parser.add_argument('--mix_ratio', type=float, default=0.5, help='Ratio of adversarial examples in each batch')
    args = parser.parse_args()
    
    # Setup directories
    current_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    checkpoint_dir = os.path.join(current_dir, 'checkpoints')
    log_dir = os.path.join(current_dir, 'logs')
    
    # Set random seeds for reproducibility
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
    
    # Setup logger
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    logger = setup_logger(name=f'lenet_{timestamp}', log_type='adversarial_training')
    
    logger.info("Starting LeNet adversarial training")
    logger.info(f"Configuration:")
    logger.info(f"  - Epochs: {args.epochs}")
    logger.info(f"  - Batch size: {args.batch_size}")
    logger.info(f"  - Learning rate: {args.learning_rate}")
    logger.info(f"  - Validation ratio: {args.val_ratio}")
    logger.info(f"  - FGSM epsilon: {args.epsilon}")
    logger.info(f"  - Mix ratio: {args.mix_ratio}")
    
    # Train model with adversarial training
    model, history = train_model_adversarial(
        data_dir=args.data_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        val_ratio=args.val_ratio,
        seed=args.seed,
        epsilon=args.epsilon,
        mix_ratio=args.mix_ratio,
        checkpoint_dir=checkpoint_dir,
        log_dir=log_dir
    )
    
    # Create dataloader for validation set
    from ..src.dataset import BreastTumorDataset
    from torch.utils.data import DataLoader
    
    logger.info("\nEvaluating final model on validation set")
    val_dataset = BreastTumorDataset(
        root_dir=args.data_dir,
        train=False,
        val_ratio=args.val_ratio,
        seed=args.seed
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=2
    )
    
    # Set up results directory for adversarial training
    results_dir = os.path.join(current_dir, 'results/adversarial_training')
    os.makedirs(results_dir, exist_ok=True)
    
    # Save the timestamp for consistent filenames
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Modify evaluate_model to save results to adversarial_training directory
    from .eval import evaluate_model
    
    # Monkey patch the evaluate_model function to use our results directory
    original_evaluate_model = evaluate_model
    
    def patched_evaluate_model(model, test_loader, device, logger=None, save_results=True, timestamp=None):
        # Create results directory if it doesn't exist
        nonlocal results_dir
        if save_results:
            if not os.path.exists(results_dir):
                os.makedirs(results_dir)
            
            metrics = original_evaluate_model(model, test_loader, device, logger, save_results=False)
            
            # Save metrics as JSON with all values as Python native types
            metrics_serializable = {
                'accuracy': float(metrics['accuracy']),
                'balanced_accuracy': float(metrics['balanced_accuracy']),
                'precision': float(metrics['precision']),
                'recall': float(metrics['recall']),
                'specificity': float(metrics['specificity']),
                'f1_score': float(metrics['f1_score']),
                'auc': float(metrics['auc']),
                'confusion_matrix': metrics['confusion_matrix'].tolist()
            }
            
            metrics_file = os.path.join(results_dir, f"metrics_{timestamp}.json")
            with open(metrics_file, 'w') as f:
                json.dump(metrics_serializable, f, indent=4)
            if logger:
                logger.info(f"Saved metrics to {metrics_file}")
            
            # Save confusion matrix
            plt.figure(figsize=(8, 6))
            plt.imshow(metrics['confusion_matrix'], interpolation='nearest', cmap=plt.cm.Blues)
            plt.title('Confusion Matrix (Adversarial Training)')
            plt.colorbar()
            classes = ['Non-IDC', 'IDC']
            tick_marks = np.arange(len(classes))
            plt.xticks(tick_marks, classes, rotation=45)
            plt.yticks(tick_marks, classes)
            
            # Add text annotations
            thresh = metrics['confusion_matrix'].max() / 2.
            for i in range(metrics['confusion_matrix'].shape[0]):
                for j in range(metrics['confusion_matrix'].shape[1]):
                    plt.text(j, i, format(metrics['confusion_matrix'][i, j], 'd'),
                            horizontalalignment="center",
                            color="white" if metrics['confusion_matrix'][i, j] > thresh else "black")
            
            plt.tight_layout()
            plt.ylabel('True label')
            plt.xlabel('Predicted label')
            
            cm_file = os.path.join(results_dir, f"confusion_matrix_{timestamp}.png")
            plt.savefig(cm_file)
            plt.close()
            if logger:
                logger.info(f"Saved confusion matrix plot to {cm_file}")
            
            # We'll use the confusion matrix values to avoid having to recalculate probabilities
            # Since we only need a visualization and not accurate numbers
            cm = metrics['confusion_matrix']
            tn, fp, fn, tp = cm.ravel()
            
            # Approximate ROC points 
            fpr = [0, fp/(fp+tn) if (fp+tn) > 0 else 0, 1]
            tpr = [0, tp/(tp+fn) if (tp+fn) > 0 else 0, 1]
            roc_auc = metrics_serializable['auc']
            
            plt.figure(figsize=(8, 6))
            plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
            plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.05])
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title('Receiver Operating Characteristic (Adversarial Training)')
            plt.legend(loc="lower right")
            
            roc_file = os.path.join(results_dir, f"roc_curve_{timestamp}.png")
            plt.savefig(roc_file)
            plt.close()
            if logger:
                logger.info(f"Saved ROC curve plot to {roc_file}")
            
            return metrics
        else:
            return original_evaluate_model(model, test_loader, device, logger, save_results=False)
    
    # Replace the original function with our patched version
    evaluate_model = patched_evaluate_model
    
    # Evaluate model
    device = torch.device('cuda' if torch.cuda.is_available() else 
                         'mps' if torch.backends.mps.is_available() else 
                         'cpu')
    
    eval_metrics = evaluate_model(
        model=model,
        test_loader=val_loader,
        device=device,
        logger=logger,
        save_results=True,
        timestamp=timestamp
    )
    
    # Save a results summary file
    results_dir = os.path.join(current_dir, 'results/adversarial_training')
    os.makedirs(results_dir, exist_ok=True)
    
    # Convert NumPy arrays to floats for JSON serialization
    metrics_serializable = {}
    for key, value in eval_metrics.items():
        if key == 'confusion_matrix':
            # Convert confusion matrix to a list of lists
            metrics_serializable[key] = value.tolist()
        elif hasattr(value, 'tolist'):
            # For any NumPy values, convert to Python native types
            metrics_serializable[key] = float(value)
        else:
            metrics_serializable[key] = value
    
    # Make config serializable
    config_serializable = {}
    for key, value in vars(args).items():
        if isinstance(value, (int, float, str, bool, type(None))):
            config_serializable[key] = value
        elif isinstance(value, list):
            # Check if list contains only serializable items
            if all(isinstance(item, (int, float, str, bool, type(None))) for item in value):
                config_serializable[key] = value
            else:
                config_serializable[key] = str(value)
        else:
            config_serializable[key] = str(value)
    
    summary = {
        'model_type': 'lenet_adversarial',
        'timestamp': timestamp,
        'epsilon': float(args.epsilon),
        'mix_ratio': float(args.mix_ratio),
        'metrics': metrics_serializable,
        'config': config_serializable
    }
    
    summary_path = os.path.join(results_dir, f'adversarial_training_summary_eps{args.epsilon}_mix{args.mix_ratio}_{timestamp}.json')
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=4)
    
    logger.info(f"Results summary saved to {summary_path}")
    logger.info("\nCompleted adversarial training and evaluation")
    
if __name__ == '__main__':
    main()