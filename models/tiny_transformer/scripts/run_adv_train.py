from ..src.model import TinyTransformer
from ..src.dataset import BreastHistopathologyDataset
from .train_adversarial import train_model_adversarial
from .eval import evaluate_model
from ..src.logger import setup_logger
import torch
import os
import argparse
import datetime
import json

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Train Tiny Transformer model with adversarial training')
    parser.add_argument('--data_path', type=str, default='data/', help='Path to the dataset')
    parser.add_argument('--epochs', type=int, default=1, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=1024, help='Batch size for training and evaluation')
    parser.add_argument('--learning_rate', type=float, default=1e-5, help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-5, help='Weight decay for Adam optimizer')
    parser.add_argument('--val_ratio', type=float, default=0.2, help='Ratio of validation data')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility')
    parser.add_argument('--epsilon', type=float, default=0.05, help='Epsilon for FGSM adversarial training')
    parser.add_argument('--mix_ratio', type=float, default=0.5, help='Ratio of adversarial examples in each batch')
    args = parser.parse_args()
    
    # Set random seeds for reproducibility
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
    
    # Set up logger with a distinct folder for adversarial training logs
    log_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'logs/adversarial_training.log')
    os.makedirs(os.path.dirname(log_dir), exist_ok=True)
    logger = setup_logger(log_dir)
    
    logger.info("Starting Tiny Transformer classifier with adversarial training")
    logger.info(f"Configuration:")
    logger.info(f"  - Epochs: {args.epochs}")
    logger.info(f"  - Batch size: {args.batch_size}")
    logger.info(f"  - Learning rate: {args.learning_rate}")
    logger.info(f"  - Weight decay: {args.weight_decay}")
    logger.info(f"  - Validation ratio: {args.val_ratio}")
    logger.info(f"  - FGSM epsilon: {args.epsilon}")
    logger.info(f"  - Mix ratio: {args.mix_ratio}")
    
    logger.info(f"\nLoading datasets from {args.data_path}")
    
    # Create separate training and validation datasets
    train_dataset = BreastHistopathologyDataset(args.data_path, train=True, 
                                               val_ratio=args.val_ratio, seed=args.seed)
    val_dataset = BreastHistopathologyDataset(args.data_path, train=False, 
                                             val_ratio=args.val_ratio, seed=args.seed)
    
    logger.info(f"Training set size: {len(train_dataset)}")
    logger.info(f"Validation set size: {len(val_dataset)}")
    
    logger.info("\nInitializing Tiny Transformer model")
    model = TinyTransformer()
    
    # Log parameter count
    num_params = model.count_parameters()
    logger.info(f"Model parameters: {num_params:,} ({num_params/(1000000):.2f}M)")

    device = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
    logger.info(f"Using device: {device}")
    
    logger.info("\nStarting model training with adversarial examples")
    train_model_adversarial(model, train_dataset, val_dataset, 
                           epochs=args.epochs, 
                           batch_size=args.batch_size, 
                           lr=args.learning_rate, 
                           device=device, 
                           logger=logger,
                           epsilon=args.epsilon,
                           mix_ratio=args.mix_ratio)
    
    logger.info("\nStarting model evaluation on validation set")
    metrics = evaluate_model(model, val_dataset, batch_size=args.batch_size, 
                            device=device, logger=logger, save_results=True)
    
    # Save the model with adversarial training indicator in the name
    current_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    model_dir = os.path.join(current_dir, 'checkpoints')
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    model_path = os.path.join(model_dir, f'tiny_transformer_adversarial_eps{args.epsilon}_mix{args.mix_ratio}_{timestamp}.pth')
    
    # Save model and metadata
    torch.save({
        'model_state_dict': model.state_dict(),
        'metrics': metrics,
        'config': vars(args),
        'adversarial_training': {
            'epsilon': args.epsilon,
            'mix_ratio': args.mix_ratio
        }
    }, model_path)
    logger.info(f"Model and metrics saved to {model_path}")
    
    # Save results to a dedicated adversarial training directory
    results_dir = os.path.join(current_dir, 'results/adversarial_training')
    os.makedirs(results_dir, exist_ok=True)
    
    # Convert NumPy arrays to floats for JSON serialization
    metrics_serializable = {}
    for key, value in metrics.items():
        if key == 'confusion_matrix':
            # Convert confusion matrix to a list of lists
            metrics_serializable[key] = value.tolist()
        elif hasattr(value, 'tolist'):
            # For any NumPy values, convert to Python native types
            metrics_serializable[key] = float(value)
        else:
            metrics_serializable[key] = value
    
    # Make config serializable by converting any non-serializable objects
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
        'model_type': 'tiny_transformer_adversarial',
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