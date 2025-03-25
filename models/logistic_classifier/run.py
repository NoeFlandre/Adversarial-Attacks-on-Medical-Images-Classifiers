from model import LogisticRegressionModel
from dataset import BreastHistopathologyDataset
from train import train_model
from eval import evaluate_model
from logger import setup_logger
import torch
import os
import argparse
import datetime

if __name__ == '__main__':
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Train and evaluate logistic regression model for breast histopathology')
    parser.add_argument('--data_path', type=str, default='data/', help='Path to the dataset')
    parser.add_argument('--epochs', type=int, default=10, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=1024, help='Batch size for training and evaluation')
    parser.add_argument('--learning_rate', type=float, default=1e-5, help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-5, help='Weight decay for Adam optimizer')
    parser.add_argument('--val_ratio', type=float, default=0.2, help='Ratio of validation data')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility')
    args = parser.parse_args()
    
    # Set random seeds for reproducibility
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
    
    # Set up logger
    logger = setup_logger()
    
    logger.info("Starting Logistic Regression classifier training and evaluation")
    logger.info(f"Configuration:")
    logger.info(f"  - Epochs: {args.epochs}")
    logger.info(f"  - Batch size: {args.batch_size}")
    logger.info(f"  - Learning rate: {args.learning_rate}")
    logger.info(f"  - Weight decay: {args.weight_decay}")
    logger.info(f"  - Validation ratio: {args.val_ratio}")
    
    logger.info(f"\nLoading datasets from {args.data_path}")
    
    # Create separate training and validation datasets
    train_dataset = BreastHistopathologyDataset(args.data_path, train=True, 
                                               val_ratio=args.val_ratio, seed=args.seed)
    val_dataset = BreastHistopathologyDataset(args.data_path, train=False, 
                                             val_ratio=args.val_ratio, seed=args.seed)
    
    logger.info(f"Training set size: {len(train_dataset)}")
    logger.info(f"Validation set size: {len(val_dataset)}")
    
    logger.info("\nInitializing Logistic Regression model")
    model = LogisticRegressionModel()

    device = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
    logger.info(f"Using device: {device}")
    
    logger.info("\nStarting model training")
    train_model(model, train_dataset, val_dataset, epochs=args.epochs, batch_size=args.batch_size, 
                lr=args.learning_rate, device=device, logger=logger)
    
    logger.info("\nStarting model evaluation on validation set")
    metrics = evaluate_model(model, val_dataset, batch_size=args.batch_size, 
                            device=device, logger=logger, save_results=True)
    
    # Save the model
    current_dir = os.path.dirname(os.path.abspath(__file__))
    model_dir = os.path.join(current_dir, 'checkpoints')
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    model_path = os.path.join(model_dir, f'logistic_regression_model_{timestamp}.pth')
    torch.save({
        'model_state_dict': model.state_dict(),
        'metrics': metrics,
        'config': vars(args)
    }, model_path)
    logger.info(f"Model and metrics saved to {model_path}")
    
    logger.info("\nCompleted training and evaluation")
