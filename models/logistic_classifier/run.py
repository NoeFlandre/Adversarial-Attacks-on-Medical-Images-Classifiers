from model import LogisticRegressionModel
from dataset import BreastHistopathologyDataset
from train import train_model
from eval import evaluate_model
from logger import setup_logger
import torch
import os
import argparse

if __name__ == '__main__':
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Train and evaluate logistic regression model for breast histopathology')
    parser.add_argument('--data_path', type=str, default='data/', help='Path to the dataset')
    parser.add_argument('--epochs', type=int, default=5, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size for training and evaluation')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning rate for optimizer')
    parser.add_argument('--val_ratio', type=float, default=0.2, help='Ratio of validation data')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility')
    args = parser.parse_args()
    
    # Set up logger
    logger = setup_logger()
    
    logger.info("Starting Logistic Regression classifier training and evaluation")
    logger.info(f"Loading datasets from {args.data_path}")
    
    # Create separate training and validation datasets
    train_dataset = BreastHistopathologyDataset(args.data_path, train=True, 
                                               val_ratio=args.val_ratio, seed=args.seed)
    val_dataset = BreastHistopathologyDataset(args.data_path, train=False, 
                                             val_ratio=args.val_ratio, seed=args.seed)
    
    logger.info(f"Training set size: {len(train_dataset)}")
    logger.info(f"Validation set size: {len(val_dataset)}")
    
    logger.info("Initializing Logistic Regression model")
    model = LogisticRegressionModel()

    device = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
    logger.info(f"Using device: {device}")
    
    logger.info("Starting model training")
    train_model(model, train_dataset, epochs=args.epochs, batch_size=args.batch_size, 
                lr=args.learning_rate, device=device, logger=logger)
    
    logger.info("Starting model evaluation on validation set")
    metrics = evaluate_model(model, val_dataset, batch_size=args.batch_size, 
                            device=device, logger=logger, save_results=True)
    
    # Save the model
    current_dir = os.path.dirname(os.path.abspath(__file__))
    model_dir = os.path.join(current_dir, 'checkpoints')
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    
    model_path = os.path.join(model_dir, 'logistic_regression_model.pth')
    torch.save(model.state_dict(), model_path)
    logger.info(f"Model saved to {model_path}")
    
    logger.info("Completed training and evaluation")
