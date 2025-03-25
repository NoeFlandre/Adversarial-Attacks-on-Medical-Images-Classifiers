import os
import argparse
from dataset import BreastHistopathologyDataset
from train import train_model
from eval import evaluate_model
from logger import setup_logger

def main():
    parser = argparse.ArgumentParser(description='Train and evaluate LeNet model for breast histopathology classification')
    parser.add_argument('--data_dir', type=str, default='data/', help='Path to the dataset directory')
    parser.add_argument('--epochs', type=int, default=20, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=1024, help='Batch size for training')
    parser.add_argument('--lr', type=float, default=1e-4, help='Initial learning rate')
    parser.add_argument('--dropout_rate', type=float, default=0.5, help='Dropout rate')
    parser.add_argument('--val_ratio', type=float, default=0.2, help='Ratio of validation data')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility')
    parser.add_argument('--patience', type=int, default=5, help='Number of epochs to wait before early stopping')
    parser.add_argument('--eval_freq', type=int, default=1, help='Frequency of validation evaluation (in epochs)')
    
    args = parser.parse_args()
    
    # Setup logger
    current_dir = os.path.dirname(os.path.abspath(__file__))
    logs_dir = os.path.join(current_dir, 'logs')
    if not os.path.exists(logs_dir):
        os.makedirs(logs_dir)
    
    logger = setup_logger('lenet', os.path.join(logs_dir, 'lenet.log'))
    logger.info(f"Starting LeNet training with parameters: {args}")
    
    # Create datasets
    train_dataset = BreastHistopathologyDataset(
        root_dir=args.data_dir,
        train=True,
        val_ratio=args.val_ratio,
        seed=args.seed
    )
    
    val_dataset = BreastHistopathologyDataset(
        root_dir=args.data_dir,
        train=False,
        val_ratio=args.val_ratio,
        seed=args.seed
    )
    
    # Train model
    logger.info("Starting model training...")
    model = train_model(
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        logger=logger,
        patience=args.patience,
        eval_freq=args.eval_freq
    )
    
    # Evaluate model
    logger.info("Evaluating model on validation set...")
    metrics = evaluate_model(model, val_dataset, logger=logger)
    
    logger.info("Training and evaluation completed successfully!")

if __name__ == '__main__':
    main() 