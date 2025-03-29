import os
import argparse
import torch
import datetime

from ..src.logger import setup_logger
from .train import train_model
from .eval import evaluate_model_from_checkpoint

def main():
    """Main function for training and evaluating LeNet model"""
    parser = argparse.ArgumentParser(description='Train and evaluate LeNet model for breast tumor classification')
    parser.add_argument('--data_dir', type=str, default='data/', help='Path to the dataset directory')
    parser.add_argument('--epochs', type=int, default=20, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=1024, help='Batch size for training')
    parser.add_argument('--learning_rate', type=float, default=1e-4, help='Initial learning rate')
    parser.add_argument('--val_ratio', type=float, default=0.2, help='Ratio of validation data')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility')
    parser.add_argument('--eval_only', action='store_true', help='Only evaluate model (no training)')
    parser.add_argument('--checkpoint', type=str, default=None, help='Path to model checkpoint for evaluation')
    
    args = parser.parse_args()
    
    # Setup logger
    current_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    # Create a single logger for both training and evaluation
    logger = setup_logger(f'lenet_{timestamp}', log_type='training_evaluation')
    logger.info(f"Starting LeNet with parameters: {args}")
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 
                         'mps' if torch.backends.mps.is_available() else 
                         'cpu')
    logger.info(f"Using device: {device}")
    
    # Set checkpoint directory
    checkpoint_dir = os.path.join(current_dir, 'checkpoints')
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    # If eval_only, just evaluate the model
    if args.eval_only:
        if args.checkpoint is None:
            # Use default checkpoint if not specified
            args.checkpoint = os.path.join(checkpoint_dir, 'lenet_model_best.pth')
            
        logger.info(f"Evaluating model from checkpoint: {args.checkpoint}")
        metrics = evaluate_model_from_checkpoint(
            data_dir=args.data_dir,
            checkpoint_path=args.checkpoint,
            batch_size=args.batch_size
        )
        
        logger.info("Evaluation completed successfully!")
    else:
        # Train model
        logger.info("Starting model training...")
        model, history = train_model(
            data_dir=args.data_dir,
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            val_ratio=args.val_ratio,
            seed=args.seed,
            checkpoint_dir=checkpoint_dir
        )
        
        logger.info("Training completed successfully!")
        
        # Evaluate the final model
        if args.checkpoint is None:
            args.checkpoint = os.path.join(checkpoint_dir, 'lenet_model_best.pth')
            
        logger.info(f"Evaluating best model from checkpoint: {args.checkpoint}")
        metrics = evaluate_model_from_checkpoint(
            data_dir=args.data_dir,
            checkpoint_path=args.checkpoint,
            batch_size=args.batch_size,
            logger=logger,
            save_results=True,  # Save results during evaluation
            timestamp=timestamp  # Use the same timestamp as training for consistency
        )
        
        logger.info("Evaluation completed successfully!")
    
    return metrics

if __name__ == '__main__':
    main() 