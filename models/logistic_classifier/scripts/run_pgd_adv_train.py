#!/usr/bin/env python3
"""
Script to run PGD adversarial training for Logistic Regression model
"""
import argparse
import torch
import os
import sys
# Ensure src and scripts packages are importable
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.model import LogisticRegressionModel
from src.dataset import BreastHistopathologyDataset
from src.logger import setup_logger
import datetime
import json

def main():
    parser = argparse.ArgumentParser(description='Train logistic regression model with PGD adversarial training')
    parser.add_argument('--data_path', type=str, default='data/', help='Path to the dataset')
    parser.add_argument('--epochs', type=int, default=1, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=1024, help='Batch size for training and evaluation')
    parser.add_argument('--learning_rate', type=float, default=1e-5, help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-5, help='Weight decay for Adam optimizer')
    parser.add_argument('--val_ratio', type=float, default=0.2, help='Ratio of validation data')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility')
    parser.add_argument('--epsilon', type=float, default=0.1, help='Epsilon for PGD attack')
    parser.add_argument('--alpha', type=float, default=0.01, help='Step size for PGD attack')
    parser.add_argument('--num_iter', type=int, default=20, help='Number of iterations for PGD attack')
    parser.add_argument('--random_start', action='store_true', help='Use random initialization for PGD attack')
    parser.add_argument('--mix_ratio', type=float, default=0.5, help='Ratio of adversarial examples in each batch')
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
        torch.backends.cudnn.deterministic = True

    # Logger setup
    log_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'logs/pgd_adversarial_training.log')
    os.makedirs(os.path.dirname(log_dir), exist_ok=True)
    logger = setup_logger(log_dir)

    logger.info("Starting Logistic Regression with PGD adversarial training")
    logger.info(f"Configuration: epochs={args.epochs}, batch_size={args.batch_size}, lr={args.learning_rate}, weight_decay={args.weight_decay}, val_ratio={args.val_ratio}, epsilon={args.epsilon}, alpha={args.alpha}, num_iter={args.num_iter}, random_start={args.random_start}, mix_ratio={args.mix_ratio}")

    train_dataset = BreastHistopathologyDataset(args.data_path, train=True, val_ratio=args.val_ratio, seed=args.seed)
    val_dataset = BreastHistopathologyDataset(args.data_path, train=False, val_ratio=args.val_ratio, seed=args.seed)

    logger.info(f"Train size: {len(train_dataset)}, Val size: {len(val_dataset)}")
    model = LogisticRegressionModel()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    logger.info(f"Using device: {device}")

    # Train
    from scripts.train_adversarial_pgd import train_model_adversarial_pgd
    train_model_adversarial_pgd(model, train_dataset, val_dataset,
                                epochs=args.epochs,
                                batch_size=args.batch_size,
                                lr=args.learning_rate,
                                device=device,
                                logger=logger,
                                epsilon=args.epsilon,
                                alpha=args.alpha,
                                num_iter=args.num_iter,
                                random_start=args.random_start,
                                mix_ratio=args.mix_ratio)

    # Evaluate on validation set
    from scripts.eval import evaluate_model
    metrics = evaluate_model(model, val_dataset, batch_size=args.batch_size, device=device, logger=logger, save_results=True)

    # Save model
    current = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    ckpt_dir = os.path.join(current, 'checkpoints')
    os.makedirs(ckpt_dir, exist_ok=True)
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    model_path = os.path.join(ckpt_dir, f'logistic_pgd_adv_eps{args.epsilon}_alpha{args.alpha}_iter{args.num_iter}_{ts}.pth')
    torch.save({'model_state_dict': model.state_dict(), 'metrics': metrics, 'config': vars(args)}, model_path)
    logger.info(f"Saved model to {model_path}")

    # Save summary
    results_dir = os.path.join(current, 'results/adversarial_training_pgd')
    os.makedirs(results_dir, exist_ok=True)
    summary = {'timestamp': ts, 'epsilon': args.epsilon, 'alpha': args.alpha, 'num_iter': args.num_iter, 'metrics': metrics}
    summary_path = os.path.join(results_dir, f'pgd_summary_eps{args.epsilon}_alpha{args.alpha}_iter{args.num_iter}_{ts}.json')
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=4)
    logger.info(f"Saved summary to {summary_path}")

if __name__ == '__main__':
    main()