#!/usr/bin/env python3
"""
Main entry point for the Breast Tumor Classification project.
This script provides commands to run training, evaluation, adversarial attacks,
and visualization of results.
"""

import argparse
import os
import sys

# Ensure the scripts package is importable when running as a script
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def main():
    parser = argparse.ArgumentParser(
        description="Breast Tumor Classification - Logistic Regression with Adversarial Attacks",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Command to run')
    
    # Training command
    train_parser = subparsers.add_parser('train', help='Train the logistic regression model')
    train_parser.add_argument('--data_path', type=str, default='data/', help='Path to the dataset')
    train_parser.add_argument('--epochs', type=int, default=1, help='Number of training epochs')
    train_parser.add_argument('--batch_size', type=int, default=1024, help='Batch size for training and evaluation')
    train_parser.add_argument('--learning_rate', type=float, default=1e-5, help='Learning rate')
    train_parser.add_argument('--weight_decay', type=float, default=1e-5, help='Weight decay for Adam optimizer')
    train_parser.add_argument('--val_ratio', type=float, default=0.2, help='Ratio of validation data')
    train_parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility')
    
    # Adversarial training command
    adv_train_parser = subparsers.add_parser('adv_train', help='Train the logistic regression model with adversarial training')
    adv_train_parser.add_argument('--data_path', type=str, default='data/', help='Path to the dataset')
    adv_train_parser.add_argument('--epochs', type=int, default=1, help='Number of training epochs')
    adv_train_parser.add_argument('--batch_size', type=int, default=1024, help='Batch size for training and evaluation')
    adv_train_parser.add_argument('--learning_rate', type=float, default=1e-5, help='Learning rate')
    adv_train_parser.add_argument('--weight_decay', type=float, default=1e-5, help='Weight decay for Adam optimizer')
    adv_train_parser.add_argument('--val_ratio', type=float, default=0.2, help='Ratio of validation data')
    adv_train_parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility')
    adv_train_parser.add_argument('--epsilon', type=float, default=0.05, help='Epsilon for FGSM adversarial training')
    adv_train_parser.add_argument('--mix_ratio', type=float, default=0.5, help='Ratio of adversarial examples in each batch')

    # PGD adversarial training command
    pgd_adv_train_parser = subparsers.add_parser('pgd_adv_train', help='Train the logistic regression model with PGD adversarial training')
    pgd_adv_train_parser.add_argument('--data_path', type=str, default='data/', help='Path to the dataset')
    pgd_adv_train_parser.add_argument('--epochs', type=int, default=1, help='Number of training epochs')
    pgd_adv_train_parser.add_argument('--batch_size', type=int, default=1024, help='Batch size for training and evaluation')
    pgd_adv_train_parser.add_argument('--learning_rate', type=float, default=1e-5, help='Learning rate')
    pgd_adv_train_parser.add_argument('--weight_decay', type=float, default=1e-5, help='Weight decay for Adam optimizer')
    pgd_adv_train_parser.add_argument('--val_ratio', type=float, default=0.2, help='Ratio of validation data')
    pgd_adv_train_parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility')
    pgd_adv_train_parser.add_argument('--epsilon', type=float, default=0.1, help='Epsilon for PGD attack')
    pgd_adv_train_parser.add_argument('--alpha', type=float, default=0.01, help='Step size for PGD attack')
    pgd_adv_train_parser.add_argument('--num_iter', type=int, default=20, help='Number of iterations for PGD attack')
    pgd_adv_train_parser.add_argument('--random_start', action='store_true', help='Use random initialization for PGD attack')
    pgd_adv_train_parser.add_argument('--mix_ratio', type=float, default=0.5, help='Ratio of adversarial examples in each batch')

    
    # DeepFool adversarial training command
    deepfool_adv_train_parser = subparsers.add_parser('deepfool_adv_train', help='Train the logistic regression model with DeepFool adversarial training')
    deepfool_adv_train_parser.add_argument('--data_path', type=str, default='data/', help='Path to the dataset')
    deepfool_adv_train_parser.add_argument('--epochs', type=int, default=1, help='Number of training epochs')
    deepfool_adv_train_parser.add_argument('--batch_size', type=int, default=1024, help='Batch size for training and evaluation')
    deepfool_adv_train_parser.add_argument('--learning_rate', type=float, default=1e-5, help='Learning rate')
    deepfool_adv_train_parser.add_argument('--weight_decay', type=float, default=1e-5, help='Weight decay for Adam optimizer')
    deepfool_adv_train_parser.add_argument('--val_ratio', type=float, default=0.2, help='Ratio of validation data')
    deepfool_adv_train_parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility')
    deepfool_adv_train_parser.add_argument('--max_iter', type=int, default=50, help='Max iterations for DeepFool attack')
    deepfool_adv_train_parser.add_argument('--overshoot', type=float, default=0.02, help='Overshoot factor for DeepFool attack')
    
    # Adversarial attacks command
    attack_parser = subparsers.add_parser('attack', help='Run adversarial attacks on the model')
    attack_parser.add_argument('--data_dir', type=str, default='data/', help='Path to the dataset directory')
    attack_parser.add_argument('--checkpoint', type=str, default=None, help='Path to model checkpoint')
    attack_parser.add_argument('--batch_size', type=int, default=64, help='Batch size for evaluation')
    attack_parser.add_argument('--epsilons', type=float, nargs='+', default=[0.01, 0.05, 0.1, 0.2], 
                        help='Epsilon values for FGSM attack (attack strength)')
    attack_parser.add_argument('--save_adv_examples', action='store_true', help='Save adversarial examples')
    # Select attack type and DeepFool parameters
    attack_parser.add_argument('--attack', choices=['FGSM','DeepFool'], default='FGSM', help='Type of adversarial attack to run (FGSM or DeepFool)')
    attack_parser.add_argument('--max_iter', type=int, default=50, help='Max iterations for DeepFool attack')
    attack_parser.add_argument('--overshoot', type=float, default=0.02, help='Overshoot factor for DeepFool attack')
    
    # Visualization command
    vis_parser = subparsers.add_parser('visualize', help='Visualize attack results')
    vis_parser.add_argument('--results_dir', type=str, default=None, 
                        help='Path to adversarial results directory')
    vis_parser.add_argument('--adv_dataset', type=str, default=None,
                        help='Path to saved adversarial dataset')
    vis_parser.add_argument('--num_samples', type=int, default=5,
                        help='Number of samples to visualize')
    
    # Compare models command
    compare_parser = subparsers.add_parser('compare', help='Compare standard and adversarially trained models')
    compare_parser.add_argument('--standard_model', type=str, required=True, 
                              help='Path to standard trained model checkpoint')
    compare_parser.add_argument('--adversarial_model', type=str, required=True, 
                              help='Path to adversarially trained model checkpoint')
    compare_parser.add_argument('--data_path', type=str, default='data/', 
                              help='Path to the dataset directory')
    compare_parser.add_argument('--epsilons', type=float, nargs='+', default=[0.01, 0.05, 0.1, 0.2], 
                              help='Epsilon values for FGSM attack (attack strength)')
    compare_parser.add_argument('--batch_size', type=int, default=64, 
                              help='Batch size for evaluation')
    
    args = parser.parse_args()
    
    if args.command == 'train':
        # Import here to avoid circular imports
        from scripts.run import main as train_main
        # Convert namespace to dictionary and pass as kwargs
        train_args = vars(args)
        del train_args['command']  # Remove the command key
        # Run the training script
        sys.argv = ['train.py']  # Reset sys.argv to avoid conflicts with argparse
        for key, value in train_args.items():
            if value is not None:
                if isinstance(value, bool) and value:
                    sys.argv.append(f'--{key}')
                else:
                    sys.argv.extend([f'--{key}', str(value)])
        train_main()
    
    elif args.command == 'adv_train':
        from scripts.run_adv_train import main as adv_train_main
        # Run the adversarial training script
        sys.argv = ['run_adv_train.py']  # Reset sys.argv
        adv_train_args = vars(args)
        del adv_train_args['command']
        for key, value in adv_train_args.items():
            if value is not None:
                if isinstance(value, bool) and value:
                    sys.argv.append(f'--{key}')
                else:
                    sys.argv.extend([f'--{key}', str(value)])
        adv_train_main()
        
    elif args.command == 'deepfool_adv_train':
        from scripts.run_deepfool_adv_train import main as deepfool_adv_train_main
        # Run the DeepFool adversarial training script
        sys.argv = ['run_deepfool_adv_train.py']  # Reset sys.argv
        deepfool_adv_train_args = vars(args)
        del deepfool_adv_train_args['command']
        for key, value in deepfool_adv_train_args.items():
            if value is not None:
                if isinstance(value, bool) and value:
                    sys.argv.append(f'--{key}')
                else:
                    sys.argv.extend([f'--{key}', str(value)])
        deepfool_adv_train_main()

    elif args.command == 'pgd_adv_train':
        from scripts.run_pgd_adv_train import main as pgd_adv_train_main
        # Run the PGD adversarial training script
        sys.argv = ['run_pgd_adv_train.py']  # Reset sys.argv
        pgd_adv_train_args = vars(args)
        del pgd_adv_train_args['command']
        for key, value in pgd_adv_train_args.items():
            if value is not None:
                if isinstance(value, bool) and value:
                    sys.argv.append(f'--{key}')
                else:
                    sys.argv.extend([f'--{key}', str(value)])
        pgd_adv_train_main()
        
    elif args.command == 'attack':
        from scripts.run_attacks import main as attack_main
        # Run the attack script
        sys.argv = ['run_attacks.py']  # Reset sys.argv
        attack_args = vars(args)
        del attack_args['command']
        for key, value in attack_args.items():
            if value is not None and key != 'epsilons' and key != 'save_adv_examples':
                sys.argv.extend([f'--{key}', str(value)])
            elif key == 'epsilons':
                sys.argv.append('--epsilons')
                for eps in value:
                    sys.argv.append(str(eps))
            elif key == 'save_adv_examples' and value:
                sys.argv.append('--save_adv_examples')
        attack_main()
        
    elif args.command == 'visualize':
        from scripts.visualize_attacks import main as visualize_main
        # Run the visualization script
        sys.argv = ['visualize_attacks.py']  # Reset sys.argv
        vis_args = vars(args)
        del vis_args['command']
        for key, value in vis_args.items():
            if value is not None:
                if isinstance(value, bool) and value:
                    sys.argv.append(f'--{key}')
                else:
                    sys.argv.extend([f'--{key}', str(value)])
        visualize_main()
    
    elif args.command == 'compare':
        from scripts.compare_models import main as compare_main
        # Run the comparison script
        sys.argv = ['compare_models.py']  # Reset sys.argv
        compare_args = vars(args)
        del compare_args['command']
        for key, value in compare_args.items():
            if value is not None and key != 'epsilons':
                sys.argv.extend([f'--{key}', str(value)])
            elif key == 'epsilons':
                sys.argv.append('--epsilons')
                for eps in value:
                    sys.argv.append(str(eps))
        compare_main()
        
    else:
        parser.print_help()

if __name__ == '__main__':
    main() 