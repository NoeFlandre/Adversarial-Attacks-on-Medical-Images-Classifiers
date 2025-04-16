import argparse
import torch
import os
import sys
from ..src.model import TinyTransformer
from ..src.dataset import BreastHistopathologyDataset
from ..src.attacks import FGSM, evaluate_attack, AdversarialDataset
from ..src.logger import setup_logger

def main():
    parser = argparse.ArgumentParser(description='Run adversarial attacks on the model')
    parser.add_argument('--data_dir', type=str, default='data/', help='Path to the dataset directory')
    parser.add_argument('--checkpoint', type=str, default=None, help='Path to model checkpoint')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size for evaluation')
    parser.add_argument('--epsilons', type=float, nargs='+', default=[0.01, 0.05, 0.1, 0.2], 
                        help='Epsilon values for FGSM attack (attack strength)')
    parser.add_argument('--save_adv_examples', action='store_true', help='Save adversarial examples')
    args = parser.parse_args()

    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 
                         'mps' if torch.backends.mps.is_available() else 
                         'cpu')
    
    # Setup logger
    current_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    log_dir = os.path.join(current_dir, 'logs')
    os.makedirs(log_dir, exist_ok=True)
    logger = setup_logger(os.path.join(log_dir, 'adversarial_attacks.log'))
    
    logger.info(f"Running adversarial attack evaluation with device: {device}")
    
    # Load dataset
    logger.info(f"Loading dataset from {args.data_dir}")
    test_dataset = BreastHistopathologyDataset(
        root_dir=args.data_dir,
        train=False,
        val_ratio=0.2
    )
    logger.info(f"Test dataset size: {len(test_dataset)}")
    
    # Load model
    model = TinyTransformer()
    
    # Log parameter count
    num_params = model.count_parameters()
    logger.info(f"Tiny Transformer parameters: {num_params:,} ({num_params/(1000000):.2f}M)")
    
    if args.checkpoint:
        logger.info(f"Loading model checkpoint from {args.checkpoint}")
        checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        # Look for latest checkpoint in checkpoint directory
        checkpoint_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'checkpoints')
        if os.path.exists(checkpoint_dir):
            # Find latest tiny_transformer checkpoint (standard or adversarial)
            checkpoints = [f for f in os.listdir(checkpoint_dir) 
                           if f.endswith('.pth') and (f.startswith('tiny_transformer_model') or f.startswith('tiny_transformer_adversarial'))]
            if checkpoints:
                latest_checkpoint = max(checkpoints, key=lambda x: os.path.getmtime(os.path.join(checkpoint_dir, x)))
                checkpoint_path = os.path.join(checkpoint_dir, latest_checkpoint)
                logger.info(f"Loading latest model checkpoint from {checkpoint_path}")
                checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
                model.load_state_dict(checkpoint['model_state_dict'])
            else:
                logger.error("No checkpoint found in checkpoint directory")
                sys.exit(1)
        else:
            logger.error("No checkpoint specified and checkpoint directory not found")
            sys.exit(1)
    
    model.to(device)
    model.eval()
    
    # Create loss function with class weights for FGSM
    criterion = torch.nn.CrossEntropyLoss()
    
    # Run FGSM attacks with different epsilon values
    for epsilon in args.epsilons:
        logger.info(f"Running FGSM attack with epsilon={epsilon}")
        
        # Create FGSM attack
        fgsm_attack = FGSM(model, criterion=criterion, epsilon=epsilon)
        
        # Evaluate attack
        metrics = evaluate_attack(
            model=model,
            dataset=test_dataset,
            attack=fgsm_attack,
            batch_size=args.batch_size,
            device=device,
            logger=logger,
            save_results=True
        )
        
        # Save adversarial examples if requested
        if args.save_adv_examples:
            logger.info(f"Generating and saving adversarial dataset with epsilon={epsilon}")
            adv_dataset = AdversarialDataset(test_dataset, fgsm_attack, device=device)
            
            # Save the adversarial dataset
            adv_dataset_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'results', 'adversarial', f'fgsm_eps{epsilon}')
            os.makedirs(adv_dataset_dir, exist_ok=True)
            
            torch.save({
                'adversarial_images': adv_dataset.adversarial_images,
                'labels': adv_dataset.labels
            }, os.path.join(adv_dataset_dir, 'adversarial_dataset.pt'))
            
            logger.info(f"Saved adversarial dataset to {adv_dataset_dir}")

if __name__ == '__main__':
    main()