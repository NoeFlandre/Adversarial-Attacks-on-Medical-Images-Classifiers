# Logistic Regression Classifier for Tumor Classification

A PyTorch-based classifier for breast tumor histopathology images with adversarial attack.

## Structure
- `src/`: Core source code (model, dataset, attacks)
- `scripts/`: Runtime scripts for training, evaluation, and attacks
- `checkpoints/`: Saved model weights
- `results/`: Output metrics and visualizations
- `logs/`: Application logs

## Quick Start

All functionality is accessible through the main.py interface with three primary commands:
- `train`: Train the logistic regression model
- `attack`: Run adversarial attacks on the trained model
- `visualize`: Generate visualizations of attack results

You can use the module path :

```bash
python -m models.logistic_classifier.main train
```

## Usage

## Key Parameters

### Training
- `--data_path`: Path to dataset directory
- `--epochs`: Number of training epochs
- `--batch_size`: Batch size for training
- `--learning_rate`: Learning rate for optimizer
- `--val_ratio`: Validation set ratio (default: 0.2)
- `--seed`: Random seed for reproducibility (default: 42)

### Attacks
- `--data_dir`: Path to dataset directory
- `--checkpoint`: Model checkpoint path (uses latest if not specified)
- `--batch_size`: Batch size for attack evaluation
- `--epsilons`: Attack strength values (multiple values allowed)
- `--save_adv_examples`: Flag to save adversarial examples for later analysis

### Visualization
- `--results_dir`: Path to results directory (default: models/logistic_classifier/results/adversarial)
- `--adv_dataset`: Path to specific adversarial dataset (optional)
- `--num_samples`: Number of samples to visualize (default: 5)

