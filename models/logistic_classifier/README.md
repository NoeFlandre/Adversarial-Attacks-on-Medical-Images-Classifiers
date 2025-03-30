# Logistic Regression Classifier for Tumor Classification

A PyTorch-based classifier for breast tumor histopathology images with support for adversarial attacks and adversarial training.

## Structure
- `src/`: Core source code (model, dataset, attacks)
- `scripts/`: Runtime scripts for training, evaluation, and attacks
- `checkpoints/`: Saved model weights
- `results/`: Output metrics and visualizations
- `logs/`: Application logs

## Quick Start

All functionality is accessible through the main.py interface with these commands:
- `train`: Train the standard logistic regression model
- `adv_train`: Train the model with adversarial training (FGSM)
- `attack`: Run adversarial attacks on a trained model
- `visualize`: Generate visualizations of attack results
- `compare`: Compare standard and adversarially trained models


## Usage Examples

### Standard Training
```bash
python -m models.logistic_classifier.main train --data_path data/ --epochs 10 --batch_size 1024 --learning_rate 1e-5
```

### Adversarial Training
```bash
python -m models.logistic_classifier.main adv_train --data_path data/ --epochs 10 --batch_size 1024 --epsilon 0.05 --mix_ratio 0.5
```

### Running Attacks
```bash
python -m models.logistic_classifier.main attack --data_dir data/ --checkpoint models/logistic_classifier/checkpoints/my_model.pth --epsilons 0.01 0.05 0.1 0.2
```

### Compare Models
```bash
python -m models.logistic_classifier.main compare --standard_model models/logistic_classifier/checkpoints/standard_model.pth --adversarial_model models/logistic_classifier/checkpoints/adversarial_model.pth
```

## Key Parameters

### Standard Training
- `--data_path`: Path to dataset directory
- `--epochs`: Number of training epochs
- `--batch_size`: Batch size for training
- `--learning_rate`: Learning rate for optimizer
- `--val_ratio`: Validation set ratio (default: 0.2)
- `--seed`: Random seed for reproducibility (default: 42)

### Adversarial Training
- All parameters from standard training, plus:
- `--epsilon`: Perturbation size for FGSM adversarial examples (default: 0.05)
- `--mix_ratio`: Ratio of adversarial examples in each batch (default: 0.5)

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

### Model Comparison
- `--standard_model`: Path to standard trained model checkpoint
- `--adversarial_model`: Path to adversarially trained model checkpoint
- `--data_path`: Path to dataset directory
- `--epsilons`: Attack strength values to test both models against
- `--batch_size`: Batch size for evaluation

## Adversarial Training

The adversarial training implementation uses the Fast Gradient Sign Method (FGSM) to generate adversarial examples during training. The key parameters are:

- `epsilon`: Controls the perturbation magnitude (attack strength)
- `mix_ratio`: Controls what proportion of each batch consists of adversarial examples

During training, the model learns from a mixture of clean and adversarial examples, making it more robust against adversarial attacks.

## Model Comparison

The comparison tool evaluates both standard and adversarially trained models against FGSM attacks of different strengths. It produces:

1. A JSON file with detailed metrics
2. A comparative plot showing how accuracy degrades with increasing attack strength
3. Logs detailing the performance differences

Results are saved to the `results/comparisons` directory.

