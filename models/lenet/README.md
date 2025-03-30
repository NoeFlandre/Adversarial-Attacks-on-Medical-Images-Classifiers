# LeNet Classifier for Tumor Classification

A PyTorch-based LeNet CNN classifier for breast tumor histopathology images with support for adversarial attacks and adversarial training.

## Structure
- `src/`: Core source code (model, dataset, logger)
- `scripts/`: Runtime scripts for training, evaluation, and attacks
- `checkpoints/`: Saved model weights
- `results/`: Output metrics and visualizations
  - `training_evaluation/`: Training and evaluation metrics and visualizations
  - `adversarial/`: Adversarial attack results and examples
  - `adversarial_training/`: Results from adversarial training
  - `comparisons/`: Comparison results between standard and adversarially trained models
  - `visualizations/`: Visual analysis of attack performance
- `logs/`: Application logs
  - `adversarial_attacks.log/`: Logs from adversarial attack runs
  - `training_evaluation.log/`: Logs from training and evaluation runs
  - `adversarial_training.log/`: Logs from adversarial training runs
  - `model_comparison.log/`: Logs from model comparison runs
  - `visualize_attacks.log/`: Logs from visualization processes

## Quick Start

All functionality is accessible through the main.py interface with six primary commands:
- `train`: Train the standard LeNet model
- `adv_train`: Train the LeNet model with adversarial training
- `eval`: Evaluate model performance on test data
- `attack`: Run adversarial attacks on a trained model
- `compare`: Compare standard and adversarially trained models
- `visualize`: Generate visualizations of attack results

You can use the module path:

```bash
python -m models.lenet.main train
```

## Usage

### Standard Training
Train the LeNet model on the breast tumor dataset:

```bash
python -m models.lenet.main train --data_dir data/ --epochs 20 --batch_size 1024 --learning_rate 1e-4
```

### Adversarial Training
Train the LeNet model with adversarial examples:

```bash
python -m models.lenet.main adv_train --data_dir data/ --epochs 20 --epsilon 0.05 --mix_ratio 0.5
```

### Evaluation
Evaluate the model on the test dataset:

```bash
python -m models.lenet.main eval --data_dir data/ --checkpoint models/lenet/checkpoints/lenet_model_best.pth
```

### Running Adversarial Attacks
Run FGSM attacks with various epsilon values:

```bash
python -m models.lenet.main attack --data_dir data/ --epsilons 0.01 0.05 0.1 0.2
```

### Comparing Models
Compare standard and adversarially trained models:

```bash
python -m models.lenet.main compare --standard_model models/lenet/checkpoints/lenet_model_best.pth --adversarial_model models/lenet/checkpoints/lenet_adv_model_best.pth
```

### Visualizing Attack Results
Generate epsilon vs accuracy plots:

```bash
python -m models.lenet.main visualize
```

## Key Parameters

### Standard Training
- `--data_dir`: Path to dataset directory
- `--epochs`: Number of training epochs (default: 20)
- `--batch_size`: Batch size for training (default: 1024)
- `--learning_rate`: Learning rate for optimizer (default: 1e-4)
- `--val_ratio`: Validation set ratio (default: 0.2)
- `--seed`: Random seed for reproducibility (default: 42)

### Adversarial Training
- All parameters from standard training, plus:
- `--epsilon`: Perturbation size for FGSM adversarial examples (default: 0.05)
- `--mix_ratio`: Ratio of adversarial examples in each batch (default: 0.5)

### Evaluation
- `--data_dir`: Path to dataset directory
- `--checkpoint`: Model checkpoint path (uses latest if not specified)
- `--batch_size`: Batch size for evaluation (default: 64)

### Attacks
- `--data_dir`: Path to dataset directory
- `--checkpoint`: Model checkpoint path (uses latest if not specified)
- `--batch_size`: Batch size for attack evaluation (default: 64)
- `--epsilons`: Attack strength values (default: [0.01, 0.05, 0.1, 0.2])

### Model Comparison
- `--standard_model`: Path to standard trained model checkpoint
- `--adversarial_model`: Path to adversarially trained model checkpoint
- `--data_dir`: Path to dataset directory
- `--epsilons`: Attack strength values to test both models against
- `--batch_size`: Batch size for evaluation

### Visualization
- `--results_dir`: Path to results directory (default: models/lenet/results/adversarial)

## Model Architecture

LeNet is a classic convolutional neural network architecture with the following components:
- 2 convolutional layers with max pooling
- 3 fully connected layers
- ReLU activation and batch normalization
- Dropout for regularization

This implementation is optimized for binary classification of breast tumor histopathology images.

## Adversarial Training

The adversarial training implementation uses the Fast Gradient Sign Method (FGSM) to generate adversarial examples during training. The key parameters are:

- `epsilon`: Controls the perturbation magnitude (attack strength)
- `mix_ratio`: Controls what proportion of each batch consists of adversarial examples

During training, the model learns from a mixture of clean and adversarial examples, making it more robust against adversarial attacks. The adversarial training process includes:

1. Splitting each batch into clean and adversarial portions based on mix_ratio
2. Generating adversarial examples for the adversarial portion using FGSM
3. Training the model on both clean and adversarial examples
4. Evaluating performance on both clean and adversarial validation data

## Model Comparison

The comparison tool evaluates both standard and adversarially trained models against FGSM attacks of different strengths. It produces:

1. A JSON file with detailed metrics
2. A comparative plot showing how accuracy degrades with increasing attack strength
3. Logs detailing the performance differences

This analysis provides insight into how effective adversarial training is at improving model robustness.

## Output Organization

### Results
Results are automatically saved in appropriate subfolders:
- `results/training_evaluation/`: Contains:
  - `confusion_matrix_<timestamp>.png`: Confusion matrix for the standard trained model
  - `metrics_<timestamp>.json`: Performance metrics
  - `roc_curve_<timestamp>.png`: ROC curve visualization
  - `training_validation_loss.png`: Training and validation loss curve

- `results/adversarial_training/`: Contains:
  - `training_validation_loss_eps<value>_mix<value>_<timestamp>.png`: Loss curves for adversarially trained model
  - `clean_vs_adversarial_accuracy_eps<value>_mix<value>_<timestamp>.png`: Comparison of clean vs. adversarial accuracy
  - `training_history_eps<value>_mix<value>_<timestamp>.json`: Training history and metrics
  - `adversarial_training_summary_eps<value>_mix<value>_<timestamp>.json`: Summary metrics

- `results/adversarial/`: Contains:
  - `FGSM_examples_eps<value>_<timestamp>.png`: Generated adversarial examples
  - `FGSM_metrics_eps<value>_<timestamp>.json`: Attack performance metrics

- `results/comparisons/`: Contains:
  - `model_comparison_<timestamp>.json`: Metrics comparing standard vs. adversarially trained models
  - `model_comparison_plot_<timestamp>.png`: Plot showing accuracy vs. attack strength

- `results/visualizations/`: Contains:
  - `epsilon_vs_accuracy.png`: Comparison of accuracy across different attack strengths

### Logs
Logs are organized as follows:
- `logs/training_evaluation.log/`: Logs from standard training and evaluation runs
- `logs/adversarial_training.log/`: Logs from adversarial training runs
- `logs/adversarial_attacks.log/`: Logs from adversarial attack runs
- `logs/model_comparison.log/`: Logs from model comparison runs
- `logs/visualize_attacks.log/`: Logs from visualization processes