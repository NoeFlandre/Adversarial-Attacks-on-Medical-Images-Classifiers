# LeNet Classifier for Tumor Classification

A PyTorch-based LeNet CNN classifier for breast tumor histopathology images with adversarial attack capabilities.

## Structure
- `src/`: Core source code (model, dataset, logger)
- `scripts/`: Runtime scripts for training, evaluation, and attacks
- `checkpoints/`: Saved model weights
- `results/`: Output metrics and visualizations
  - `training_evaluation/`: Training and evaluation metrics and visualizations
  - `adversarial/`: Adversarial attack results and examples
  - `visualizations/`: Visual analysis of attack performance
- `logs/`: Application logs
  - `adversarial_attacks.log/`: Logs from adversarial attack runs with timestamped files
  - `training_evaluation.log/`: Logs from training and evaluation runs with timestamped files
  - `visualize_attacks.log/`: Logs from visualization processes with timestamped files

## Quick Start

All functionality is accessible through the main.py interface with four primary commands:
- `train`: Train the LeNet model
- `eval`: Evaluate model performance on test data
- `attack`: Run adversarial attacks on the trained model
- `visualize`: Generate visualizations of attack results

You can use the module path:

```bash
python -m models.lenet.main train
```

## Usage

### Training
Train the LeNet model on the breast tumor dataset:

```bash
python -m models.lenet.main train --data_dir data/ --epochs 20 --batch_size 1024 --learning_rate 1e-4
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

### Visualizing Attack Results
Generate epsilon vs accuracy plots:

```bash
python -m models.lenet.main visualize
```

## Key Parameters

### Training
- `--data_dir`: Path to dataset directory
- `--epochs`: Number of training epochs (default: 20)
- `--batch_size`: Batch size for training (default: 1024)
- `--learning_rate`: Learning rate for optimizer (default: 1e-4)
- `--val_ratio`: Validation set ratio (default: 0.2)
- `--seed`: Random seed for reproducibility (default: 42)

### Evaluation
- `--data_dir`: Path to dataset directory
- `--checkpoint`: Model checkpoint path (uses latest if not specified)
- `--batch_size`: Batch size for evaluation (default: 64)

### Attacks
- `--data_dir`: Path to dataset directory
- `--checkpoint`: Model checkpoint path (uses latest if not specified)
- `--batch_size`: Batch size for attack evaluation (default: 64)
- `--epsilons`: Attack strength values (default: [0.01, 0.05, 0.1, 0.2])

### Visualization
- `--results_dir`: Path to results directory (default: models/lenet/results/adversarial)

## Model Architecture

LeNet is a classic convolutional neural network architecture with the following components:
- 2 convolutional layers with max pooling
- 3 fully connected layers
- ReLU activation and batch normalization
- Dropout for regularization

This implementation is optimized for binary classification of breast tumor histopathology images.

## Output Organization

### Results
Results are automatically saved in appropriate subfolders:
- `results/training_evaluation/`: Contains:
  - `confusion_matrix_<timestamp>.png`: Confusion matrix for the trained model
  - `metrics_<timestamp>.json`: Performance metrics
  - `roc_curve_<timestamp>.png`: ROC curve visualization
  - `training_validation_loss.png`: Training and validation loss curve

- `results/adversarial/`: Contains:
  - `FGSM_examples_eps<value>_<timestamp>.png`: Generated adversarial examples
  - `FGSM_metrics_eps<value>_<timestamp>.json`: Attack performance metrics

- `results/visualizations/`: Contains:
  - `epsilon_vs_accuracy.png`: Comparison of accuracy across different attack strengths

### Logs
Logs are organized as follows:
- `logs/training_evaluation.log/`: Directory containing logs from training and evaluation runs
  - `lenet_<timestamp>.log`: Individual log file for each training/evaluation run
- `logs/adversarial_attacks.log/`: Directory containing logs from adversarial attack runs
  - `lenet_<timestamp>.log`: Individual log file for each attack run
- `logs/visualize_attacks.log/`: Directory containing logs from visualization processes
  - `lenet_<timestamp>.log`: Individual log file for each visualization run