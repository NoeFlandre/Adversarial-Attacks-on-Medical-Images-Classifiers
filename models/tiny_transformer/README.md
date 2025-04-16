# Tiny Transformer for Breast Tumor Classification

Here we implement a Tiny Transformer model for classifying breast cancer histopathology images.
It includes scripts for training, evaluation, adversarial attacks (FGSM), 
and visualization of results.


## Features

- Classification of 50x50 pixel breast histopathology images.
- Tiny Transformer architecture (low parameter count).
- Training with balanced class weights to handle dataset imbalance.
- Evaluation metrics: Accuracy, Balanced Accuracy, Precision, Recall, Specificity, F1 Score, AUC, Confusion Matrix.
- Adversarial training using FGSM.
- Adversarial attacks using FGSM to test model robustness.
- Visualization of training progress, evaluation results, and adversarial examples.
- Comparison script for standard vs. adversarially trained models.

## Project Structure

```
tiny_transformer/
├── checkpoints/        # Saved model checkpoints
├── data/               # Placeholder for dataset (not included)
├── logs/               # Log files for training, evaluation, attacks
├── results/
│   ├── adversarial/      # Results from adversarial attacks
│   ├── adversarial_training/ # Results from adversarial training
│   ├── comparisons/      # Results from model comparisons
│   ├── training_evaluation/ # Results from standard training/evaluation
│   └── visualizations/   # Saved plots and visualizations
├── scripts/            # Python scripts for different tasks (train, eval, attack)
│   ├── __init__.py
│   ├── compare_models.py
│   ├── eval.py
│   ├── run.py              # Main script for standard training
│   ├── run_adv_train.py    # Main script for adversarial training
│   ├── run_attacks.py      # Main script for running attacks
│   ├── train.py
│   ├── train_adversarial.py
│   └── visualize_attacks.py
├── src/                # Source code for model, dataset, attacks, logger
│   ├── __init__.py
│   ├── attacks.py
│   ├── dataset.py
│   ├── logger.py
│   └── model.py          # Tiny Transformer model definition
├── .gitignore
├── __main__.py         # Main entry point for command-line interface
└── README.md
```

## Usage

The main entry point is `__main__.py`. Use the `--help` flag to see available commands and options:

```bash
python -m models.tiny_transformer --help
```

### Commands

- `train`: Train the Tiny Transformer model.
- `adv_train`: Train the model using FGSM adversarial training.
- `attack`: Perform FGSM attacks on a trained model.
- `visualize`: Visualize attack results (epsilon vs accuracy, adversarial examples).
- `compare`: Compare the robustness of a standard vs. adversarially trained model.

**Example Usage:**

1.  **Train the model:**
    ```bash
    python -m models.tiny_transformer train --data_path /path/to/your/dataset --epochs 20 --batch_size 512 --learning_rate 1e-4
    ```

2.  **Train with adversarial examples:**
    ```bash
    python -m models.tiny_transformer adv_train --data_path /path/to/your/dataset --epochs 20 --batch_size 512 --learning_rate 1e-4 --epsilon 0.03 --mix_ratio 0.5
    ```

3.  **Run FGSM attacks:**
    ```bash
    python -m models.tiny_transformer attack --data_dir /path/to/your/dataset --checkpoint checkpoints/tiny_transformer_model_TIMESTAMP.pth --epsilons 0.01 0.03 0.05 0.1 --save_adv_examples
    ```

4.  **Visualize results:**
    ```bash
    python -m models.tiny_transformer visualize --results_dir results/adversarial/ --adv_dataset results/adversarial/fgsm_eps0.05/adversarial_dataset.pt
    ```

5.  **Compare models:**
    ```bash
    python -m models.tiny_transformer compare --standard_model checkpoints/tiny_transformer_model_TIMESTAMP.pth --adversarial_model checkpoints/tiny_transformer_adversarial_eps0.03_mix0.5_TIMESTAMP.pth --data_path /path/to/your/dataset
    ```