# MLP Classifier for Breast Histopathology Images

This module implements a Multi-Layer Perceptron (MLP) classifier for breast histopathology images to classify between Invasive Ductal Carcinoma (IDC) and non-IDC tissues.

## Model Architecture

The MLP model consists of:
- Input layer: Flattened 50x50x3 RGB images (7,500 features)
- Batch normalization layer
- First hidden layer: 32 neurons with ReLU activation
- Dropout layer (0.5 rate)
- Second hidden layer: 16 neurons with ReLU activation  
- Dropout layer (0.5 rate)
- Output layer: 2 neurons for binary classification

The model provides a `count_parameters()` method that returns the total number of trainable parameters. This count is automatically logged during training, adversarial training, and model evaluation.

## Adversarial Attacks

The model is evaluated against Fast Gradient Sign Method (FGSM) adversarial attacks with various epsilon values (attack strengths).

## Usage

### Training

To train the standard MLP model:

```bash
python -m models.mlp.main train --data_path data/ --epochs 10 --batch_size 1024 --learning_rate 1e-5 --hidden_size 32
```

### Adversarial Training

To train the MLP model with adversarial examples:

```bash
python -m models.mlp.main adv_train --data_path data/ --epochs 10 --batch_size 1024 --learning_rate 1e-5 --epsilon 0.05 --mix_ratio 0.5 --hidden_size 32
```

### Running Adversarial Attacks

To evaluate the model against FGSM attacks:

```bash
python -m models.mlp.main attack --data_dir data/ --checkpoint models/mlp/checkpoints/[checkpoint_file].pth --epsilons 0.01 0.05 0.1 0.2
```

### Visualizing Results

To visualize attack results:

```bash
python -m models.mlp.main visualize --results_dir models/mlp/results/adversarial
```

### Comparing Models

To compare standard and adversarially trained models:

```bash
python -m models.mlp.main compare --standard_model models/mlp/checkpoints/[standard_model].pth --adversarial_model models/mlp/checkpoints/[adversarial_model].pth --data_path data/
```

## Results

Results are saved in the `models/mlp/results/` directory:
- Training/validation metrics and plots in `training_evaluation/`
- Adversarial attack results in `adversarial/`
- Adversarial training results in `adversarial_training/`
- Model comparison results in `comparisons/`
- Visualizations in `visualizations/`