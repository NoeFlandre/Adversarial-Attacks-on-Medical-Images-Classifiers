# Adversarial Attacks on Medical Image Classifiers

This project explores the vulnerability of medical image classifiers to adversarial attacks and investigates adversarial training as a defense mechanism. The project focuses on breast histopathology images, classifying between Invasive Ductal Carcinoma (IDC) and non-IDC tissues.

## Project Structure

```
.
├── data/                      # Dataset files
├── models/                    # Model implementations
│   ├── logistic_classifier/   # Logistic Regression classifier
│   ├── lenet/                 # LeNet CNN classifier
│   ├── mlp/                   # Multi-Layer Perceptron classifier
│   └── tiny_resnet/           # TinyResNet CNN classifier
├── notebooks/                 # Jupyter notebooks for data exploration
└── reports/                   # Analysis reports and findings
```

## Implemented Models

### 1. Logistic Classifier
A simple logistic regression model that serves as a baseline.

### 2. LeNet
A CNN architecture based on the classic LeNet-5 design, adapted for image classification.

### 3. MLP (Multi-Layer Perceptron)
A fully connected neural network with configurable hidden layers and dropout for regularization.

### 4. TinyResNet
A small ResNet-style CNN with approximately 200K parameters, using residual blocks for improved gradient flow.

## Training Features

All neural network models include:
- **Early Stopping**: Training automatically stops when validation loss stops improving
- **Learning Rate Scheduling**: Reduces learning rate when validation loss plateaus
- **Class Weighting**: Weighted loss functions to handle class imbalance
- **Parameter Counting**: Tracking and logging of model complexity
- **Comprehensive Metrics**: Accuracy, precision, recall, F1-score, ROC curves

## Adversarial Attacks

All models are evaluated against the Fast Gradient Sign Method (FGSM) attack with various epsilon values (attack strength). Visualizations and performance metrics are generated for each attack scenario.

## Adversarial Training

Models can be trained using a mixture of clean and adversarially perturbed examples to improve robustness against attacks. The project includes scripts to compare standard and adversarially trained models.

## Key Features

- Model parameter counting and logging
- Comprehensive evaluation metrics (accuracy, precision, recall, F1-score, ROC curves)
- Visualization of adversarial examples and their effects
- Modular architecture to easily add new model types
- Command-line interfaces for all models with consistent parameters

## If using a VM 

You can transfer your data to the VM using this command

```bash
rsync -avzP /Users/noeflandre/Adversarial-Attacks-on-Medical-Images-Classifiers cs736:/workspace/
```

## Running the Code

Each model can be trained, evaluated, and tested against adversarial attacks using its respective command-line interface. For example, to train the TinyResNet model:

```bash
python -m models.tiny_resnet.main train --data_path data/ --epochs 10 --batch_size 64
```

To run adversarial attacks on a trained model:

```bash
python -m models.tiny_resnet.main attack --data_dir data/ --checkpoint models/tiny_resnet/checkpoints/[checkpoint_file].pth
```

Refer to the individual model README files for detailed usage instructions.

## Results

Results for each model are stored in their respective directories under `models/[model_name]/results/`. These include:
- Training/validation metrics and plots
- Adversarial attack results
- Adversarial training results
- Model comparisons
- Visualizations

## Dependencies

See `requirements.txt` for the complete list of dependencies.
