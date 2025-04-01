# TinyResNet Classifier for Breast Histopathology Images

This module implements a small ResNet-style classifier for breast histopathology images to classify between Invasive Ductal Carcinoma (IDC) and non-IDC tissues.

## Model Architecture

The TinyResNet model consists of:
- Initial convolution layer (3→24 channels)
- First residual block layer (24→36 channels)
- Second residual block layer (36→64 channels) with stride 2 for downsampling
- Third residual block layer (64→84 channels) with stride 2 for downsampling
- Global average pooling
- Fully connected layer to 2 output classes

Each residual block contains:
- Two convolutional layers with batch normalization and ReLU
- A shortcut connection that may include a 1x1 convolution if dimensions change

The model has approximately 200K trainable parameters (199,442) and provides a `count_parameters()` method that returns the total number of trainable parameters. This count is automatically logged during training, adversarial training, and model evaluation.

## Training Features

- **Early Stopping**: Training automatically stops when validation loss stops improving for 5 consecutive epochs
- **Learning Rate Scheduling**: ReduceLROnPlateau scheduler reduces learning rate when validation loss plateaus
- **Class Weighting**: Weighted loss function to handle class imbalance
- **Parameter Tracking**: Logging of model parameter count and training metrics

## Adversarial Attacks

The model is evaluated against Fast Gradient Sign Method (FGSM) adversarial attacks with various epsilon values (attack strengths).

## Usage

### Training

To train the standard TinyResNet model:

```bash
python -m models.tiny_resnet.main train --data_path data/ --epochs 10 --batch_size 64 --learning_rate 1e-4 --num_blocks 1
```

### Adversarial Training

To train the TinyResNet model with adversarial examples:

```bash
python -m models.tiny_resnet.main adv_train --data_path data/ --epochs 10 --batch_size 64 --learning_rate 1e-4 --epsilon 0.05 --mix_ratio 0.5 --num_blocks 1
```

### Running Adversarial Attacks

To evaluate the model against FGSM attacks:

```bash
python -m models.tiny_resnet.main attack --data_dir data/ --checkpoint models/tiny_resnet/checkpoints/[checkpoint_file].pth --epsilons 0.01 0.05 0.1 0.2
```

### Visualizing Results

To visualize attack results:

```bash
python -m models.tiny_resnet.main visualize --results_dir models/tiny_resnet/results/adversarial
```

### Comparing Models

To compare standard and adversarially trained models:

```bash
python -m models.tiny_resnet.main compare --standard_model models/tiny_resnet/checkpoints/[standard_model].pth --adversarial_model models/tiny_resnet/checkpoints/[adversarial_model].pth --data_path data/
```

## Results

Results are saved in the `models/tiny_resnet/results/` directory:
- Training/validation metrics and plots in `training_evaluation/`
- Adversarial attack results in `adversarial/`
- Adversarial training results in `adversarial_training/`
- Model comparison results in `comparisons/`
- Visualizations in `visualizations/`