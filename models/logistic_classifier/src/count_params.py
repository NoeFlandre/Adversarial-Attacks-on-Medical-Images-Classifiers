import torch
import sys
import os

# Add the parent directory to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.model import LogisticRegressionModel

# Create model instance
model = LogisticRegressionModel(input_size=50*50*3, num_classes=2)

# Count parameters
total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

print(f"LogisticRegressionModel parameters:")
print(f"Total parameters: {total_params:,}")
print(f"Trainable parameters: {trainable_params:,}")

# Print parameters for each layer
for name, param in model.named_parameters():
    print(f"{name}: {param.numel():,} parameters")