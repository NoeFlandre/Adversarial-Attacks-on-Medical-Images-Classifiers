import torch.nn as nn
import numpy as np

class MLPModel(nn.Module):
    def __init__(self, input_size=50*50*3, hidden_size=32, num_classes=2, dropout_rate=0.5):
        super(MLPModel, self).__init__()
        
        self.model = nn.Sequential(
            nn.Flatten(),
            nn.BatchNorm1d(input_size),
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_size // 2, num_classes)
        )
        
    def forward(self, x):
        return self.model(x)
        
    def count_parameters(self):
        """Count the number of trainable parameters in the model"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)