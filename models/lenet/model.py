import torch
import torch.nn as nn
import torch.nn.functional as F
import os

class LeNet(nn.Module):
    def __init__(self, input_channels=3, num_classes=2, dropout_rate=0.5):
        super(LeNet, self).__init__()
        
        # First Convolutional Block
        self.conv1 = nn.Conv2d(input_channels, 6, kernel_size=5, stride=1, padding=2)
        self.bn1 = nn.BatchNorm2d(6)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Second Convolutional Block
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5, stride=1, padding=2)
        self.bn2 = nn.BatchNorm2d(16)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Fully Connected Layers
        self.fc1 = nn.Linear(16 * 12 * 12, 120)  # Adjusted for 50x50 input
        self.dropout1 = nn.Dropout(dropout_rate)
        self.fc2 = nn.Linear(120, 84)
        self.dropout2 = nn.Dropout(dropout_rate)
        self.fc3 = nn.Linear(84, num_classes)

    def forward(self, x):
        # First Convolutional Block
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.pool1(x)
        
        # Second Convolutional Block
        x = self.conv2(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = self.pool2(x)
        
        # Flatten
        x = x.view(x.size(0), -1)
        
        # Fully Connected Layers
        x = F.relu(self.fc1(x))
        x = self.dropout1(x)
        x = F.relu(self.fc2(x))
        x = self.dropout2(x)
        x = self.fc3(x)
        
        return x

    def save(self, path):
        """
        Save the model to disk
        Args:
            path: Path to save the model
        """
        torch.save(self.state_dict(), path)

    def load(self, path):
        """
        Load the model from disk
        Args:
            path: Path to load the model from
        """
        if os.path.exists(path):
            self.load_state_dict(torch.load(path))
        else:
            raise FileNotFoundError(f"Model file not found at {path}") 