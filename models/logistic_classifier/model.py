import torch.nn as nn

class LogisticRegressionModel(nn.Module):
    def __init__(self, input_size=50*50*3, num_classes=2):
        super(LogisticRegressionModel, self).__init__()
        self.linear = nn.Linear(input_size, num_classes)

    def forward(self, x):
        x = x.view(x.size(0), -1)  # flatten
        return self.linear(x)
