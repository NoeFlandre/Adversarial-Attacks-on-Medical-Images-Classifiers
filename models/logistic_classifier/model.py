import torch.nn as nn

class LogisticRegressionModel(nn.Module):
    def __init__(self, input_size=50*50*3, num_classes=2, dropout_rate=0.5):
        super(LogisticRegressionModel, self).__init__()
        self.batch_norm = nn.BatchNorm1d(input_size)
        self.dropout = nn.Dropout(dropout_rate)
        self.linear = nn.Linear(input_size, num_classes)

    def forward(self, x):
        x = x.view(x.size(0), -1)  # flatten all the dimensions but not the batch size
        x = self.batch_norm(x)
        x = self.dropout(x)
        return self.linear(x)
