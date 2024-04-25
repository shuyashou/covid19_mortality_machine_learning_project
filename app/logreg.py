import torch
import torch.nn as nn
class LogisticRegressionModel(nn.Module):
    def __init__(self, input_size):
        super(LogisticRegressionModel, self).__init__()
        self.linear = nn.Linear(input_size, 1)

    def forward(self, x):
        logits = self.linear(x)
        probabilities = torch.sigmoid(logits)
        return probabilities