import torch
import torch.nn as nn
import torch.nn.functional as F

# models.py
__all__ = ['WaveModel']


class WaveModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=128, kernel_size=128, stride=1, dilation=4, padding=0)  # causal padding
        self.pool1 = nn.MaxPool1d(kernel_size=2, stride=2)
        self.drop1 = nn.Dropout(0.3)

        self.conv2 = nn.Conv1d(in_channels=128, out_channels=64, kernel_size=64, dilation=4, padding=0)  # causal
        self.pool2 = nn.MaxPool1d(kernel_size=2, stride=2)
        self.drop2 = nn.Dropout(0.3)

        self.conv3 = nn.Conv1d(in_channels=64, out_channels=32, kernel_size=32, dilation=4, padding=0)  # causal
        self.pool3 = nn.MaxPool1d(kernel_size=2, stride=2)
        self.drop3 = nn.Dropout(0.3)

        self.flatten_dim = 16*32  # check output after last pool
        self.fc1 = nn.Linear(self.flatten_dim, 16)
        self.drop4 = nn.Dropout(0.3)
        self.fc2 = nn.Linear(16, 2)
        self.temperature = nn.Parameter(torch.ones(1) * 1.0)  # learnable T

    def forward(self, x):
        # x: (batch, seq_len, channels) → PyTorch expects (batch, channels, seq_len)
        # x = x.transpose(1,2)
        pad = (self.conv1.dilation[0] * (self.conv1.kernel_size[0]-1), 0)
        x = F.pad(x, pad)
        x = F.relu(self.conv1(x))
        x = self.pool1(x)
        x = self.drop1(x)

        pad = (self.conv2.dilation[0] * (self.conv2.kernel_size[0]-1), 0)
        x = F.pad(x, pad)
        x = F.relu(self.conv2(x))
        x = self.pool2(x)
        x = self.drop2(x)

        pad = (self.conv3.dilation[0] * (self.conv3.kernel_size[0]-1), 0)
        x = F.pad(x, pad)
        x = F.relu(self.conv3(x))
        x = self.pool3(x)
        x = self.drop3(x)
        x = x.transpose(1,2)
        x = x.flatten(1)
        x = F.relu(self.fc1(x))
        x = self.drop4(x)
        x = self.fc2(x)
        return x
    
    def predict_proba(self, x):
        logits = self.forward(x)
        return F.softmax(logits, dim=1)
    
    def predict_proba_temp(self, x):
        logits = self.forward(x)
        scaled_logits = logits / self.temperature
        return F.softmax(scaled_logits, dim=1)