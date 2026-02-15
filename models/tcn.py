import torch
import torch.nn as nn

class TCNBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, dilation, dropout=0.2):
        super(TCNBlock, self).__init__()
        self.conv = nn.Conv1d(
            in_channels, out_channels,
            kernel_size=kernel_size,
            padding=(kernel_size - 1) * dilation,
            dilation=dilation
        )
        self.norm = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = self.conv(x)
        out = self.norm(out)
        out = self.relu(out)
        return self.dropout(out)


class TCN(nn.Module):
    def __init__(self, input_dim, num_channels=[32, 16, 8], kernel_size=3, dropout=0.2):
        super(TCN, self).__init__()
        self.input_dim = input_dim
        self.num_channels = num_channels
        self.kernel_size = kernel_size
        self.dropout = dropout
        layers = []
        num_levels = len(self.num_channels)
        for i in range(num_levels):
            dilation = 2 ** i
            in_channels = self.input_dim if i == 0 else self.num_channels[i-1]
            out_channels = self.num_channels[i]
            layers.append(
                TCNBlock(in_channels, out_channels, kernel_size=self.kernel_size, 
                         dilation=dilation, dropout=self.dropout)
            )
        self.network = nn.Sequential(*layers)
        self.fc = nn.Linear(self.num_channels[-1], 1)

    def forward(self, x):
        x = x.transpose(1, 2)
        x = self.network(x)
        x = x.mean(dim=2)
        return self.fc(x)