import torch
import torch.nn as nn


# Hochreiter, S., & Schmidhuber, J. (1997).
# Long Short-Term Memory. Neural Computation, 9(8), 1735-1780
class LSTM(nn.Module):
    def __init__(self, input_dim: int, hidden_size=64, num_layers=2, dropout=0.2):
        super(LSTM, self).__init__()
        self.input_dim = input_dim
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout

        self.lstm = nn.LSTM(
            input_size=self.input_dim,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            batch_first=True,
            dropout=self.dropout,
        )
        self.fc_layers = nn.Sequential(
            nn.Linear(in_features=self.hidden_size, out_features=self.hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(p=self.dropout),
            nn.Linear(in_features=self.hidden_size // 2, out_features=1),
        )
        self.layer_norm = nn.LayerNorm(self.hidden_size)

    def forward(self, x):
        # x shape: [batch_size, sequence_length, features]
        lstm_out, _ = self.lstm(x)
        last_hidden = lstm_out[:, -1, :]
        normalized = self.layer_norm(last_hidden)
        output = self.fc_layers(normalized)
        return output
