import math
import torch
import torch.nn as nn


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.d_model = d_model
        self.max_len = max_len
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(0, self.max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, self.d_model, 2).float()
            * (-math.log(10000.0) / self.d_model)
        )
        pos = torch.zeros(self.max_len, self.d_model)
        pos[:, 0::2] = torch.sin(position * div_term)
        pos[:, 1::2] = torch.cos(position * div_term)
        pos = pos.unsqueeze(0)
        self.register_buffer("pe", pos)

    def forward(self, x):
        x = x + self.pe[:, : x.size(1), :]
        # x = torch.add(x, self.pe[:, :x.size(1), :])
        return self.dropout(x)


class EncoderOnlyTransformer(nn.Module):
    def __init__(
        self,
        input_dim,
        nhead=4,
        num_layers=2,
        d_model=128,
        dropout=0.2,
        max_seq_len=5000,
    ):
        self.input_dim = input_dim
        self.nhead = nhead
        self.num_layers = num_layers
        self.d_model = d_model
        self.dropout = dropout
        self.max_seq_len = max_seq_len
        super(EncoderOnlyTransformer, self).__init__()
        self.input_embedding = nn.Linear(self.input_dim, self.d_model)
        self.pos_encoder = PositionalEncoding(
            d_model=self.d_model, dropout=self.dropout, max_len=self.max_seq_len
        )
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.d_model,
            nhead=self.nhead,
            dim_feedforward=self.d_model * 4,
            dropout=self.dropout,
            batch_first=True,
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=self.num_layers
        )

        # Minimal decoder setup
        # Here we use a single linear layer to act as a minimal "decoder"
        self.fc_out = nn.Linear(d_model, 1)

    def forward(self, x):
        # x shape: [batch_size, seq_length, input_dim]
        x = self.input_embedding(x)
        x = self.pos_encoder(x)
        x = self.transformer_encoder(x)
        # Use the output of the last token for prediction
        return self.fc_out(x[:, -1, :])
