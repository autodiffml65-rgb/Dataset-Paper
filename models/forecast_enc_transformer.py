import math
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, Subset

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2).float()
                        * (-math.log(10000.0)/d_model))
        pe[:,0::2] = torch.sin(pos*div)
        pe[:,1::2] = torch.cos(pos*div)
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)

class ForecastTransformerEncOnly(nn.Module):
    def __init__(self, num_features, horizon,
                 d_model=64, nhead=4, enc_layers=4,
                 dim_ff=256, dropout=0.1):
        super().__init__()
        self.horizon = horizon
        self.src_emb = nn.Linear(num_features, d_model)
        self.pos_enc = PositionalEncoding(d_model, dropout)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead,
            dim_feedforward=dim_ff, dropout=dropout,
            batch_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer,
                                             num_layers=enc_layers)
        # project the last time-step's representation to H future values
        self.out = nn.Linear(d_model, horizon)

    def forward(self, src):
        # src: (B, w, F)
        s = self.pos_enc(self.src_emb(src))  # (B, w, d_model)
        enc = self.encoder(s)                # (B, w, d_model)
        last = enc[:, -1, :]                 # (B, d_model)
        y = self.out(last)                   # (B, H)
        return y.unsqueeze(-1)               # (B, H, 1)
