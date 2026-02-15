import sys
import os
from pathlib import Path
import pandas as pd
import numpy as np
import polars as pl
import torch
import matplotlib.pyplot as plt
from tqdm.auto import tqdm
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import math
from huggingface_hub import login

login(token="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")


# GluonTS / Uni2TS Imports for Moirai
try:
    from gluonts.dataset.pandas import PandasDataset
    from gluonts.dataset.split import split
    from uni2ts.model.moirai import MoiraiForecast, MoiraiModule
except ImportError:
    print("Warning: gluonts or uni2ts not installed. Moirai code will fail.")

from chronos import ChronosPipeline

# Ensure local utils are discoverable
sys.path.append(os.path.abspath('utils'))
from potsimloader import potsimloader


# Model Definitions
MODELS = {
    'chronos_tiny': {'id': 'amazon/chronos-t5-tiny', 'type': 'chronos'},
    'chronos_base': {'id': 'amazon/chronos-t5-base', 'type': 'chronos'},
    'moirai_small': {'id': 'Salesforce/moirai-1.0-R-small', 'type': 'moirai'},
    'moirai_base': {'id': 'Salesforce/moirai-1.0-R-base', 'type': 'moirai'}
}

# Check for GPU
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device to load models: {device}")

#definitions to help load models
def load_chronos_pipeline(model_id):
    return ChronosPipeline.from_pretrained(
        model_id,
        device_map=device,
        torch_dtype=torch.bfloat16
    )

def load_moirai_module(model_id):
    # Loads weights
    return MoiraiModule.from_pretrained(model_id)

def load_models(models_dict):
    loaded = {}
    for name, config in models_dict.items():
        try:
            if config['type'] == 'chronos':
                loaded[name] = load_chronos_pipeline(config['id'])
            elif config['type'] == 'moirai':
                loaded[name] = load_moirai_module(config['id'])
            print(f"Loaded {name}")
        except Exception as e:
            print(f"Failed to load {name}: {e}")
    return loaded

# loaded_models = load_models(MODELS)


#Forecasting models

# def forecast_chronos(pipeline, target_series, horizon):
#     # User provided snippet: pipe.predict(context, prediction_length=24, num_samples=100)
#     context = torch.tensor(target_series.dropna().values, dtype=torch.float32)
    
#     # Chronos predict expects input shape (batch_size, context_length)
#     # If passing single series, we need to add batch dimension? 
#     # The snippet says: `context = torch.tensor(df.iloc[:, 0].dropna().values...)` then `pipe.predict(context...)`
#     # ChronosPipeline usually handles list of tensors or tensor.
#     # If 1D tensor is passed, it might be interpreted as (1, length) or (length,).
#     # Documentation says: `context`: torch.Tensor of shape (batch_size, sequence_length) or list of tensors.
    
#     if context.ndim == 1:
#         context = context.unsqueeze(0) # (1, seq_len)
    
#     pred = pipeline.predict(context, prediction_length=horizon, num_samples=100)
#     # pred shape: (batch_size, num_samples, horizon) -> (1, 100, horizon)
    
#     median = torch.quantile(pred, 0.5, dim=1).cpu().numpy()[0]
#     low = torch.quantile(pred, 0.1, dim=1).cpu().numpy()[0]
#     high = torch.quantile(pred, 0.9, dim=1).cpu().numpy()[0]
#     return median, low, high


    
# def forecast_moirai_past_exog(module, df_hist, target_col, past_exog_cols, horizon,
#                              freq="D", context_length=40, num_samples=100):
#     # ensure datetime index
#     if "Date" in df_hist.columns and not isinstance(df_hist.index, pd.DatetimeIndex):
#         df_hist = df_hist.copy()
#         df_hist["Date"] = pd.to_datetime(df_hist["Date"])
#         df_hist = df_hist.set_index("Date")

#     df_hist = df_hist.sort_index()
#     df_hist = df_hist[~df_hist.index.duplicated(keep="last")]

#     T = len(df_hist)
#     if T < 3:
#         raise ValueError(f"Need >=3 history points, got {T}.")

#     # 1D target (T,)
#     y = df_hist[target_col].astype(float).to_numpy()

#     # past exog: (K, T)
#     x_past = df_hist[past_exog_cols].astype(float).to_numpy().T

#     ds = ListDataset(
#         [{
#             "start": df_hist.index[0],
#             "target": y,
#             "past_feat_dynamic_real": x_past,
#         }],
#         freq=freq
#     )

#     CTX = min(context_length, T)

#     model = MoiraiForecast(
#         module=module,
#         prediction_length=horizon,
#         context_length=CTX,
#         patch_size="auto",
#         num_samples=num_samples,
#         target_dim=1,
#         feat_dynamic_real_dim=0,
#         past_feat_dynamic_real_dim=len(past_exog_cols),
#     )

#     predictor = model.create_predictor(batch_size=32)
#     forecast = next(iter(predictor.predict(ds)))

#     samples = forecast.samples  # (num_samples, horizon) or (num_samples, horizon, 1)
#     if samples.ndim == 3:
#         samples = samples[:, :, 0]

#     med = np.quantile(samples, 0.5, axis=0)
#     lo  = np.quantile(samples, 0.1, axis=0)
#     hi  = np.quantile(samples, 0.9, axis=0)
#     return med, lo, hi


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=1000):
        super().__init__()
        self.dropout = nn.Dropout(dropout)

        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))  # (1, max_len, d_model)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)

class RollingTransformer1Step(nn.Module):
    """
    Input: (B, W, F+1) where last channel is y history
    Output: (B,) next-step y
    """
    def __init__(self, input_dim, d_model=128, nhead=4, num_layers=3, dim_ff=512, dropout=0.1, max_len=500):
        super().__init__()
        self.inp = nn.Linear(input_dim, d_model)
        self.pos = PositionalEncoding(d_model, dropout, max_len=max_len)

        layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead,
            dim_feedforward=dim_ff, dropout=dropout,
            batch_first=True
        )
        self.enc = nn.TransformerEncoder(layer, num_layers=num_layers)
        self.out = nn.Linear(d_model, 1)

    def forward(self, x):
        h = self.pos(self.inp(x))      # (B,W,d)
        h = self.enc(h)                # (B,W,d)
        last = h[:, -1, :]             # (B,d)
        yhat = self.out(last).squeeze(-1)  # (B,)
        return yhat


# Code that works for both versions of Moirai and Chronos

import torch

def load_chronos_any(model_id, device="auto"):
    # chronos-t5-*  -> ChronosPipeline
    # chronos-2     -> Chronos2Pipeline
    from chronos import ChronosPipeline, Chronos2Pipeline

    PipelineCls = Chronos2Pipeline if model_id == "amazon/chronos-2" else ChronosPipeline

    # torch_dtype warning: use dtype=
    return PipelineCls.from_pretrained(
        model_id,
        device_map=device,
        dtype=torch.bfloat16,
    )

def load_moirai_any(model_id):
    # moirai-1.0 -> MoiraiModule (weights-only, you wrap with MoiraiForecast later)
    # moirai-2.0 -> Moirai2Module (weights-only, you wrap with Moirai2Forecast later)
    if "moirai-2.0" in model_id:
        from uni2ts.model.moirai2 import Moirai2Module
        return Moirai2Module.from_pretrained(model_id)
    else:
        from uni2ts.model.moirai import MoiraiModule
        return MoiraiModule.from_pretrained(model_id)

def load_models_any(models_dict, device="auto"):
    loaded = {}
    for name, config in models_dict.items():
        try:
            if config["type"] == "chronos":
                loaded[name] = load_chronos_any(config["id"], device=device)
            elif config["type"] == "moirai":
                loaded[name] = load_moirai_any(config["id"])
            else:
                raise ValueError(f"Unknown type: {config['type']}")
            print(f"Loaded {name} -> {config['id']}")
        except Exception as e:
            print(f"Failed to load {name} ({config.get('id')}): {e}")
            loaded[name] = None
    return loaded
