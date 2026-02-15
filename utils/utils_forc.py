# System and other standard libraries
import os
import pickle
import random
import time
import re
import itertools
import json
from pathlib import Path

# Data processing, metrics and plot libraries
import polars as pl
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import r2_score, mean_absolute_error, root_mean_squared_error

# Pytorch and related libraries
import torch
import torch.nn as nn
import torch.optim as optim
from forecasting_models import*
from torch.utils.data import DataLoader, TensorDataset
import torch.multiprocessing as mp
from torchinfo import summary

# Custom packages 
from utils import preprocessing as ppsr
from utils.potsimloader import potsimloader as psl
from utils import split
from models import linearregression, mlp, cnn, tcn, lstm, transformer,forecast_enc_transformer
from sklearn.preprocessing import MinMaxScaler
from utils import scaler
from training import train
from testing import evaluate

# Enabling polars global string cache, for more informaiton read below:
# https://docs.pola.rs/api/python/stable/reference/api/polars.StringCache.html
pl.enable_string_cache()


#Hugging face login
from huggingface_hub import login

login(token="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")


def get_unique_scenarios(df):
    cols = ["Treatment", "PlantingDay", "NFirstApp", "IrrgDep", "IrrgThresh", "Year"]
    for col in cols:
        vals = df[col].dropna().unique()
        print(f"{col}: {vals[:10]}")


def get_one_scenario(df, year=2021, treatment="56-56-56", planting_day=29, n_first_app="Npl", irr_dep=30, irr_thresh=70):
    return df[(df['Year'] == year) & (df['Treatment'] == treatment) & (df['PlantingDay'] == planting_day) & (df['NFirstApp'] == n_first_app) & (df['IrrgDep'] == irr_dep) & (df['IrrgThresh'] == irr_thresh)]

def visualize_interaction(df,var1="NLeach",var2="NTotL1", title_suffix=""):
    """
    Plots two targets on separate axes + covariates on a third axis.
    Assumes df has: 'DayAfterPlant', 'NLeach', 'NTotL1', and covariates
    like 'Rain', 'SolarRad', 'AirTempC', 'NApp' (optional).
    """

    fig, ax1 = plt.subplots(figsize=(12, 6))

    # --- Axis 1 (LEFT): Target 1 = NLeach ---
    if var1 not in df.columns:
        print("NLeach column not found in dataframe.")
        return
    ax1.plot(df["DayAfterPlant"], df[var1],
             color="black", linewidth=2.5, label="NLeach (Target)")
    ax1.set_xlabel("Day After Plant")
    ax1.set_ylabel(var1+" (kg/ha)", color="black", fontweight="bold")
    ax1.tick_params(axis="y", labelcolor="black")
    ax1.grid(True, which="major", linestyle="--", alpha=0.5)

    # --- Axis 2 (RIGHT): Target 2 = NTotL1 ---
    ax2 = ax1.twinx()
    if var2 not in df.columns:
        print("NTotL1 column not found in dataframe.")
        return
    ax2.plot(df["DayAfterPlant"], df[var2],
             color="purple", linewidth=2.5, linestyle="-", label="NTotL1 (Target)")
    ax2.set_ylabel(var2+" (kg/ha)", color="purple", fontweight="bold")
    ax2.tick_params(axis="y", labelcolor="purple")

    # --- Axis 3 (EXTRA RIGHT, OFFSET): Covariates ---
    ax3 = ax1.twinx()
    ax3.spines["right"].set_position(("axes", 1.12))  # push outward
    ax3.spines["right"].set_visible(True)

    # Covariates (all optional)
    if "Rain" in df.columns:
        ax3.bar(df["DayAfterPlant"], df["Rain"],
                alpha=0.25, label="Rain", width=1.0)

    if "NApp" in df.columns:
        n_app_events = df[df["NApp"] > 0]
        if not n_app_events.empty:
            ax3.stem(n_app_events["DayAfterPlant"], n_app_events["NApp"],
                     linefmt="r-", markerfmt="ro", basefmt=" ", label="N App")

    if "AirTempC" in df.columns:
        ax3.plot(df["DayAfterPlant"], df["AirTempC"],
                 linestyle="--", linewidth=1.5, label="Air Temp")

    if "SolarRad" in df.columns:
        ax3.plot(df["DayAfterPlant"], df["SolarRad"],
                 linestyle=":", linewidth=2.0, label="Solar Rad")

    ax3.set_ylabel("Covariates", fontweight="bold")
    ax3.tick_params(axis="y")
    #setting same y axis level
    ax1.set_ylim(bottom=0)
    ax2.set_ylim(bottom=0)
    ax3.set_ylim(bottom=0)

    # --- Combined legend across all three axes ---
    h1, l1 = ax1.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    h3, l3 = ax3.get_legend_handles_labels()
    ax1.legend(h1 + h2 + h3, l1 + l2 + l3, loc="upper right", shadow=True)

    plt.title(f"{title_suffix}".strip())
    plt.tight_layout()
    plt.show()

def get_loaded_forc_mdl( ckpt_path):
 
    # device
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # load checkpoint
    ckpt = torch.load(ckpt_path, map_location=device) 
    model = RollingTransformer1Step(
        input_dim=5,
        d_model=128,
        nhead=4,
        num_layers=3,
        dim_ff=512,
        dropout=0.1,
        max_len=500,   # can be >= window used in training
    ).to(device)


    # load weights
    model.load_state_dict(ckpt["state_dict"])
    return model

def get_ip_op(data,feat_cols,targ_var):
    #This definition will divide the data such that every example covers crop cycle length of features, the final 
    #result having the shape (#examples,#crop cycle or episode length, #features)
    inp_data=[]
    op_data=[]
    counter=0
    for yr in data['Year'].unique():
        for tmt in data['Treatment'].unique():
              for days in data['PlantingDay'].unique():
                    for irrgdep in data['IrrgDep'].unique():
                        for irrthresh in data['IrrgThresh'].unique():
                            counter+=1
                            # print(counter,"Year:",yr,"| Treatment:",tmt,"| Planting Day:",days,"| Irrigation Dep:",irrgdep,"| Irrg Threshold:",irrthresh)
                            base=data[(data['Year']==yr)&(data['Treatment']==tmt)&(data['PlantingDay']==days)&(data['IrrgDep']==irrgdep)&(data['IrrgThresh']==irrthresh)]
                            inp_data.append(base[feat_cols].to_numpy())
                            op_data.append(base[targ_var].to_numpy())
                   
              
    return inp_data,op_data

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def prepare_timeseries_tensors(
    df_forc: pd.DataFrame,
    covariates=("NApp", "Rain", "SolarRad", "AirTempC"),
    target="NLeach",
    time_col="DayAfterPlant",
    scenario_cols=("Year", "Treatment", "NFirstApp", "PlantingDay", "IrrgDep", "IrrgThresh"),
    seq_len=119,
    test_size=0.2,
    random_state=42,
    drop_incomplete=True,
):
    """
    Returns:
      df_model: modeling df with scenario + time + covariates + target
      X_train_s, X_test_s: scaled tensors (n_cycles, seq_len, n_covariates)
      y_train_s, y_test_s: scaled targets (n_cycles, seq_len)
      x_scaler, y_scaler: fitted scalers (fit on train only)
      scen_train, scen_test: scenario metadata aligned with X/y first dimension
    """

    # -------- 1) Build modeling dataframe (only needed columns) --------
    covariates = list(covariates)
    scenario_cols = list(scenario_cols)

    needed_cols = scenario_cols + [time_col] + covariates + [target]
    missing = [c for c in needed_cols if c not in df_forc.columns]
    if missing:
        raise ValueError(f"Missing columns in df_forc: {missing}")

    df_model = df_forc.loc[:, needed_cols].copy()

    # Keep only valid rows (optional; adjust as you like)
    # Typically you do NOT want to drop NaNs in covariates blindly if you plan to impute.
    # Here we drop rows where target is NaN because training needs it.
    df_model = df_model.dropna(subset=[target])

    # Sort by scenario + time for consistent ordering
    df_model.sort_values(scenario_cols + [time_col], inplace=True, kind="mergesort")

    # -------- 2) Filter to complete cycles of length seq_len (optional but recommended) --------
    # This prevents ragged sequences.
    grp_sizes = df_model.groupby(scenario_cols, sort=False).size()
    if drop_incomplete:
        valid_scenarios = grp_sizes[grp_sizes == seq_len].index
        df_model = df_model.set_index(scenario_cols).loc[valid_scenarios].reset_index()
    else:
        # If not dropping, you'll need padding logic later.
        pass

    # Recompute after filtering
    df_model.sort_values(scenario_cols + [time_col], inplace=True, kind="mergesort")

    # -------- 3) Build unique scenario table and split by scenario (IMPORTANT: prevents leakage) --------
    scen_df = df_model[scenario_cols].drop_duplicates(ignore_index=True)

    scen_train, scen_test = train_test_split(
        scen_df, test_size=test_size, random_state=random_state, shuffle=True
    )

    # Make scenario keys hashable for filtering
    train_keys = set(map(tuple, scen_train.to_numpy()))
    test_keys  = set(map(tuple, scen_test.to_numpy()))

    def in_keys(row):
        return tuple(row) in train_keys

    # Efficient filtering using merge (recommended)
    df_train = df_model.merge(scen_train, on=scenario_cols, how="inner")
    df_test  = df_model.merge(scen_test,  on=scenario_cols, how="inner")

    # -------- 4) Convert each split into tensors (n_cycles, seq_len, n_features) --------
    def to_tensors(df_split: pd.DataFrame):
        # Assumes each scenario has exactly seq_len rows after filtering
        # Use groupby and stack in the sorted order.
        groups = []
        y_groups = []
        scen_rows = []

        for scen_key, g in df_split.groupby(scenario_cols, sort=False):
            g = g.sort_values(time_col, kind="mergesort")
            # Optional: enforce time index length
            if len(g) != seq_len:
                continue
            groups.append(g[covariates].to_numpy())
            y_groups.append(g[target].to_numpy())
            scen_rows.append(scen_key)

        X = np.stack(groups, axis=0)  # (n_cycles, seq_len, n_covariates)
        y = np.stack(y_groups, axis=0)  # (n_cycles, seq_len)

        scen_meta = pd.DataFrame(scen_rows, columns=scenario_cols)
        return X, y, scen_meta

    X_train, y_train, scen_train_aligned = to_tensors(df_train)
    X_test,  y_test,  scen_test_aligned  = to_tensors(df_test)

    # -------- 5) Fit scalers on TRAIN only; transform train/test --------
    x_scaler = StandardScaler()
    y_scaler = StandardScaler()

    # X scaling (flatten cycles*time)
    X_train_2d = X_train.reshape(-1, X_train.shape[-1])
    X_test_2d  = X_test.reshape(-1, X_test.shape[-1])

    x_scaler.fit(X_train_2d)
    X_train_s = x_scaler.transform(X_train_2d).reshape(X_train.shape)
    X_test_s  = x_scaler.transform(X_test_2d).reshape(X_test.shape)

    # y scaling (flatten cycles*time)
    y_train_2d = y_train.reshape(-1, 1)
    y_test_2d  = y_test.reshape(-1, 1)

    y_scaler.fit(y_train_2d)
    y_train_s = y_scaler.transform(y_train_2d).reshape(y_train.shape)
    y_test_s  = y_scaler.transform(y_test_2d).reshape(y_test.shape)

    return (
        df_model,
        X_train_s, X_test_s,
        y_train_s, y_test_s,
        x_scaler, y_scaler,
        scen_train_aligned, scen_test_aligned
    )

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

def prepare_timeseries_tensors_inference(
    df_forc: pd.DataFrame,
    covariates=("NApp", "Rain", "SolarRad", "AirTempC"),
    target="NLeach",
    time_col="DayAfterPlant",
    scenario_cols=("Year", "Treatment", "NFirstApp", "PlantingDay", "IrrgDep", "IrrgThresh"),
    seq_len=119,
    drop_incomplete=True,
):
    """
    Builds ONE full tensor X and y (no train/test split).

    Returns:
      df_model: modeling df with scenario + time + covariates + target
      X_s: scaled covariate tensor (n_cycles, seq_len, n_covariates)
      y_s: scaled target tensor (n_cycles, seq_len)
      x_scaler, y_scaler: fitted scalers (fit on ALL data)
      scen_aligned: scenario metadata aligned with X/y first dimension
    """
    covariates = list(covariates)
    scenario_cols = list(scenario_cols)

    # -------- 1) Build modeling dataframe (only needed columns) --------
    needed_cols = scenario_cols + [time_col] + covariates + [target]
    missing = [c for c in needed_cols if c not in df_forc.columns]
    if missing:
        raise ValueError(f"Missing columns in df_forc: {missing}")

    df_model = df_forc.loc[:, needed_cols].copy()

    # Drop rows where target is NaN (training needs target)
    df_model = df_model.dropna(subset=[target])

    # Sort by scenario + time for consistent ordering
    df_model.sort_values(scenario_cols + [time_col], inplace=True, kind="mergesort")

    # -------- 2) Filter to complete cycles of length seq_len (optional) --------
    grp_sizes = df_model.groupby(scenario_cols, sort=False).size()
    if drop_incomplete:
        valid_scenarios = grp_sizes[grp_sizes == seq_len].index
        df_model = df_model.set_index(scenario_cols).loc[valid_scenarios].reset_index()
    else:
        # If not dropping, you'll need padding logic later.
        pass

    # Re-sort after filtering
    df_model.sort_values(scenario_cols + [time_col], inplace=True, kind="mergesort")

    # -------- 3) Convert to tensors (n_cycles, seq_len, n_features) --------
    groups = []
    y_groups = []
    scen_rows = []

    for scen_key, g in df_model.groupby(scenario_cols, sort=False):
        g = g.sort_values(time_col, kind="mergesort")
        if len(g) != seq_len:
            continue
        groups.append(g[covariates].to_numpy())
        y_groups.append(g[target].to_numpy())
        scen_rows.append(scen_key)

    if len(groups) == 0:
        raise ValueError(
            "No complete cycles found. Check seq_len, scenario_cols grouping, and drop_incomplete."
        )

    X = np.stack(groups, axis=0)      # (n_cycles, seq_len, n_covariates)
    y = np.stack(y_groups, axis=0)    # (n_cycles, seq_len)
    scen_aligned = pd.DataFrame(scen_rows, columns=scenario_cols)

    # -------- 4) Fit scalers on ALL data; transform --------
    x_scaler = StandardScaler()
    y_scaler = StandardScaler()

    X_2d = X.reshape(-1, X.shape[-1])
    x_scaler.fit(X_2d)
    X_s = x_scaler.transform(X_2d).reshape(X.shape)

    y_2d = y.reshape(-1, 1)
    y_scaler.fit(y_2d)
    y_s = y_scaler.transform(y_2d).reshape(y.shape)

    return df_model, X_s, y_s, x_scaler, y_scaler, scen_aligned

import math
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt


# ----------------------------
# Positional encoding
# ----------------------------
class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 1000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)  # (T, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)  # (T, 1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)  # even
        pe[:, 1::2] = torch.cos(position * div_term)  # odd
        pe = pe.unsqueeze(0)  # (1, T, d_model)

        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, d_model)
        x = x + self.pe[:, : x.size(1), :]
        return self.dropout(x)


# ----------------------------
# Dataset that injects y-history
# ----------------------------
class ExogPlusHistoryDataset(Dataset):
    """
    X: (N, T, F)
    y: (N, T)
    k: history length (0, 40, 70)
    """
    def __init__(self, X: np.ndarray, y: np.ndarray, k: int):
        self.X = X.astype(np.float32)
        self.y = y.astype(np.float32)
        self.k = int(k)
        self.T = self.X.shape[1]

        # Precompute mask: (T,)
        mask = np.zeros((self.T,), dtype=np.float32)
        if self.k > 0:
            mask[: self.k] = 1.0
        self.mask = mask

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        x = self.X[idx]            # (T, F)
        y = self.y[idx]            # (T,)
        y_hist = np.zeros_like(y)  # (T,)
        if self.k > 0:
            y_hist[: self.k] = y[: self.k]

        hist_mask = self.mask      # (T,)
        return x, y, y_hist, hist_mask


# ----------------------------
# Encoder-only transformer that predicts y for all time steps
# ----------------------------
class EncoderOnlyForecaster(nn.Module):
    def __init__(
        self,
        input_dim: int,       # F + 2 (y_hist + hist_mask)
        d_model: int = 128,
        nhead: int = 4,
        num_layers: int = 3,
        dim_ff: int = 512,
        dropout: float = 0.1,
        max_len: int = 500,
    ):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, d_model)
        self.pos_enc = PositionalEncoding(d_model, dropout=dropout, max_len=max_len)

        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_ff,
            dropout=dropout,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=num_layers)

        # Predict y at each timestep
        self.out = nn.Linear(d_model, 1)

    def forward(self, x_in: torch.Tensor) -> torch.Tensor:
        # x_in: (B, T, input_dim)
        h = self.input_proj(x_in)      # (B, T, d_model)
        h = self.pos_enc(h)            # (B, T, d_model)
        h = self.encoder(h)            # (B, T, d_model)
        yhat = self.out(h).squeeze(-1) # (B, T)
        return yhat


# ----------------------------
# Masked loss (only forecast region t>=k)
# ----------------------------
def masked_mse(yhat: torch.Tensor, y: torch.Tensor, hist_mask: torch.Tensor):
    """
    hist_mask: 1 where history is provided (t<k), 0 otherwise
    We want loss on forecast region => weight = 1 - hist_mask
    """
    w = 1.0 - hist_mask  # (B, T)
    # avoid division by 0
    denom = torch.clamp(w.sum(), min=1.0)
    return ((yhat - y) ** 2 * w).sum() / denom


# ----------------------------
# Train one model for a given k
# ----------------------------
def train_one_model(
    X_train_s, y_train_s, X_test_s, y_test_s,
    k: int,
    *,
    d_model=128, nhead=4, num_layers=3, dim_ff=512, dropout=0.1,
    lr=1e-3, weight_decay=0.0,
    batch_size=64, epochs=30,
    grad_clip=1.0,
    device=None,
    seed=42
):
    torch.manual_seed(seed)
    np.random.seed(seed)

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    T = X_train_s.shape[1]
    F = X_train_s.shape[2]
    input_dim = F + 2  # + y_hist + hist_mask

    train_ds = ExogPlusHistoryDataset(X_train_s, y_train_s, k=k)
    test_ds  = ExogPlusHistoryDataset(X_test_s,  y_test_s,  k=k)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=False)
    test_loader  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False, drop_last=False)

    model = EncoderOnlyForecaster(
        input_dim=input_dim,
        d_model=d_model,
        nhead=nhead,
        num_layers=num_layers,
        dim_ff=dim_ff,
        dropout=dropout,
        max_len=max(500, T + 5),
    ).to(device)

    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    history = {"train_loss": [], "test_loss": []}

    for ep in range(1, epochs + 1):
        # ---- train ----
        model.train()
        tr_losses = []
        for x, y, y_hist, hist_mask in train_loader:
            x = torch.tensor(x, device=device)                 # (B,T,F)
            y = torch.tensor(y, device=device)                 # (B,T)
            y_hist = torch.tensor(y_hist, device=device)       # (B,T)
            # hist_mask = torch.tensor(hist_mask, device=device) # (T,)
            # hist_mask = hist_mask.unsqueeze(0).repeat(x.size(0), 1)  # (B,T)
            hist_mask = torch.tensor(hist_mask, device=device)  # already (B, T)


            x_in = torch.cat([x, y_hist.unsqueeze(-1), hist_mask.unsqueeze(-1)], dim=-1)  # (B,T,F+2)

            opt.zero_grad(set_to_none=True)
            yhat = model(x_in)
            loss = masked_mse(yhat, y, hist_mask)
            loss.backward()
            if grad_clip is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            opt.step()
            tr_losses.append(loss.item())

        # ---- eval ----
        model.eval()
        te_losses = []
        with torch.no_grad():
            for x, y, y_hist, hist_mask in test_loader:
                x = torch.tensor(x, device=device)
                y = torch.tensor(y, device=device)
                y_hist = torch.tensor(y_hist, device=device)
                # hist_mask = torch.tensor(hist_mask, device=device)
                # hist_mask = hist_mask.unsqueeze(0).repeat(x.size(0), 1)
                hist_mask = torch.tensor(hist_mask, device=device)  # already (B, T)

                x_in = torch.cat([x, y_hist.unsqueeze(-1), hist_mask.unsqueeze(-1)], dim=-1)
                yhat = model(x_in)
                loss = masked_mse(yhat, y, hist_mask)
                te_losses.append(loss.item())

        history["train_loss"].append(float(np.mean(tr_losses)))
        history["test_loss"].append(float(np.mean(te_losses)))

        print(f"[k={k:>2}] epoch {ep:>3}/{epochs} | train {history['train_loss'][-1]:.5f} | test {history['test_loss'][-1]:.5f}")

    return model, history


# ----------------------------
# Train the 3-model suite
# ----------------------------
def train_three_forecasters(
    X_train_s, y_train_s, X_test_s, y_test_s,
    ks=(0, 40, 70),
    **hparams
):
    models = {}
    histories = {}

    for k in ks:
        m, h = train_one_model(X_train_s, y_train_s, X_test_s, y_test_s, k=k, **hparams)
        models[k] = m
        histories[k] = h

    # Plot losses
    plt.figure(figsize=(10, 5))
    for k in ks:
        plt.plot(histories[k]["train_loss"], label=f"train k={k}")
        plt.plot(histories[k]["test_loss"],  label=f"test  k={k}", linestyle="--")
    plt.xlabel("Epoch")
    plt.ylabel("Masked MSE (forecast region only)")
    plt.title("Train/Test Loss Curves (3 forecasting setups)")
    plt.legend()
    plt.tight_layout()
    plt.show()

    return models, histories


# ----------------------------
# Inference + plot for one scenario
# ----------------------------
import numpy as np
import torch
import matplotlib.pyplot as plt

def forecast_one_and_plot(
    models: dict,
    X_test_s: np.ndarray,
    y_test_s: np.ndarray,
    y_scaler,                 # sklearn scaler used for y
    idx: int = 0,
    ks=(0, 40, 70),
    device=None,
    title="Three transformer forecasts",
    show_split_lines=True,
):
    """
    Plots:
      - True y (inverse-scaled)
      - For each k:
          * k=0: full predicted curve
          * k>0: true history (0..k-1) + predicted future (k..T-1)

    Notes:
      - Model input uses SCALED y history (since trained on scaled y).
      - Plot is in ORIGINAL units (inverse-scaled).
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    x = X_test_s[idx].astype(np.float32)        # (T, F) scaled covariates
    y_true_s = y_test_s[idx].astype(np.float32) # (T,) scaled target
    T, F = x.shape

    # Inverse-scale the true y for plotting
    y_true = y_scaler.inverse_transform(y_true_s.reshape(-1, 1)).ravel()

    plt.figure(figsize=(12, 6))
    plt.plot(np.arange(T), y_true, linewidth=2.5, label="True")

    for k in ks:
        model = models[k].to(device)
        model.eval()

        # Build scaled y history + mask for model input
        y_hist_s = np.zeros((T,), dtype=np.float32)
        mask = np.zeros((T,), dtype=np.float32)
        if k > 0:
            y_hist_s[:k] = y_true_s[:k]  # scaled history goes into model
            mask[:k] = 1.0

        x_in = np.concatenate([x, y_hist_s[:, None], mask[:, None]], axis=1)  # (T, F+2)
        x_in_t = torch.tensor(x_in[None, :, :], device=device)                # (1, T, F+2)

        with torch.no_grad():
            yhat_s = model(x_in_t).detach().cpu().numpy().ravel()  # (T,) scaled pred

        # Inverse-scale prediction for plotting
        yhat = y_scaler.inverse_transform(yhat_s.reshape(-1, 1)).ravel()

        # ---- Key correction: for k>0, show TRUE history + predicted future ----
        if k > 0:
            y_plot = y_true.copy()
            y_plot[k:] = yhat[k:]   # only forecast region comes from model
        else:
            y_plot = yhat           # full curve for exog-only model

        plt.plot(np.arange(T), y_plot, label=f"Pred (k={k})")

        if show_split_lines and k > 0:
            plt.axvline(k - 0.5, linestyle=":", linewidth=1)

    plt.xlabel("Day index (0..T-1)")
    plt.ylabel("Target (original units)")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.show()

import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import r2_score

def _predict_batches(model, X_s, y_s, k, batch_size=128, device=None):
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    model = model.to(device)
    model.eval()

    X_s = np.asarray(X_s, dtype=np.float32)
    y_s = np.asarray(y_s, dtype=np.float32)

    N, T, F = X_s.shape

    mask_1d = np.zeros((T,), dtype=np.float32)
    if k > 0:
        mask_1d[:k] = 1.0

    yhat_all = np.empty((N, T), dtype=np.float32)

    idx_loader = DataLoader(np.arange(N), batch_size=batch_size, shuffle=False, drop_last=False)

    with torch.no_grad():
        for idxs in idx_loader:
            idxs = idxs.numpy()

            x = torch.from_numpy(X_s[idxs]).to(device=device, dtype=torch.float32)  # (B,T,F)
            y = torch.from_numpy(y_s[idxs]).to(device=device, dtype=torch.float32)  # (B,T)

            y_hist = torch.zeros_like(y)
            if k > 0:
                y_hist[:, :k] = y[:, :k]

            hist_mask = torch.from_numpy(mask_1d).to(device=device, dtype=torch.float32)
            hist_mask = hist_mask.unsqueeze(0).expand(x.size(0), -1)  # (B,T)

            x_in = torch.cat([x, y_hist.unsqueeze(-1), hist_mask.unsqueeze(-1)], dim=-1)  # float32
            yhat = model(x_in)  # (B,T) float32

            yhat_all[idxs] = yhat.detach().cpu().numpy().astype(np.float32, copy=False)

    return y_s, yhat_all



def evaluate_r2_three_models(
    models: dict,
    X_train_s, y_train_s,
    X_test_s, y_test_s,
    y_scaler,
    ks=(0, 40, 70),
    batch_size=128,
    device=None
):
    """
    Computes R² for each model on train and test.
    - k=0: full horizon
    - k>0: only forecast part (t>=k)

    Returns:
      results: dict like:
        results[k]["train_r2"], results[k]["test_r2"]
    """
    results = {}

    for k in ks:
        # ---- TRAIN ----
        y_true_s_tr, y_pred_s_tr = _predict_batches(
            models[k], X_train_s, y_train_s, k=k,
            batch_size=batch_size, device=device
        )

        # ---- TEST ----
        y_true_s_te, y_pred_s_te = _predict_batches(
            models[k], X_test_s, y_test_s, k=k,
            batch_size=batch_size, device=device
        )

        # Inverse-scale to original units for interpretability
        # reshape(-1,1) -> inverse -> reshape back
        y_true_tr = y_scaler.inverse_transform(y_true_s_tr.reshape(-1, 1)).reshape(y_true_s_tr.shape)
        y_pred_tr = y_scaler.inverse_transform(y_pred_s_tr.reshape(-1, 1)).reshape(y_pred_s_tr.shape)

        y_true_te = y_scaler.inverse_transform(y_true_s_te.reshape(-1, 1)).reshape(y_true_s_te.shape)
        y_pred_te = y_scaler.inverse_transform(y_pred_s_te.reshape(-1, 1)).reshape(y_pred_s_te.shape)

        # Mask region: forecast part only for k>0
        if k > 0:
            y_true_tr_eval = y_true_tr[:, k:].ravel()
            y_pred_tr_eval = y_pred_tr[:, k:].ravel()

            y_true_te_eval = y_true_te[:, k:].ravel()
            y_pred_te_eval = y_pred_te[:, k:].ravel()
        else:
            y_true_tr_eval = y_true_tr.ravel()
            y_pred_tr_eval = y_pred_tr.ravel()

            y_true_te_eval = y_true_te.ravel()
            y_pred_te_eval = y_pred_te.ravel()

        # R²
        train_r2 = r2_score(y_true_tr_eval, y_pred_tr_eval)
        test_r2  = r2_score(y_true_te_eval, y_pred_te_eval)

        results[k] = {"train_r2": float(train_r2), "test_r2": float(test_r2)}

    return results

import numpy as np
import torch
import matplotlib.pyplot as plt

def run_forecast(
    model,
    X_s, y_s,
    y_scaler,
    k: int,
    idx: int = 0,
    device: str | None = None,
    plot: bool = False,
    title: str | None = None,
    combine_history: bool = True,
):
    """
    Run inference for one scenario on a model that expects inputs [X, y_hist, mask].

    Parameters
    ----------
    model : torch.nn.Module
        One of the trained/loaded models (k=0,40,70).
    X_s : np.ndarray
        Scaled covariates, shape (N, T, F).
    y_s : np.ndarray
        Scaled target, shape (N, T).
    y_scaler : sklearn scaler
        Scaler fitted on train y (used for inverse transform).
    k : int
        History length used by the model (0, 40, 70).
    idx : int
        Which sample to run from X_s/y_s.
    device : str or None
        "cuda" or "cpu". If None, auto.
    plot : bool
        If True, plot prediction vs truth.
    title : str or None
        Plot title.
    combine_history : bool
        If True and k>0, output curve uses true history for t<k and model forecast for t>=k.
        If False, returns full model output curve (note: history part is unconstrained).

    Returns
    -------
    out : dict with keys:
        - y_true_scaled: (T,)
        - y_pred_scaled: (T,)
        - y_true: (T,) inverse-scaled
        - y_pred: (T,) inverse-scaled raw model output
        - y_plot: (T,) inverse-scaled curve used for plotting (may splice history)
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    model = model.to(device)
    model.eval()

    x = np.asarray(X_s[idx], dtype=np.float32)   # (T,F)
    y = np.asarray(y_s[idx], dtype=np.float32)   # (T,)
    T, F = x.shape

    # Build scaled history + mask
    y_hist = np.zeros((T,), dtype=np.float32)
    mask = np.zeros((T,), dtype=np.float32)
    if k > 0:
        y_hist[:k] = y[:k]
        mask[:k] = 1.0

    # Model input: (1, T, F+2)
    x_in = np.concatenate([x, y_hist[:, None], mask[:, None]], axis=1)  # (T, F+2)
    x_in_t = torch.from_numpy(x_in).unsqueeze(0).to(device=device, dtype=torch.float32)

    with torch.no_grad():
        y_pred_scaled = model(x_in_t).detach().cpu().numpy().ravel().astype(np.float32)

    # Inverse-scale for human-readable values
    y_true = y_scaler.inverse_transform(y.reshape(-1, 1)).ravel()
    y_pred = y_scaler.inverse_transform(y_pred_scaled.reshape(-1, 1)).ravel()

    # For plotting / final curve: splice true history if requested
    if combine_history and k > 0:
        y_plot = y_true.copy()
        y_plot[k:] = y_pred[k:]
    else:
        y_plot = y_pred

    if plot:
        plt.figure(figsize=(12, 6))
        plt.plot(np.arange(T), y_true, linewidth=2.5, label="True")
        plt.plot(np.arange(T), y_plot, linewidth=2.0, label=f"Pred (k={k})")
        if k > 0:
            plt.axvline(k - 0.5, linestyle=":", linewidth=1)
        plt.xlabel("Day index")
        plt.ylabel("Target (original units)")
        plt.title(title if title is not None else f"Forecast (k={k}) sample idx={idx}")
        plt.legend()
        plt.tight_layout()
        plt.show()

    return {
        "y_true_scaled": y,
        "y_pred_scaled": y_pred_scaled,
        "y_true": y_true,
        "y_pred": y_pred,
        "y_plot": y_plot,
    }

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import math

class SlidingWindowDataset(Dataset):
    """
    Builds (context -> next-step) samples from (X, y).
    X: (N, T, F)  scaled
    y: (N, T)     scaled
    window: context length W
    """
    def __init__(self, X, y, window=30):
        self.X = np.asarray(X, dtype=np.float32)
        self.y = np.asarray(y, dtype=np.float32)
        self.W = int(window)
        self.N, self.T, self.F = self.X.shape

        # total samples per series: (T - W)
        self.samples_per_series = self.T - self.W
        if self.samples_per_series <= 0:
            raise ValueError(f"window={window} too large for T={self.T}")

    def __len__(self):
        return self.N * self.samples_per_series

    def __getitem__(self, idx):
        n = idx // self.samples_per_series
        t0 = idx % self.samples_per_series
        t1 = t0 + self.W

        # context window [t0, t1)
        x_ctx = self.X[n, t0:t1, :]         # (W, F)
        y_ctx = self.y[n, t0:t1]            # (W,)

        # next-step target at time t1
        y_next = self.y[n, t1]              # scalar

        # model input tokens: [X, y] per timestep
        inp = np.concatenate([x_ctx, y_ctx[:, None]], axis=1)  # (W, F+1)
        return inp, np.float32(y_next)

def train_rolling_model(
    X_train_s, y_train_s, X_test_s, y_test_s,
    *,
    window=30,
    d_model=128, nhead=4, num_layers=3, dim_ff=512, dropout=0.1,
    lr=1e-3, weight_decay=1e-4,
    batch_size=256, epochs=30,
    grad_clip=1.0,
    device=None,
    seed=42
):
    torch.manual_seed(seed)
    np.random.seed(seed)

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    train_ds = SlidingWindowDataset(X_train_s, y_train_s, window=window)
    test_ds  = SlidingWindowDataset(X_test_s,  y_test_s,  window=window)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=False)
    test_loader  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False, drop_last=False)

    F = X_train_s.shape[-1]
    model = RollingTransformer1Step(
        input_dim=F + 1,
        d_model=d_model, nhead=nhead, num_layers=num_layers,
        dim_ff=dim_ff, dropout=dropout, max_len=max(500, window + 5)
    ).to(device)

    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fn = nn.MSELoss()

    hist = {"train": [], "test": []}

    for ep in range(1, epochs + 1):
        model.train()
        tr = []
        for xb, yb in train_loader:
            xb = xb.to(device=device, dtype=torch.float32)
            yb = yb.to(device=device, dtype=torch.float32)

            opt.zero_grad(set_to_none=True)
            pred = model(xb)
            loss = loss_fn(pred, yb)
            loss.backward()
            if grad_clip is not None:
                nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            opt.step()
            tr.append(loss.item())

        model.eval()
        te = []
        with torch.no_grad():
            for xb, yb in test_loader:
                xb = xb.to(device=device, dtype=torch.float32)
                yb = yb.to(device=device, dtype=torch.float32)
                pred = model(xb)
                te.append(loss_fn(pred, yb).item())

        hist["train"].append(float(np.mean(tr)))
        hist["test"].append(float(np.mean(te)))
        print(f"epoch {ep:>3}/{epochs} | train {hist['train'][-1]:.5f} | test {hist['test'][-1]:.5f}")

    # plot losses
    plt.figure(figsize=(9,4))
    plt.plot(hist["train"], label="train")
    plt.plot(hist["test"], label="test")
    plt.xlabel("Epoch")
    plt.ylabel("MSE (1-step)")
    plt.title("Rolling Transformer training curves")
    plt.legend()
    plt.tight_layout()
    plt.show()

    return model, hist

def rolling_forecast_chunks(
    model,
    x_s, y_s,
    *,
    window=30,
    chunk=10,
    exog_fill="persistence",   # "persistence" or "zeros"
    device=None
):
    """
    x_s: (T,F) scaled exog for the whole season (available only up to 'now' in reality)
    y_s: (T,)  scaled true target (used only as "observed so far" when updating)
    Returns:
      yhat_s: (T,) scaled rolling forecast (stitched)
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    model = model.to(device)
    model.eval()

    x_s = np.asarray(x_s, dtype=np.float32)
    y_s = np.asarray(y_s, dtype=np.float32)
    T, F = x_s.shape

    # This will hold our stitched predictions
    yhat = np.full((T,), np.nan, dtype=np.float32)

    # We assume at time t_end we have observed up to t_end (inclusive)
    # Start after we have enough history for window
    t_obs_end = window - 1

    # Initialize "observed y" history with true y up to t_obs_end
    y_known = y_s.copy()  # in real deployment this would be the observed target so far

    while t_obs_end < T - 1:
        # Forecast the next chunk: (t_obs_end+1) .. (t_next_end)
        t_next_end = min(T - 1, t_obs_end + chunk)

        # For each step in the chunk, do 1-step prediction recursively
        y_roll = y_known.copy()

        for t in range(t_obs_end + 1, t_next_end + 1):
            # Build context window indices [t-window, t)
            t0 = t - window
            t1 = t

            # Exog window: only exog up to t_obs_end is "known" in reality.
            x_ctx = x_s[t0:t1].copy()  # (W,F)

            # fill unknown exog (t > t_obs_end) inside the context if any
            if t1 - 1 > t_obs_end:
                if exog_fill == "zeros":
                    x_ctx[(t_obs_end + 1 - t0):, :] = 0.0
                else:  # persistence
                    last_known = x_s[t_obs_end]
                    x_ctx[(t_obs_end + 1 - t0):, :] = last_known

            # y context: use observed (true) up to t_obs_end, and predicted thereafter within chunk
            y_ctx = y_roll[t0:t1]  # (W,)

            inp = np.concatenate([x_ctx, y_ctx[:, None]], axis=1)  # (W, F+1)
            inp_t = torch.from_numpy(inp[None, :, :]).to(device=device, dtype=torch.float32)

            with torch.no_grad():
                y_pred = model(inp_t).cpu().numpy().item()

            y_roll[t] = np.float32(y_pred)

        # commit predictions for this chunk
        yhat[t_obs_end + 1 : t_next_end + 1] = y_roll[t_obs_end + 1 : t_next_end + 1]

        # Now "time advances": exog up to t_next_end becomes available, and (optionally) y too
        t_obs_end = t_next_end

        # In real life: you’d replace y_known up to t_obs_end with observed measurements.
        # Here we assume you DO have the true target (from test) up to t_obs_end.
        # That’s the “correction” step.
        y_known[: t_obs_end + 1] = y_s[: t_obs_end + 1]

    # for plotting convenience: fill initial window with true values
    yhat[:window] = y_s[:window]
    return yhat

from sklearn.metrics import r2_score

def plot_rolling_forecast(
    model, X_test_s, y_test_s, y_scaler,
    idx=0, window=30, chunk=10, exog_fill="persistence",plot=True
):
    x = X_test_s[idx]  # (T,F) scaled
    y = y_test_s[idx]  # (T,)  scaled

    yhat_s = rolling_forecast_chunks(
        model, x, y,
        window=window, chunk=chunk,
        exog_fill=exog_fill
    )

    y_true = y_scaler.inverse_transform(y.reshape(-1,1)).ravel()
    y_pred = y_scaler.inverse_transform(yhat_s.reshape(-1,1)).ravel()

    # R2 over all predicted points after window
    r2 = r2_score(y_true[window:], y_pred[window:])
    if plot==True:
        plt.figure(figsize=(12,6))
        plt.plot(y_true, label="True", linewidth=2.5)
        plt.plot(y_pred, label=f"Rolling pred (chunk={chunk}, fill={exog_fill})", linewidth=2.0)
        for t in range(window, len(y_true), chunk):
            plt.axvline(t-0.5, linestyle=":", linewidth=0.8)
        plt.title(f"Rolling forecast (idx={idx}) | R2(after window)={r2:.3f}")
        plt.xlabel("Day")
        plt.ylabel("Target (original units)")
        plt.legend()
        plt.tight_layout()
        plt.show()

    return y_true, y_pred


import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

def leaching_timing_and_magnitude_metrics(
    y_true,
    y_pred,
    threshold=0.0,
    magnitude_metric="mse",   # "mse", "mae", or "r2"
    reduce="global",          # "global" or "per_sample"
    eps=1e-12,
    window=0                  # NEW: ignore [0:window) to avoid cheating in rolling forecasts
):
    """
    Metrics:
      1) Timing score (recall over true leaching days):
         |pred_event ∩ true_event| / |true_event| * 100

      2) Magnitude metric computed ONLY on intersection days (timing-correct days).
         Options: mse, mae, r2

    Inputs can be:
      - (T,) or (N,T)

    Returns:
      if reduce="global": (timing_pct, magnitude_value)
      if reduce="per_sample": (timing_pct_per_sample, magnitude_per_sample)
    """

    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    if y_true.shape != y_pred.shape:
        raise ValueError(f"Shape mismatch: y_true{y_true.shape} vs y_pred{y_pred.shape}")

    # Ensure 2D (N,T)
    if y_true.ndim == 1:
        y_true = y_true[None, :]
        y_pred = y_pred[None, :]

    # ---- NEW: drop the initial 'window' part from evaluation ----
    if window is not None and window > 0:
        if window >= y_true.shape[1]:
            raise ValueError(f"window={window} must be < sequence length {y_true.shape[1]}")
        y_true = y_true[:, window:]
        y_pred = y_pred[:, window:]

    N, T = y_true.shape

    # Binary event masks
    true_event = y_true > threshold
    pred_event = y_pred > threshold

    # Intersection (timing correct)
    inter = true_event & pred_event

    # --- Timing score (recall on true events) ---
    true_counts = true_event.sum(axis=1)               # (N,)
    inter_counts = inter.sum(axis=1)                   # (N,)

    # If a sample has 0 true event days, define timing as NaN (or 0). We'll use NaN then ignore in global.
    timing_pct = np.where(true_counts > 0, (inter_counts / (true_counts + eps)) * 100.0, np.nan)

    # --- Magnitude metric on intersection days only ---
    mag_vals = np.full((N,), np.nan, dtype=float)

    for i in range(N):
        idx = inter[i]
        if idx.sum() == 0:
            mag_vals[i] = np.nan
            continue

        yt = y_true[i, idx]
        yp = y_pred[i, idx]

        if magnitude_metric == "mse":
            mag_vals[i] = mean_squared_error(yt, yp)
        elif magnitude_metric == "mae":
            mag_vals[i] = mean_absolute_error(yt, yp)
        elif magnitude_metric == "r2":
            # r2_score requires at least 2 points and some variance to be meaningful
            if yt.size < 2 or np.allclose(np.var(yt), 0.0):
                mag_vals[i] = np.nan
            else:
                mag_vals[i] = r2_score(yt, yp)
        else:
            raise ValueError("magnitude_metric must be one of: 'mse', 'mae', 'r2'")

    if reduce == "per_sample":
        return timing_pct, mag_vals

    # global reduce: average over valid samples (ignore NaNs)
    timing_global = float(np.nanmean(timing_pct)) if np.any(~np.isnan(timing_pct)) else np.nan
    mag_global = float(np.nanmean(mag_vals)) if np.any(~np.isnan(mag_vals)) else np.nan

    return timing_global, mag_global


import numpy as np

def get_Y_true_pred_all(
    model,
    X_test_s,
    y_test_s,
    *,
    window=30,
    chunk=10,
    exog_fill="persistence",
    device=None
):
    """
    Run rolling forecasting on ALL test scenarios.

    Parameters
    ----------
    model : torch.nn.Module
        Trained rolling (1-step) Transformer.
    X_test_s : np.ndarray
        Scaled exogenous inputs, shape (N, T, F).
    y_test_s : np.ndarray
        Scaled targets, shape (N, T).
    window : int
        Context window length (must match training).
    chunk : int
        Update frequency (can vary at inference).
    exog_fill : str
        How to fill unknown future exogenous inputs ("persistence" or "zeros").
    device : str or None
        "cpu" or "cuda".

    Returns
    -------
    Y_true_all : np.ndarray
        Scaled true targets, shape (N, T).
    Y_pred_all : np.ndarray
        Scaled rolling predictions, shape (N, T).
    """

    X_test_s = np.asarray(X_test_s, dtype=np.float32)
    y_test_s = np.asarray(y_test_s, dtype=np.float32)

    N, T, F = X_test_s.shape

    Y_true_all = y_test_s.copy()
    Y_pred_all = np.full((N, T), np.nan, dtype=np.float32)

    for i in range(N):
        yhat_s = rolling_forecast_chunks(
            model,
            X_test_s[i],
            y_test_s[i],
            window=window,
            chunk=chunk,
            exog_fill=exog_fill,
            device=device
        )
        Y_pred_all[i] = yhat_s

    return Y_true_all, Y_pred_all
import numpy as np
import pandas as pd
import torch

from gluonts.dataset.common import ListDataset
from uni2ts.model.moirai import MoiraiForecast, MoiraiModule
from chronos import ChronosPipeline


# -------------------------
# Helpers: inverse-scaling
# -------------------------
def _inv_y(y_1d, y_scaler=None):
    y_1d = np.asarray(y_1d, dtype=np.float32)
    if y_scaler is None:
        return y_1d
    return y_scaler.inverse_transform(y_1d.reshape(-1, 1)).ravel().astype(np.float32)

def _inv_x(x_2d, x_scaler=None):
    x_2d = np.asarray(x_2d, dtype=np.float32)
    if x_scaler is None:
        return x_2d
    x2 = x_2d.reshape(-1, x_2d.shape[-1])
    x_inv = x_scaler.inverse_transform(x2).reshape(x_2d.shape)
    return x_inv.astype(np.float32)


# -------------------------
# Chronos batch forecast
# -------------------------
import numpy as np
import torch

def forecast_chronos_batch_from_start(
    pipeline,
    y_test_s,
    *,
    idxs,
    context_length=30,
    horizon=None,
    y_scaler=None,
    num_samples=100,
):
    y_test_s = np.asarray(y_test_s)
    N, T = y_test_s.shape
    idxs = np.asarray(idxs, dtype=int)

    H = (T - context_length) if horizon is None else int(horizon)
    H = max(1, H)

    contexts = []
    for i in idxs:
        y = y_test_s[i].astype(np.float32)
        if y_scaler is not None:
            y = y_scaler.inverse_transform(y.reshape(-1,1)).ravel().astype(np.float32)

        ctx = y[:context_length]  # <-- FIRST 30 days (not last)
        contexts.append(torch.tensor(ctx, dtype=torch.float32))

    context = torch.stack(contexts, dim=0)  # (B, L)
    pred = pipeline.predict(context, prediction_length=H, num_samples=num_samples)  # (B,S,H)

    median = torch.quantile(pred, 0.5, dim=1).cpu().numpy()
    low    = torch.quantile(pred, 0.1, dim=1).cpu().numpy()
    high   = torch.quantile(pred, 0.9, dim=1).cpu().numpy()
    return median, low, high

# -------------------------
# Moirai batch forecast (past exog)
# -------------------------
import pandas as pd
from gluonts.dataset.common import ListDataset
from uni2ts.model.moirai import MoiraiForecast

def forecast_moirai_batch_from_start(
    module,
    X_test_s,
    y_test_s,
    *,
    idxs,
    context_length=30,
    horizon=None,
    freq="D",
    y_scaler=None,
    x_scaler=None,
    num_samples=100,
    start_timestamp="2021-01-01",
):
    X_test_s = np.asarray(X_test_s)
    y_test_s = np.asarray(y_test_s)
    N, T, F = X_test_s.shape
    idxs = np.asarray(idxs, dtype=int)

    H = (T - context_length) if horizon is None else int(horizon)
    H = max(1, H)

    data_list = []
    for i in idxs:
        y = y_test_s[i].astype(np.float32)
        x = X_test_s[i].astype(np.float32)

        if y_scaler is not None:
            y = y_scaler.inverse_transform(y.reshape(-1,1)).ravel().astype(np.float32)
        if x_scaler is not None:
            x2 = x.reshape(-1, x.shape[-1])
            x = x_scaler.inverse_transform(x2).reshape(x.shape).astype(np.float32)

        # <-- ONLY provide context part (first 30 days) to avoid using future info
        y_ctx = y[:context_length]          # (L,)
        x_ctx = x[:context_length, :]       # (L,F)

        data_list.append({
            "start": pd.Timestamp(start_timestamp),
            "target": y_ctx,
            "past_feat_dynamic_real": x_ctx.T,  # (F,L)
        })

    ds = ListDataset(data_list, freq=freq)

    model = MoiraiForecast(
        module=module,
        prediction_length=H,
        context_length=context_length,
        patch_size="auto",
        num_samples=num_samples,
        target_dim=1,
        feat_dynamic_real_dim=0,
        past_feat_dynamic_real_dim=F,
    )

    predictor = model.create_predictor(batch_size=min(32, len(idxs)))
    forecasts = list(predictor.predict(ds))

    med = np.zeros((len(idxs), H), dtype=np.float32)
    lo  = np.zeros((len(idxs), H), dtype=np.float32)
    hi  = np.zeros((len(idxs), H), dtype=np.float32)

    for j, fc in enumerate(forecasts):
        samples = fc.samples
        if samples.ndim == 3:
            samples = samples[:, :, 0]
        med[j] = np.quantile(samples, 0.5, axis=0)
        lo[j]  = np.quantile(samples, 0.1, axis=0)
        hi[j]  = np.quantile(samples, 0.9, axis=0)

    return med, lo, hi


# -------------------------
# Wrapper: run both models with your fixed window=30
# -------------------------
def forecast_all_foundation_models_from_start(
    loaded_models: dict,
    X_test_s,
    y_test_s,
    *,
    idxs=0,
    context_length=30,
    min_horizon=60,
    freq="D",
    y_scaler=None,
    x_scaler=None,
    num_samples=100,
    start_timestamp="2021-01-01",
):
    if isinstance(idxs, (int, np.integer)):
        idxs = [int(idxs)]
    idxs = list(map(int, idxs))

    T = y_test_s.shape[1]
    horizon = max(int(min_horizon), int(T - context_length))

    out = {"meta": {
        "context_length": context_length,
        "horizon": horizon,
        "idxs": idxs,
        "freq": freq,
        "num_samples": num_samples,
    }}

    for name, model in loaded_models.items():
        if model is None:
            continue

        if "chronos" in name:
            med, lo, hi = forecast_chronos_batch_from_start(
                model, y_test_s,
                idxs=idxs,
                context_length=context_length,
                horizon=horizon,
                y_scaler=y_scaler,
                num_samples=num_samples,
            )
            out[name] = {"type": "chronos", "median": med, "low": lo, "high": hi}

        elif "moirai" in name:
            med, lo, hi = forecast_moirai_batch_from_start(
                model, X_test_s, y_test_s,
                idxs=idxs,
                context_length=context_length,
                horizon=horizon,
                freq=freq,
                y_scaler=y_scaler,
                x_scaler=x_scaler,
                num_samples=num_samples,
                start_timestamp=start_timestamp,
            )
            out[name] = {"type": "moirai", "median": med, "low": lo, "high": hi}

    return out


import numpy as np
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt

# def plot_idx_context_and_forecast_all(
#     *,
#     idx,
#     model_roll,
#     X_test_s,
#     y_test_s,
#     y_scaler,
#     foundation_results,
#     window=30,
#     chunk=10,
#     exog_fill="persistence",
#     title=None,
# ):
#     meta = foundation_results["meta"]
#     H = meta["horizon"]
#     idxs_used = meta["idxs"]
#     if idx not in idxs_used:
#         raise ValueError(f"idx={idx} not in foundation_results['meta']['idxs']={idxs_used}. Re-run wrapper with idx included.")

#     bpos = idxs_used.index(idx)

#     # True (original units)
#     y_true_full = y_scaler.inverse_transform(
#         np.asarray(y_test_s[idx], dtype=np.float32).reshape(-1, 1)
#     ).ravel()
#     T = len(y_true_full)

#     # Rolling forecast (original units)
#     yhat_s_roll = rolling_forecast_chunks(
#         model_roll,
#         np.asarray(X_test_s[idx], dtype=np.float32),
#         np.asarray(y_test_s[idx], dtype=np.float32),
#         window=window,
#         chunk=chunk,
#         exog_fill=exog_fill,
#     )
#     y_pred_roll = y_scaler.inverse_transform(
#         np.asarray(yhat_s_roll, dtype=np.float32).reshape(-1, 1)
#     ).ravel()

#     # Plot range: 0..window+H-1 (capped)
#     end = min(T, window + H)
#     x = np.arange(end)

#     plt.figure(figsize=(13, 6))

#     # Context region (0..window-1): show true as light solid (optional)
#     plt.plot(x[:window], y_true_full[:window],color='steelblue', linewidth=2.5, label="True (context)")

#     # True AFTER window only (solid)
#     plt.plot(x[window:], y_true_full[window:end],color='steelblue', linewidth=3.0)

#     # Separator
#     plt.axvline(window - 0.5, linewidth=1.5)

#     # Rolling (dashed; only after window)
#     plt.plot(x[window:], y_pred_roll[window:end], linestyle="--", linewidth=2.2,color='orange',
#              label=f"Rolling Forecasting (window={window}, chunk={chunk})")

#     # Foundation models: dashed variants
#     styles = ["-.", ":", (0, (5, 2)), (0, (3, 1, 1, 1)), (0, (7, 2, 2, 2))]
#     style_i = 0

#     for name, res in foundation_results.items():
#         if name == "meta":
#             continue
#         if res is None or "median" not in res:
#             continue

#         med = np.asarray(res["median"])[bpos]  # (H,)

#         # Place forecast starting at window
#         y_f = np.full((end,), np.nan, dtype=np.float32)
#         h_use = min(H, end - window)
#         y_f[window:window + h_use] = med[:h_use]

#         ls = styles[style_i % len(styles)]
#         style_i += 1
#         plt.plot(x, y_f, linestyle=ls, linewidth=2.0, label=f"{name} (median)")

#     plt.xlabel("Day index",fontsize=14)
#     plt.ylabel("Target (Kg/Ha)",fontsize=14)
#     # plt.title(title if title else f"Idx={idx}: context={window}, horizon={H}")
#     plt.legend(fontsize=10)
#     plt.xticks(fontsize=12)
#     plt.yticks(fontsize=12)
#     plt.tight_layout()
#     plt.show()
# import numpy as np
# from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

def evaluate_forecast_metrics(
    Y_true_all,
    Y_pred_all,
    *,
    window=0,
    metric="mse",      # "mse", "rmse", "mae", "r2"
    reduce="global"    # "global" or "per_sample"
):
    """
    Computes regression metrics ONLY for t >= window.

    Parameters
    ----------
    Y_true_all : array-like, shape (N,T) or (T,)
    Y_pred_all : array-like, shape (N,T) or (T,)
    window : int
        Ignore timesteps [0:window)
    metric : str
        "mse", "rmse", "mae", or "r2"
    reduce : str
        "global" -> average over samples
        "per_sample" -> return one value per sample

    Returns
    -------
    float or np.ndarray
    """

    Y_true_all = np.asarray(Y_true_all, dtype=float)
    Y_pred_all = np.asarray(Y_pred_all, dtype=float)

    if Y_true_all.shape != Y_pred_all.shape:
        raise ValueError(f"Shape mismatch: {Y_true_all.shape} vs {Y_pred_all.shape}")

    # Ensure 2D
    if Y_true_all.ndim == 1:
        Y_true_all = Y_true_all[None, :]
        Y_pred_all = Y_pred_all[None, :]

    # Remove cheating region
    if window > 0:
        Y_true_all = Y_true_all[:, window:]
        Y_pred_all = Y_pred_all[:, window:]

    N, T = Y_true_all.shape
    vals = np.zeros(N, dtype=float)

    for i in range(N):
        yt = Y_true_all[i]
        yp = Y_pred_all[i]

        if metric == "mse":
            vals[i] = mean_squared_error(yt, yp)
        elif metric == "rmse":
            vals[i] = np.sqrt(mean_squared_error(yt, yp))
        elif metric == "mae":
            vals[i] = mean_absolute_error(yt, yp)
        elif metric == "r2":
            vals[i] = r2_score(yt, yp)
        else:
            raise ValueError("metric must be one of: 'mse', 'rmse', 'mae', 'r2'")

    if reduce == "per_sample":
        return vals

    return float(np.mean(vals))
import numpy as np
import pandas as pd
from gluonts.dataset.common import ListDataset
from uni2ts.model.moirai import MoiraiForecast

import numpy as np
import pandas as pd
from gluonts.dataset.common import ListDataset
from uni2ts.model.moirai import MoiraiForecast

def forecast_moirai_oracle_one(
    module,
    X_test_s,
    y_test_s,
    *,
    idx=0,
    context_length=30,
    min_horizon=60,
    freq="D",
    y_scaler=None,
    x_scaler=None,
    num_samples=200,
    start_timestamp="2021-01-01",
):
    """
    Returns (med, lo, hi) each shape (H,)
    Uses future exogenous covariates via feat_dynamic_real (oracle).
    Context is FIRST context_length timesteps; forecast starts at context_length.
    """
    X_test_s = np.asarray(X_test_s)
    y_test_s = np.asarray(y_test_s)
    T = y_test_s.shape[1]
    F = X_test_s.shape[2]

    H = max(int(min_horizon), int(T - context_length))

    def inv_y(y):
        y = np.asarray(y, dtype=np.float32)
        if y_scaler is None:
            return y
        return y_scaler.inverse_transform(y.reshape(-1, 1)).ravel().astype(np.float32)

    def inv_x(x):
        x = np.asarray(x, dtype=np.float32)
        if x_scaler is None:
            return x
        x2 = x.reshape(-1, x.shape[-1])
        return x_scaler.inverse_transform(x2).reshape(x.shape).astype(np.float32)

    y_full = inv_y(y_test_s[idx])         # (T,)
    x_full = inv_x(X_test_s[idx])         # (T,F)

    # context (first L)
    y_ctx = y_full[:context_length]                 # (L,)
    x_past = x_full[:context_length, :].T           # (F,L)

    # future exog for horizon (oracle from dataset)
    x_fut = x_full[context_length:context_length+H, :].T  # (F,H)
    if x_fut.shape[1] < H:
        pad = H - x_fut.shape[1]
        x_fut = np.pad(x_fut, ((0, 0), (0, pad)), mode="edge")

    ds = ListDataset(
        [{
            "start": pd.Timestamp(start_timestamp),
            "target": y_ctx,
            "past_feat_dynamic_real": x_past,
            "feat_dynamic_real": x_fut,
        }],
        freq=freq
    )

    model = MoiraiForecast(
        module=module,
        prediction_length=H,
        context_length=context_length,
        patch_size="auto",
        num_samples=num_samples,
        target_dim=1,
        feat_dynamic_real_dim=F,
        past_feat_dynamic_real_dim=F,
    )

    predictor = model.create_predictor(batch_size=1)
    fc = next(iter(predictor.predict(ds)))

    samples = fc.samples
    if samples.ndim == 3:
        samples = samples[:, :, 0]

    med = np.quantile(samples, 0.5, axis=0)
    lo  = np.quantile(samples, 0.1, axis=0)
    hi  = np.quantile(samples, 0.9, axis=0)
    return med.astype(np.float32), lo.astype(np.float32), hi.astype(np.float32)
# import matplotlib.pyplot as plt

def plot_idx_context_and_forecast(
    *,
    idx,
    model_roll,
    X_test_s,
    y_test_s,
    y_scaler,
    foundation_results,
    window=30,
    chunk=10,
    exog_fill="persistence",
    title=None,

    # ---- NEW optional oracle Moirai ----
    use_future_exog_moirai=False,         # default OFF
    moirai_module_for_oracle=None,        # pass loaded_models["moirai_small"] or ["moirai_base"]
    x_scaler=None,
    freq="D",
    num_samples_oracle=200,
    start_timestamp="2021-01-01",
):
    meta = foundation_results["meta"]
    H = meta["horizon"]
    idxs_used = meta["idxs"]
    if idx not in idxs_used:
        raise ValueError(f"idx={idx} not in foundation_results['meta']['idxs']={idxs_used}. Re-run wrapper with idx included.")
    bpos = idxs_used.index(idx)

    # True in original units
    y_true_full = y_scaler.inverse_transform(
        np.asarray(y_test_s[idx], dtype=np.float32).reshape(-1, 1)
    ).ravel()
    T = len(y_true_full)

    # Rolling prediction (scaled -> original)
    yhat_s_roll = rolling_forecast_chunks(
        model_roll,
        np.asarray(X_test_s[idx], dtype=np.float32),
        np.asarray(y_test_s[idx], dtype=np.float32),
        window=window,
        chunk=chunk,
        exog_fill=exog_fill,
    )
    y_pred_roll = y_scaler.inverse_transform(
        np.asarray(yhat_s_roll, dtype=np.float32).reshape(-1, 1)
    ).ravel()

    # Plot range: context + horizon
    end = min(T, window + H)
    x = np.arange(end)

    plt.figure(figsize=(13, 6))

    # Context (0..window-1): show true
    plt.plot(x[:window], y_true_full[:window], linewidth=2.5, label="True (context)")

    # True AFTER window only (solid)
    plt.plot(x[window:], y_true_full[window:end], linewidth=3.0, label="True (after window)")

    # Separator
    plt.axvline(window - 0.5, linewidth=1.5)

    # Rolling (dashed; after window)
    plt.plot(x[window:], y_pred_roll[window:end], linestyle="--", linewidth=2.2,
             label=f"Rolling (window={window},chunk={chunk})")

    # Foundation models (past-only) curves
    styles = ["-.", ":", (0, (5, 2)), (0, (3, 1, 1, 1)), (0, (7, 2, 2, 2))]
    style_i = 0

    for name, res in foundation_results.items():
        if name == "meta":
            continue
        if res is None or "median" not in res:
            continue

        med = np.asarray(res["median"])[bpos]  # (H,)

        y_f = np.full((end,), np.nan, dtype=np.float32)
        h_use = min(H, end - window)
        y_f[window:window + h_use] = med[:h_use]

        ls = styles[style_i % len(styles)]
        style_i += 1
        plt.plot(x, y_f, linestyle=ls, linewidth=2.0, label=f"{name} (median)")

    # ---- OPTIONAL: Oracle Moirai with future exog ----
    if use_future_exog_moirai:
        if moirai_module_for_oracle is None:
            raise ValueError("Set moirai_module_for_oracle=loaded_models['moirai_small' or 'moirai_base' or 'moiral_large] when use_future_exog_moirai=True")

        med_o, lo_o, hi_o = forecast_moirai_oracle_one(
            moirai_module_for_oracle,
            X_test_s,
            y_test_s,
            idx=idx,
            context_length=window,
            min_horizon=max(60, T - window),
            freq=freq,
            y_scaler=y_scaler,
            x_scaler=x_scaler,
            num_samples=num_samples_oracle,
            start_timestamp=start_timestamp,
        )

        y_o = np.full((end,), np.nan, dtype=np.float32)
        h_use = min(len(med_o), end - window)
        y_o[window:window + h_use] = med_o[:h_use]

        # clearly label it
        plt.plot(x, y_o, linestyle=(0, (2, 2)), linewidth=2.3,
                 label="moirai_large_future_exog (median)")

    plt.xlabel("Day index",fontsize=15)
    plt.ylabel("Target (Kg/Ha)",fontsize=15)
    # plt.title(title if title else f"Idx={idx}: context={window}, horizon={H}")
    plt.legend(fontsize=13,loc='upper right')
    plt.tight_layout()
    plt.show()

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score

def plot_two_windows_varying_chunks_with_true_context(
    *,
    idx,
    model_roll_w30,
    model_roll_w60,
    X_test_s,
    y_test_s,
    y_scaler,
    chunks=(10, 20, 30),
    exog_fill="persistence",
    title=None,
):
    """
    Plot:
      - True series from day 0 to end (solid)
      - Model forecasts only AFTER their window:
          w=30 curves start at day 30
          w=60 curves start at day 60
      - Vertical separators at 30 and 60
    """

    # True in original units
    y_true = y_scaler.inverse_transform(
        np.asarray(y_test_s[idx], dtype=np.float32).reshape(-1, 1)
    ).ravel()
    T = len(y_true)
    x = np.arange(T)

    plt.figure(figsize=(14, 6))

    # True from start
    plt.plot(x, y_true, linewidth=3.0, label="True")

    # separators for windows
    plt.axvline(30 - 0.5, linewidth=1.2)
    plt.axvline(60 - 0.5, linewidth=1.2)

    # line styles for different chunks
    chunk_styles = {
        10: "--",
        20: "-.",
        30: ":",
    }

    # ---- window 30 model curves (only plot from day 30 onward) ----
    for ch in chunks:
        yhat_s = rolling_forecast_chunks(
            model_roll_w30,
            np.asarray(X_test_s[idx], dtype=np.float32),
            np.asarray(y_test_s[idx], dtype=np.float32),
            window=30,
            chunk=ch,
            exog_fill=exog_fill,
        )
        y_pred = y_scaler.inverse_transform(
            np.asarray(yhat_s, dtype=np.float32).reshape(-1, 1)
        ).ravel()

        plt.plot(
            x[30:], y_pred[30:],   # <-- start plotting predictions at day 30
            linestyle=chunk_styles.get(ch, "--"),
            linewidth=2.0,
            label=f"Roll window=30, chunk={ch}"
        )

    # ---- window 60 model curves (only plot from day 60 onward) ----
    for ch in chunks:
        yhat_s = rolling_forecast_chunks(
            model_roll_w60,
            np.asarray(X_test_s[idx], dtype=np.float32),
            np.asarray(y_test_s[idx], dtype=np.float32),
            window=60,
            chunk=ch,
            exog_fill=exog_fill,
        )
        y_pred = y_scaler.inverse_transform(
            np.asarray(yhat_s, dtype=np.float32).reshape(-1, 1)
        ).ravel()

        plt.plot(
            x[60:], y_pred[60:],   # <-- start plotting predictions at day 60
            linestyle=chunk_styles.get(ch, "--"),
            linewidth=2.4,
            label=f"Roll window=60, chunk={ch}"
        )

    plt.xlabel("Day index",fontsize=15)
    plt.ylabel("Target variable (Kg/Ha)",fontsize=15)
    if title is not None:
        plt.title(title)
    # or f"Idx={idx}: forecasts start at window"
    plt.legend(fontsize=15,loc='upper right')
    plt.tight_layout()
    plt.show()

    from sklearn.metrics import r2_score

def evaluate_r2_windows_chunks(
    *,
    models_by_window,     # {30: model_roll, 60: model_roll_nleach_w60}
    X_test_s,
    y_test_s,
    y_scaler,
    windows=(30, 60),
    chunks=(10, 20, 30),
    exog_fill="persistence",
    max_samples=None,     # set e.g. 200 for quicker eval; None = full test set
):
    """
    Returns a pandas DataFrame with rows (window, chunk) and the R² on test set.
    R² is computed on ORIGINAL units and only for t >= window.
    """

    X_test_s = np.asarray(X_test_s, dtype=np.float32)
    y_test_s = np.asarray(y_test_s, dtype=np.float32)

    N, T, F = X_test_s.shape
    idxs = np.arange(N) if max_samples is None else np.arange(min(N, max_samples))

    rows = []

    for w in windows:
        model = models_by_window[w]
        for ch in chunks:
            y_true_all = []
            y_pred_all = []

            for i in idxs:
                # rolling prediction (scaled)
                yhat_s = rolling_forecast_chunks(
                    model,
                    X_test_s[i],
                    y_test_s[i],
                    window=w,
                    chunk=ch,
                    exog_fill=exog_fill,
                )

                # inverse-transform to original units
                yt = y_scaler.inverse_transform(y_test_s[i].reshape(-1, 1)).ravel()
                yp = y_scaler.inverse_transform(yhat_s.reshape(-1, 1)).ravel()

                # evaluate only after window
                y_true_all.append(yt[w:])
                y_pred_all.append(yp[w:])

            y_true_flat = np.concatenate(y_true_all)
            y_pred_flat = np.concatenate(y_pred_all)

            r2 = r2_score(y_true_flat, y_pred_flat)

            rows.append({
                "window": w,
                "chunk": ch,
                "test_r2": float(r2),
                "n_series": len(idxs),
                "n_points": int(y_true_flat.size),
            })

    return pd.DataFrame(rows).sort_values(["window", "chunk"]).reset_index(drop=True)
