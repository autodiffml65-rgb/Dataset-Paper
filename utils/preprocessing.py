import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view
import torch
from utils import scaler


def __prepare_sequences(df, feats, tgt, seq_len=None):
    scenario_vars = [
        "Year",
        "PlantingDay",
        "Treatment",
        "NFirstApp",
        "OrgIrrgDep",
        "OrgIrrgThresh",
    ]
    df_sorted = df.sort_values(scenario_vars + ["OrgDayAfterPlant"])
    data = df_sorted[feats].values
    target = df_sorted[tgt].values

    # Empty Vectorized window creation in case no data
    if data.size == 0 or len(data) < seq_len:
        return np.empty((0, seq_len, len(feats))), np.empty(0)

    # Create group boundaries using vectorized operations
    group_mask = (
        (df_sorted[scenario_vars] != df_sorted[scenario_vars].shift())
        .any(axis=1)
        .values
    )
    group_ids = np.cumsum(group_mask)

    # Create validity mask for full sequences within groups
    valid = np.zeros(len(data) - seq_len + 1, dtype=bool)
    for i in range(len(valid)):
        valid[i] = group_ids[i] == group_ids[i + seq_len - 1]

    # Create windows and transpose dimensions to [samples, seq_len, features]
    X = sliding_window_view(data, (seq_len,), axis=0)[valid].transpose(0, 2, 1)
    y = target[seq_len - 1 :][valid]
    return X, y


def __transform_data(df, feats, tgt, scaler_path, mode="transform"):
    scenario_vars = [
        "Year",
        "PlantingDay",
        "Treatment",
        "NFirstApp",
        "OrgIrrgDep",
        "OrgIrrgThresh",
    ]
    df_copy = df.copy()
    df_copy["OrgDayAfterPlant"] = df["DayAfterPlant"]
    df_copy["OrgIrrgDep"] = df["IrrgDep"]
    df_copy["OrgIrrgThresh"] = df["IrrgThresh"]
    missing_vars = [var for var in scenario_vars if var not in df_copy.columns]
    if missing_vars:
        raise ValueError(f"Missing scenario variables: {missing_vars}")
    df_copy = df_copy.sort_values(by=scenario_vars + ["OrgDayAfterPlant"])
    df_copy["NApp"] = df_copy.groupby(scenario_vars, observed=True)["NApp"].transform(
        "cumsum"
    )
    return df_copy


def process_data(df, feats, tgt, scaler_path, mode="transform", seq_len=None):
    if not all(col in df.columns for col in feats + [tgt]):
        raise ValueError("Some columns not found in dataframe")
    numcols = df.select_dtypes(exclude=["category"]).columns
    df[numcols] = df[numcols].fillna(0)
    df_copy = __transform_data(df, feats, tgt, scaler_path, mode=mode)
    df_copy, _ = scaler.normalize_columns(
        df_copy, feats + [tgt], mode=mode, scaler_path=scaler_path
    )

    # For non-sequential data preparation
    if seq_len is None:
        X = df_copy[feats].values
        y = df_copy[tgt].values
        del df_copy
        return torch.FloatTensor(X), torch.FloatTensor(y).view(-1, 1)

    # For sequence preparation
    X, y = __prepare_sequences(df_copy, feats, tgt, seq_len=seq_len)
    del df_copy
    return torch.FloatTensor(X), torch.FloatTensor(y).view(-1, 1)
