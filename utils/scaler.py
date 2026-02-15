import pickle
import numpy as np
import pandas as pd
import torch

class StandardScaler:
    def __init__(self, path=None):
        self.params = {}
        self.path = path

    def fit(self, df, columns):
        df_copy = df.copy()
        for col in columns:
            self.params[col] = {
                'mean': df_copy[col].mean(),
                'std': df_copy[col].std()
            }
        with open(self.path, 'wb') as f:
            pickle.dump(self.params, f)

    def transform(self, df, columns):
        df_copy = df.copy()
        with open(self.path, 'rb') as f:
            self.params = pickle.load(f)
        for col in columns:
            if col in self.params:
                mean = self.params[col]['mean']
                std = self.params[col]['std']
                df_copy[col] = (df_copy[col] - mean) / std
        return df_copy

    def inverse_transform(self, df, columns):
        df_copy = df.copy()
        with open(self.path, 'rb') as f:
            self.params = pickle.load(f)
        for col in columns:
            if col in self.params:
                mean = self.params[col]['mean']
                std = self.params[col]['std']
                df_copy[col] = (df_copy[col] * std) + mean
        return df_copy


class MinMaxScaler:
    def __init__(self, path=None):
        self.params = {}
        self.path = path

    def fit(self, df, columns):
        df_copy = df.copy()
        for col in columns:
            self.params[col] = {
                'min': df_copy[col].min(),
                'max': df_copy[col].max()
            }
        with open(self.path, 'wb') as f:
            pickle.dump(self.params, f)

    def transform(self, df, columns):
        df_copy = df.copy()
        with open(self.path, 'rb') as f:
            self.params = pickle.load(f)
        for col in columns:
            if col in self.params:
                min_val = self.params[col]['min']
                max_val = self.params[col]['max']
                df_copy[col] = (df_copy[col] - min_val) / (max_val - min_val)
        return df_copy

    def inverse_transform(self, df, columns):
        df_copy = df.copy()
        with open(self.path, 'rb') as f:
            self.params = pickle.load(f)
        for col in columns:
            if col in self.params:
                min_val = self.params[col]['min']
                max_val = self.params[col]['max']
                df_copy[col] = (df_copy[col] * (max_val - min_val)) + min_val
        return df_copy
    
def normalize_columns(df, columns, mode='fit', scaler_path=None):
    scaler = MinMaxScaler(scaler_path)
    # scaler = StandardScaler(scaler_path)
    if mode == 'fit':
        scaler.fit(df, columns)
    transformed_df = scaler.transform(df, columns)
    return transformed_df, scaler

def denormalize_columns(df, columns, scaler_path=None):
    # scaler = StandardScaler(scaler_path)
    scaler = MinMaxScaler(scaler_path)
    denormalized_df = scaler.inverse_transform(df, columns)
    return denormalized_df


def get_min_max(tgt, scaler_path):
    with open(scaler_path, 'rb') as f:
            params = pickle.load(f)
    if tgt not in params:
            raise ValueError(f'{tgt} not present in scaler file.')
    min_val = params[tgt]['min']
    max_val = params[tgt]['max']
    return min_val, max_val


def inverse_normalize(arr, tgt, scaler_path):
    min_val, max_val = get_min_max(tgt, scaler_path)
    return (np.array(arr) * (max_val - min_val)) + min_val