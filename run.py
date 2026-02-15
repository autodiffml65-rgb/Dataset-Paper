import pickle
import random
import json
from pathlib import Path
import argparse
import importlib

import polars as pl
import numpy as np
import pandas as pd

import torch
from torch.utils.data import DataLoader, TensorDataset
from torchinfo import summary
from utils.preprocessing import process_data
from utils import scaler
from training import train
from testing import evaluate

##########################################################################
##########################################################################


# GLOBAL PATHS & VARIABLES
CONFIG_FILE = Path("utils/config.json")
OUTPUT_DIR = Path("outputs")
DATA_DIR = Path("data")
SAVE_DIR = Path("saves")

if not CONFIG_FILE.is_file():
    raise FileNotFoundError("❌ Error: Configuration file not found or is not a file at the specified path")
try:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    DATA_DIR.mkdir(parents=True, exist_ok=True)
except Exception as e:
    print("❌ Permission Error: Could not access or execute files in current folder.")
    raise


##########################################################################
##########################################################################


def load_data(file):
    try:
        return pd.read_parquet(file)
    except FileNotFoundError:
        raise FileNotFoundError(f"Please make sure '{file.absolute().name}' exists in data folder.")
    except pd.errors.ParserError:
        print(f"Error: Failed to parse the file at '{file.absolute().name}'")
        raise
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        raise


##########################################################################
##########################################################################


def verify_data(df) -> None:
    if len(set(features) - set(df.columns)) > 0:
        raise ValueError("All required columns not present is selected dataset.")
    

##########################################################################
##########################################################################


def get_model(mdata, mdl_id, seq_len=None):
    model_map = {
        "linear": ("Linear Regression", "linearregression"),
        "mlp": ("Multi-Layered Perceptron", "mlp"),
        "cnn": ("1-D Convolutional Neural Network", "cnn"),
        "tcn": ("Temporal Convolutional Network", "tcn"),
        "lstm": ("Long Short-Term Memory", "lstm"),
        "transformer": ("EncoderOnlyTransformer", "transformer")
    }
    if mdl_id not in model_map:
        raise NameError(f"Model class for '{mdl_id}' not found.")
    model_key, submodule_name = model_map[mdl_id]
    try:
        submodule = importlib.import_module(f"models.{submodule_name}")
    except ModuleNotFoundError:
        raise ImportError(f"Module 'models.{submodule_name}' not found")
    
    model_name = mdata['models'][model_key].get('class_name')
    model_params = mdata['models'][model_key].get('params')
    if mdl_id == "transformer":
        model_params['max_seq_len'] = seq_len
    model_obj = getattr(submodule, model_name, None)
    if model_obj is None:
        raise AttributeError(f"Class {model_name} not found in models.{submodule_name}")
    model = model_obj(**model_params)
    return model, model_name


##########################################################################
##########################################################################


def run_training(model_name, model, train_set, val_set, feats, tgt, max_epochs, 
                 lr, b_sz, seq_len, device=torch.device("cpu")):
    scaler_dir = OUTPUT_DIR / tgt
    scaler_dir.mkdir(parents=True, exist_ok=True)
    scaler_filepath = scaler_dir /f"{tgt}_scaler.pkl"
    X_train, y_train = process_data(
        train_set, feats, tgt, scaler_path=scaler_filepath, 
        mode="fit", seq_len=seq_len
        )
    X_val, y_val = process_data(
        val_set, feats, tgt, scaler_path=scaler_filepath,
        mode="transform", seq_len=seq_len
        )
    print("Train/Validation shapes (X_train, y_train, X_val, y_val): ")
    print(X_train.shape, y_train.shape, X_val.shape, y_val.shape, end="\n")

    #Creating dataloader
    train_loader = DataLoader(TensorDataset(X_train, y_train),num_workers=10,prefetch_factor=4,pin_memory=True, batch_size=b_sz, shuffle=True)
    val_loader = DataLoader(TensorDataset(X_val, y_val),num_workers=4,prefetch_factor=4,pin_memory=True, batch_size=b_sz, shuffle=False)
    
    print(f"Training on device: {device}")
    summary(model)
    min_tgt, max_tgt = scaler.get_min_max(tgt, scaler_filepath)
    min_tgt = torch.tensor(min_tgt, device=device)
    max_tgt = torch.tensor(max_tgt, device=device)
    train.train_model(
        model_name, model, train_loader, val_loader, tgt, 
        lr=lr, max_epochs=max_epochs, min_tgt=min_tgt,
        max_tgt=max_tgt, device=device
        )


##########################################################################
##########################################################################


def run_inference(model_name, model, model_dir, test_set, feats, tgt, 
                  b_sz, seq_len, device=torch.device("cpu")):
    # Currently assuming that files are in 'saves' folder
    model_state_file, scaler_filepath = get_model_state_scaler(model_name, model_dir, tgt)
    model.load_state_dict(torch.load(model_state_file, weights_only=True))
    X_test, y_test = process_data(
        test_set, feats, tgt, scaler_path=scaler_filepath, 
        mode="transform", seq_len=seq_len
        )
    print("Test shapes (X_test, y_test): ", end="")
    print(X_test.shape, y_test.shape)

    data_loader = DataLoader(TensorDataset(X_test, y_test), batch_size=b_sz, 
                             num_workers=4,prefetch_factor=4,pin_memory=True, shuffle=False)
    mae, rmse, nrmse, r2 = evaluate.evaluate_model(
        model, data_loader, tgt, device=device, scaler_path=scaler_filepath
        )
    print(f"\n\nMean Absolute Error: {mae:.4f}")
    print(f"Root Mean Square Error: {rmse:.4f}")
    print(f"Normalized Root Mean Square Error: {nrmse:.4f}")
    print(f"Coefficient of determination (R-Squared): {r2:.4f}\n")


##########################################################################
##########################################################################


def get_model_state_scaler(mdl_name, mdl_dir, tgt):
    model_state = model_dir / f"{tgt}_{mdl_name}.pth"
    data_scaler = model_dir / f"{tgt}_scaler.pkl"
    return model_state, data_scaler


##########################################################################
##########################################################################


def parse_args():
    git_link = "https://github.com/GatorSense/PotSim"
    parser = argparse.ArgumentParser(
        epilog=f"For more details or help visit: \x1b]8;;{git_link}\x1b\\{git_link}\x1b]8;;\x1b\\",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    valid_tgts = ["NTotL1","NTotL2","SWatL1","SWatL2","NLeach","NPlantUp"]
    model_choices = ['linear', 'mlp', 'cnn', 'tcn', 'lstm', 'transformer']
    subparsers = parser.add_subparsers(dest="task", required=True, help="Task to run.'")

    # train arguments
    train_parser = subparsers.add_parser('train', help='Train the model')
    train_parser.add_argument('-tgt', '--target', required=True, choices=valid_tgts, 
                              help=f"Select a target variable: {', '.join(valid_tgts)}")
    train_parser.add_argument('-m', '--model', choices=model_choices, required=True, 
                              type=str, help=f"Select model: {', '.join(model_choices)}")
    train_parser.add_argument('-tdata', '--train_dataset', type=str, default="train_split", 
                              help='Train dataset to train the model on.')
    train_parser.add_argument('-vdata', '--val_dataset', type=str, default="val_split", 
                              help='Val dataset to validate the model on.')
    train_parser.add_argument("-bs", "--batch_size", default=256, type=int,
                              help="Batch size for training (default: 256).")
    train_parser.add_argument("-lr", "--learning_rate", default=0.005, type=float,
                              help="Learning rate for training (default: 0.005).")
    train_parser.add_argument("-ep", "--epochs", default=100, type=int,
                              help="Maximum number of epochs for training (default: 100).")
    train_parser.add_argument("-sl", "--seq_len", default=15, type=int,
                              help="Sequence length for sequence-based models (default: 15).")
    train_parser.add_argument('-d', '--device', choices=['cpu', 'cuda'], default=None,
                              help='Device to use: "cpu" or "cuda" (default: None)')

    # test arguments
    test_parser = subparsers.add_parser('test', help='Test the model')
    test_parser.add_argument('-tgt', '--target', required=True, choices=valid_tgts, 
                             help=f"Select a target variable: {', '.join(valid_tgts)}")
    test_parser.add_argument('-m', '--model', choices=model_choices, required=True, 
                             type=str, help=f"Select model: {', '.join(model_choices)}")
    test_parser.add_argument('-data', '--dataset', required=True, type=str,
                             help='Dataset to run test on.')
    test_parser.add_argument('-mdir', '--model_dir', type=str, choices=['outputs', 'saves'],
                             default='saves', help='Dataset to run test on.')
    test_parser.add_argument("-bs", "--batch_size", default=256, type=int,
                             help="Batch size for training (default: 256).")
    test_parser.add_argument("-sl", "--seq_len", default=15, type=int,
                             help="Sequence length for sequence-based models (default: 15).")
    test_parser.add_argument('-d', '--device', choices=['cpu', 'cuda'], default=None,
                             help='Device to use: "cpu" or "cuda" (default: None)')
        
    return parser.parse_args()


##########################################################################
##########################################################################


def get_run_config(tgt):
    try:
        with open(CONFIG_FILE, 'r') as file:
            return json.load(file).get(tgt)
    except FileNotFoundError:
        raise FileNotFoundError("Make sure 'config.json' exists in utils folder.")


##########################################################################
##########################################################################


def get_device(d):
    if d is None:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(d)


##########################################################################
##########################################################################


if __name__ == "__main__":
    args = parse_args()
    target = args.target
    model_id = args.model
    metadata = get_run_config(target)
    features = metadata["features"]
    b_sz, seq_len = args.batch_size, args.seq_len
    device = get_device(args.device)
    if args.task == "train":
        lr, max_epochs = args.learning_rate, args.epochs
        train_set = load_data(DATA_DIR / f"{args.train_dataset}.parquet")
        val_set = load_data(DATA_DIR / f"{args.val_dataset}.parquet")
        _, _ = verify_data(train_set), verify_data(val_set)
        seq_len = None if model_id in ['linear', 'mlp'] else seq_len
        model, model_name = get_model(metadata, model_id, seq_len=seq_len)
        run_training(model_name, model, train_set, val_set, features, 
                     target, max_epochs, lr, b_sz, seq_len, device=device)
    if args.task == "test":
        model_dir = (OUTPUT_DIR / target) if args.model_dir == 'outputs' else (SAVE_DIR / target)
        test_set = load_data(DATA_DIR / f"{args.dataset}.parquet")
        verify_data(test_set)
        seq_len = None if model_id in ['linear', 'mlp'] else seq_len
        model, model_name = get_model(metadata, model_id, seq_len=seq_len)
        run_inference(model_name, model, model_dir, test_set, features, target, b_sz, 
                      seq_len, device=device)