import pickle
import random
import time
from pathlib import Path

import polars as pl
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.optim as optim
from torchmetrics.regression import R2Score


def get_save_path():
    OUTPUT_DIR = Path().resolve() / "outputs"
    try:
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    except Exception as e:
        print(
            "❌ Permission Error: Could not access or execute files in current folder."
        )
        raise
    return OUTPUT_DIR


def train_epoch(
    model, loader, optimizer, criterion, device, min_val, max_val, ampscaler=None
):
    model.train()
    total_loss = 0.0
    r2_metric = R2Score().to(device)
    for inputs, targets in loader:
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        if ampscaler:
            with torch.amp.autocast(device_type=device.type):
                outputs = model(inputs)
                loss = criterion(outputs, targets)
            ampscaler.scale(loss).backward()
            # ampscaler.unscale_(optimizer) # for debugging
            # torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0) # for debugging
            ampscaler.step(optimizer)
            ampscaler.update()
        else:
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
        outputs_dn = (outputs * (max_val - min_val)) + min_val
        targets_dn = (targets * (max_val - min_val)) + min_val
        r2_metric.update(outputs_dn.float(), targets_dn.float())
        total_loss += loss.float().item()
    r2 = r2_metric.compute().cpu().item()
    r2_metric.reset()
    return total_loss / len(loader), r2


##########################################################################
##########################################################################


def val_epoch(model, loader, criterion, device, min_val, max_val):
    model.eval()
    total_loss = 0.0
    r2_metric = R2Score().to(device)
    with torch.no_grad():
        for inputs, targets in loader:
            inputs, targets = inputs.to(device), targets.to(device)
            with torch.amp.autocast(device_type=device.type):
                outputs = model(inputs)
                loss = criterion(outputs, targets)
            outputs_dn = (outputs * (max_val - min_val)) + min_val
            targets_dn = (targets * (max_val - min_val)) + min_val
            r2_metric.update(outputs_dn.float(), targets_dn.float())
            total_loss += loss.float().item()
    r2 = r2_metric.compute().cpu().item()
    r2_metric.reset()
    return total_loss / len(loader), r2


##########################################################################
##########################################################################


def train_model(
    model_name,
    model,
    train_loader,
    val_loader,
    tgt,
    lr,
    max_epochs,
    min_tgt,
    max_tgt,
    criterion=None,
    optimizer=None,
    scheduler=None,
    early_stop_patience: int = 10,
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    min_loss_reduction=1e-4,
):
    model.to(device)
    if criterion is None:
        criterion = nn.MSELoss()
    if optimizer is None:
        optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    if scheduler is None:
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.5, patience=5, min_lr=1e-8
        )
    ampscaler = torch.amp.GradScaler(device=device.type)
    model_dir = get_save_path() / tgt
    model_dir.mkdir(parents=True, exist_ok=True)
    model_save_path = model_dir / f"{tgt}_{model_name}.pth"
    logs_save_path = model_dir / f"logs_{tgt}_{model_name}.csv"

    # Training Loop
    best_val_loss = float("inf")
    train_val_logs = []
    early_stop = False
    epochs_no_improve = 0
    print("Training started....")
    for epoch in range(max_epochs):
        # Break loop if early stop triggered
        if early_stop:
            print(f"Early stopping at epoch {epoch}")
            break

        start_time = time.time()
        train_loss, train_r2 = train_epoch(
            model,
            train_loader,
            optimizer,
            criterion,
            device,
            min_tgt,
            max_tgt,
            ampscaler=ampscaler,
        )
        val_loss, val_r2 = val_epoch(
            model, val_loader, criterion, device, min_tgt, max_tgt
        )
        scheduler.step(val_loss)

        # Early stop and save logic
        if val_loss < best_val_loss - min_loss_reduction:
            best_val_loss = val_loss
            torch.save(model.state_dict(), model_save_path)
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve > early_stop_patience:
                early_stop = True

        elapsed = time.time() - start_time
        curr_lr = optimizer.param_groups[0]["lr"]
        train_val_logs.append(
            {
                "epoch": epoch+1,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "train_r2": train_r2,
                "val_r2": val_r2,
                "learning_rate": curr_lr,
            }
        )
        print(
            f"Epoch: {epoch + 1}/{max_epochs} | "
            f"Train Loss: {train_loss:.5f}, Train R²: {train_r2:.4f} | "
            f"Val Loss: {val_loss:.5f}, Val R²: {val_r2:.4f} | "
            f"LR: {curr_lr}, Time: {round(elapsed, 2)} secs"
        )
    if not model_save_path.exists():
        torch.save(model.state_dict(), model_save_path)
    train_val_df = pd.DataFrame(train_val_logs)
    train_val_df.to_csv(logs_save_path, index=False)
    print("Training Complete....!")
