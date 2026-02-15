import torch
import numpy as np
import pandas as pd
from sklearn.metrics import r2_score, mean_absolute_error, root_mean_squared_error
from utils import scaler
from utils import preprocessing as ppsr

def evaluate_model(model, data_loader, tgt, device, scaler_path):
    model.to(device)
    model.eval()
    all_preds = []
    all_targets = [] 
    min_tgt, max_tgt = scaler.get_min_max(tgt, scaler_path)
    with torch.inference_mode():
        for i, (inputs, targets) in enumerate(data_loader):
            inputs = inputs.to(device)
            outputs = model(inputs)
            
            # Flatten, format and denormalize
            targets = targets.numpy().flatten()
            outputs = outputs.cpu().numpy().flatten()
            outputs_denorm = scaler.inverse_normalize(outputs, tgt, scaler_path=scaler_path)
            targets_denorm = scaler.inverse_normalize(targets, tgt, scaler_path=scaler_path)
            
            all_preds.append(outputs_denorm)
            all_targets.append(targets_denorm)
            pgrs = (i+1) * 20 // len(data_loader)
            print(f"{i+1}/{len(data_loader)}[{'=' * pgrs}>{' ' * (19 - pgrs)}]", end='\r')
    # Concatenate all predictions and targets
    y_pred = np.concatenate(all_preds)
    y_true = np.concatenate(all_targets)
    return compute_metrics(y_true, y_pred, min_tgt, max_tgt)


def compute_metrics(y_true, y_pred, min_tgt, max_tgt):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = root_mean_squared_error(y_true, y_pred) 
    nrmse = rmse / (max_tgt - min_tgt)
    r2 = r2_score(y_true, y_pred)
    return mae, rmse, nrmse, r2



def evaluate_by_scenario(model, data, feats, tgt, device, 
                         scaler_path, seq_len=None):
    model.to(device)
    model.eval()
    results = []
    scenario_vars = ['Year', 'PlantingDay', 'Treatment', 'IrrgDep', 'IrrgThresh', 'NFirstApp']
    scenario_groups = data.groupby(scenario_vars, observed=True)
    num_groups = len(scenario_groups)
    for i, (scenario_name, group) in enumerate(scenario_groups):
        X, y_true = ppsr.process_data(group, feats=feats, tgt=tgt, scaler_path=scaler_path, 
                                        mode="transform", seq_len=seq_len)
        X_tensor = torch.FloatTensor(X).to(device)
        min_tgt, max_tgt = scaler.get_min_max(tgt, scaler_path)
        with torch.inference_mode():
            y_pred = model(X_tensor)
        
        y_true = y_true.numpy().flatten()
        y_pred = y_pred.cpu().numpy().flatten()
        
        # Denormalize before calculating metrics
        y_pred_denorm = scaler.inverse_normalize(y_pred, tgt, scaler_path=scaler_path)
        y_true_denorm = scaler.inverse_normalize(y_true, tgt, scaler_path=scaler_path)     
        
        mae, rmse, r2 = compute_metrics(y_true_denorm, y_pred_denorm, min_tgt, max_tgt)
        results.append({
            "Scenario": "_".join([str(s) for s in scenario_name]),
            "mae": mae,
            "rmse": rmse,
            "r2": r2
        })
        pgrs = (i+1) * 20 // num_groups
        print(f"{i+1}/{num_groups}[{'=' * pgrs}>{' ' * (19 - pgrs)}]", end='\r')
    results_df = pd.DataFrame(results)
    avg_mae = results_df["mae"].mean()
    avg_rmse = results_df["rmse"].mean()
    avg_r2 = results_df["r2"].mean()
    return results_df, avg_mae, avg_rmse, avg_r2