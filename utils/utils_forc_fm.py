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
from utils_forc import *

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




# Forecaating code


# Moirai Forecast
import numpy as np
import pandas as pd
import inspect
from gluonts.dataset.common import ListDataset


#helper function for moirai forecast
import numpy as np

def _forecast_to_med_lo_hi(fc, q_levels=(0.1, 0.5, 0.9)):
    """
    Works for:
      - SampleForecast: has fc.samples
      - QuantileForecast: has fc.quantile(q)
    Returns: med, lo, hi as 1D arrays of length H
    """
    # Case 1: SampleForecast-like
    if hasattr(fc, "samples") and fc.samples is not None:
        samples = fc.samples
        samples = np.asarray(samples)

        # shapes can be (S, H) or (S, H, target_dim)
        if samples.ndim == 3:
            samples = samples[:, :, 0]

        lo  = np.quantile(samples, q_levels[0], axis=0)
        med = np.quantile(samples, q_levels[1], axis=0)
        hi  = np.quantile(samples, q_levels[2], axis=0)
        return med, lo, hi

    # Case 2: QuantileForecast-like
    if hasattr(fc, "quantile"):
        lo  = np.asarray(fc.quantile(str(q_levels[0])))
        med = np.asarray(fc.quantile(str(q_levels[1])))
        hi  = np.asarray(fc.quantile(str(q_levels[2])))
        return med, lo, hi

    raise TypeError(f"Unknown forecast object type: {type(fc)} (no samples, no quantile())")



def forecast_moirai_any_batch_from_start_new(
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
            y = y_scaler.inverse_transform(y.reshape(-1, 1)).ravel().astype(np.float32)
        if x_scaler is not None:
            x2 = x.reshape(-1, x.shape[-1])
            x = x_scaler.inverse_transform(x2).reshape(x.shape).astype(np.float32)

        y_ctx = y[:context_length]
        x_ctx = x[:context_length, :]

        data_list.append({
            "start": pd.Timestamp(start_timestamp),
            "target": y_ctx,
            "past_feat_dynamic_real": x_ctx.T,  # (F, L)
        })

    ds = ListDataset(data_list, freq=freq)

    is_v2 = "moirai2" in type(module).__name__.lower()

    if is_v2:
        from uni2ts.model.moirai2 import Moirai2Forecast

        # Moirai-2: no patch_size, no num_samples (in your version)
        model = Moirai2Forecast(
            module=module,
            prediction_length=H,
            context_length=context_length,
            target_dim=1,
            feat_dynamic_real_dim=0,
            past_feat_dynamic_real_dim=F,
        )

        # pass num_samples only if create_predictor supports it
        sig = inspect.signature(model.create_predictor)
        kwargs = {"batch_size": min(32, len(idxs))}
        if "num_samples" in sig.parameters:
            kwargs["num_samples"] = num_samples
        predictor = model.create_predictor(**kwargs)

    else:
        from uni2ts.model.moirai import MoiraiForecast

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
        med_j, lo_j, hi_j = _forecast_to_med_lo_hi(fc, q_levels=(0.1, 0.5, 0.9))
        med[j] = med_j
        lo[j]  = lo_j
        hi[j]  = hi_j

    return med, lo, hi


# Moirai for future exog inputs


def forecast_moirai_oracle_one_any(
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

    y_full = inv_y(y_test_s[idx])   # (T,)
    x_full = inv_x(X_test_s[idx])   # (T,F)

    y_ctx  = y_full[:context_length]             # (L,)
    x_past = x_full[:context_length, :].T        # (F,L)

    x_fut = x_full[context_length:context_length + H, :].T  # (F,H)
    if x_fut.shape[1] < H:
        pad = H - x_fut.shape[1]
        x_fut = np.pad(x_fut, ((0, 0), (0, pad)), mode="edge")

    is_v2 = "moirai2" in type(module).__name__.lower()

    if is_v2:
        # ✅ Moirai-2 expects feat_dynamic_real over the full timeline (past+future)
        x_dyn = np.concatenate([x_past, x_fut], axis=1)  # (F, L+H)

        ds = ListDataset(
            [{
                "start": pd.Timestamp(start_timestamp),
                "target": y_ctx,
                "past_feat_dynamic_real": x_past,   # (F,L)
                "feat_dynamic_real": x_dyn,         # (F,L+H)  <-- key fix
            }],
            freq=freq,
        )

        from uni2ts.model.moirai2 import Moirai2Forecast
        model = Moirai2Forecast(
            module=module,
            prediction_length=H,
            context_length=context_length,
            target_dim=1,
            feat_dynamic_real_dim=F,
            past_feat_dynamic_real_dim=F,
        )

        sig = inspect.signature(model.create_predictor)
        kwargs = {"batch_size": 1}
        if "num_samples" in sig.parameters:
            kwargs["num_samples"] = num_samples
        predictor = model.create_predictor(**kwargs)

    else:
        # Moirai-1: your original oracle format is usually fine
        ds = ListDataset(
            [{
                "start": pd.Timestamp(start_timestamp),
                "target": y_ctx,
                "past_feat_dynamic_real": x_past,   # (F,L)
                "feat_dynamic_real": x_fut,         # (F,H)
            }],
            freq=freq,
        )

        from uni2ts.model.moirai import MoiraiForecast
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
    med, lo, hi = _forecast_to_med_lo_hi(fc, q_levels=(0.1, 0.5, 0.9))
    return med.astype(np.float32), lo.astype(np.float32), hi.astype(np.float32)

import numpy as np
import torch

import numpy as np
import torch

#Chronos helper function

def _extract_quantiles_from_chronos2_output(pred_np, pipeline, q_levels=(0.1, 0.5, 0.9)):
    """
    pred_np can be:
      (B, V, Q, H)  <- what you're seeing
      (B, Q, H)
      (B, H)        (median only)
    Returns: med, lo, hi each (B, H)
    """
    qs = list(getattr(pipeline, "quantiles", []))

    def q_idx(q):
        # find exact match if present
        if q in qs:
            return qs.index(q)
        # try close match
        if len(qs) > 0:
            arr = np.asarray(qs, dtype=float)
            return int(np.argmin(np.abs(arr - float(q))))
        # last resort assumes 3-quantile ordering
        return {0.1: 0, 0.5: 1, 0.9: 2}[q]

    if pred_np.ndim == 4:
        # (B, V, Q, H) -> squeeze V for univariate
        pred_np = pred_np[:, 0, :, :]   # (B, Q, H)

    if pred_np.ndim == 3:
        lo  = pred_np[:, q_idx(0.1), :]
        med = pred_np[:, q_idx(0.5), :]
        hi  = pred_np[:, q_idx(0.9), :]
        return med, lo, hi

    if pred_np.ndim == 2:
        med = pred_np
        lo  = np.full_like(med, np.nan)
        hi  = np.full_like(med, np.nan)
        return med, lo, hi

    raise ValueError(f"Unsupported Chronos-2 output shape: {pred_np.shape}")

import numpy as np
import torch

def forecast_chronos_batch_from_start_any(
    pipeline,
    y_test_s,
    *,
    idxs,
    context_length=30,
    horizon=None,
    y_scaler=None,
    num_samples=100,  # only for v1
):
    y_test_s = np.asarray(y_test_s)
    idxs = np.asarray(idxs, dtype=int)
    N, T = y_test_s.shape

    H = (T - context_length) if horizon is None else int(horizon)
    H = max(1, H)

    contexts = []
    for i in idxs:
        y = y_test_s[i].astype(np.float32)
        if y_scaler is not None:
            y = y_scaler.inverse_transform(y.reshape(-1, 1)).ravel().astype(np.float32)

        ctx = y[:context_length]
        contexts.append(torch.tensor(ctx, dtype=torch.float32))

    context_2d = torch.stack(contexts, dim=0)  # (B, L)

    is_chronos2 = "chronos2" in type(pipeline).__name__.lower()

    if is_chronos2:
        context_3d = context_2d.unsqueeze(1)  # (B, 1, L)

        pred = pipeline.predict(context_3d, prediction_length=H)  # no num_samples
        pred_np = pred.detach().cpu().numpy() if torch.is_tensor(pred) else np.asarray(pred)

        med, lo, hi = _extract_quantiles_from_chronos2_output(pred_np, pipeline)
        return med, lo, hi

    # v1 (chronos-t5-*)
    pred = pipeline.predict(context_2d, prediction_length=H, num_samples=num_samples)  # (B,S,H)
    median = torch.quantile(pred, 0.5, dim=1).cpu().numpy()
    low    = torch.quantile(pred, 0.1, dim=1).cpu().numpy()
    high   = torch.quantile(pred, 0.9, dim=1).cpu().numpy()
    return median, low, high

#chronos exogneous future model
import numpy as np
import torch

def forecast_chronos2_oracle_one(
    pipeline,
    X_test_s,
    y_test_s,
    *,
    idx=0,
    context_length=30,
    min_horizon=60,
    y_scaler=None,
    x_scaler=None,
    quantiles=(0.1, 0.5, 0.9),
    covariate_names=None,   # optional list of names length F
):
    """
    Chronos-2 oracle forecast using FUTURE exogenous covariates.

    This Chronos-2 build expects:
      - target: (L,) or (V, L)
      - past_covariates: dict[str, Tensor] with each Tensor shape (L,)
      - future_covariates: dict[str, Tensor] with each Tensor shape (H,)

    Returns (med, lo, hi) each shape (H,).
    """
    X_test_s = np.asarray(X_test_s)
    y_test_s = np.asarray(y_test_s)

    T = y_test_s.shape[1]
    F = X_test_s.shape[2]
    H = max(int(min_horizon), int(T - context_length))

    if covariate_names is None:
        covariate_names = [f"feat_{k}" for k in range(F)]
    if len(covariate_names) != F:
        raise ValueError(f"covariate_names must have length {F}, got {len(covariate_names)}")

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

    y_full = inv_y(y_test_s[idx])  # (T,)
    x_full = inv_x(X_test_s[idx])  # (T,F)

    y_ctx = y_full[:context_length]  # (L,)
    x_past = x_full[:context_length, :]  # (L,F)
    x_fut  = x_full[context_length:context_length+H, :]  # (H,F)

    if x_fut.shape[0] < H:
        pad = H - x_fut.shape[0]
        x_fut = np.pad(x_fut, ((0, pad), (0, 0)), mode="edge")

    # ✅ target must be 1D (L,) or 2D (V,L). We'll give (L,)
    target = torch.tensor(y_ctx, dtype=torch.float32)

    # ✅ pack covariates as dict[str, Tensor(L,)] and dict[str, Tensor(H,)]
    past_covariates = {
        name: torch.tensor(x_past[:, j], dtype=torch.float32)   # (L,)
        for j, name in enumerate(covariate_names)
    }
    future_covariates = {
        name: torch.tensor(x_fut[:, j], dtype=torch.float32)    # (H,)
        for j, name in enumerate(covariate_names)
    }

    inp = [{
        "target": target,
        "past_covariates": past_covariates,
        "future_covariates": future_covariates,
    }]

    pred_list = pipeline.predict(inp, prediction_length=H)  # list[Tensor]
    pred = pred_list[0]
    pred_np = pred.detach().cpu().numpy()

    # Make pred_np be (Q,H)
    if pred_np.ndim == 4:          # (B,V,Q,H)
        pred_np = pred_np[0, 0, :, :]
    elif pred_np.ndim == 3:
        # could be (B,Q,H) or (V,Q,H)
        if pred_np.shape[0] == 1:
            pred_np = pred_np[0, :, :]
        else:
            pred_np = pred_np[0, :, :]
    elif pred_np.ndim == 2:
        pass
    else:
        raise ValueError(f"Unexpected Chronos-2 output shape: {pred_np.shape}")

    qs = list(getattr(pipeline, "quantiles", []))

    def q_idx(q):
        if q in qs:
            return qs.index(q)
        if len(qs):
            arr = np.asarray(qs, dtype=float)
            return int(np.argmin(np.abs(arr - float(q))))
        return {0.1: 0, 0.5: 1, 0.9: 2}[q]

    lo  = pred_np[q_idx(quantiles[0]), :]
    med = pred_np[q_idx(quantiles[1]), :]
    hi  = pred_np[q_idx(quantiles[2]), :]

    return med.astype(np.float32), lo.astype(np.float32), hi.astype(np.float32)



# Wrapper Function


def forecast_all_foundation_models_from_start_new(
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
            med, lo, hi = forecast_chronos_batch_from_start_any(
                model, y_test_s,
                idxs=idxs,
                context_length=context_length,
                horizon=horizon,
                y_scaler=y_scaler,
                num_samples=num_samples,
            )
            out[name] = {"type": "chronos", "median": med, "low": lo, "high": hi}

        elif "moirai" in name:
            med, lo, hi = forecast_moirai_any_batch_from_start_new(
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

# Plotting the forecasting results:

import numpy as np
import matplotlib.pyplot as plt

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

    # --- Oracle FUTURE-EXOG models (pass these in, and they will ALWAYS be plotted) ---
    chronos2_pipeline_for_oracle=None,   # e.g., loaded_models["chronos_v2"]
    moirai_module_for_oracle=None,       # e.g., loaded_models["moirai_small_v2"] or v1 module
    x_scaler=None,
    covariate_names=None,               # e.g., ["NApp","Rain","SolarRad","AirTempC"] (Chronos-2 needs this)
    freq="D",
    num_samples_oracle=200,             # used by Moirai oracle (and sometimes Moirai2 predictor)
    start_timestamp="2021-01-01",
):
    meta = foundation_results["meta"]
    H = int(meta["horizon"])
    idxs_used = list(meta["idxs"])

    if idx not in idxs_used:
        raise ValueError(
            f"idx={idx} not in foundation_results['meta']['idxs']={idxs_used}. "
            "Re-run wrapper with idx included."
        )
    bpos = idxs_used.index(idx)

    # ---- True in original units ----
    y_true_full = y_scaler.inverse_transform(
        np.asarray(y_test_s[idx], dtype=np.float32).reshape(-1, 1)
    ).ravel()
    T = len(y_true_full)

    # ---- Rolling prediction (scaled -> original) ----
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

    # ---- Plot range: context + horizon ----
    end = min(T, window + H)
    x = np.arange(end)

    plt.figure(figsize=(13, 6))

    # Context (0..window-1): true
    plt.plot(x[:window], y_true_full[:window], linewidth=2.5, label="True (context)")

    # True after window
    plt.plot(x[window:], y_true_full[window:end], linewidth=3.0, label="True (after window)")

    # Separator
    plt.axvline(window - 0.5, linewidth=1.5)

    # Rolling dashed after window
    plt.plot(
        x[window:],
        y_pred_roll[window:end],
        linestyle="--",
        linewidth=2.2,
        label=f"Trained model (window={window},chunk={chunk})",
    )

    # ---- Foundation models (past-only median curves from foundation_results) ----
    styles = ["-.", ":", (0, (5, 2)), (0, (3, 1, 1, 1)), (0, (7, 2, 2, 2))]
    style_i = 0

    for name, res in foundation_results.items():
        if name == "meta":
            continue
        if not isinstance(res, dict) or "median" not in res:
            continue

        med_all = np.asarray(res["median"])
        try:
            med = med_all[bpos]
        except Exception:
            med = med_all

        med = np.asarray(med).squeeze()  # (H,)

        y_f = np.full((end,), np.nan, dtype=np.float32)
        h_use = min(len(med), end - window)
        y_f[window:window + h_use] = med[:h_use]

        ls = styles[style_i % len(styles)]
        style_i += 1
        plt.plot(x, y_f, linestyle=ls, linewidth=2.0, label=f"{name} (median)")

    # ---- ALWAYS: Oracle Moirai with FUTURE exog (if module is provided) ----
    if moirai_module_for_oracle is not None:
        med_o, lo_o, hi_o = forecast_moirai_oracle_one_any(
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

        med_o = np.asarray(med_o).squeeze()
        y_o = np.full((end,), np.nan, dtype=np.float32)
        h_use = min(len(med_o), end - window)
        y_o[window:window + h_use] = med_o[:h_use]

        plt.plot(
            x,
            y_o,
            linestyle=(0, (2, 2)),
            linewidth=2.3,
            label="moirai_future_exog (median)",
        )

    # ---- ALWAYS: Oracle Chronos-2 with FUTURE exog (if pipeline is provided) ----
    if chronos2_pipeline_for_oracle is not None:
        # Chronos-2 covariates need names (dict keys). If not provided, auto-generate.
        F = np.asarray(X_test_s).shape[2]
        if covariate_names is None:
            covariate_names = [f"feat_{k}" for k in range(F)]

        med_c, lo_c, hi_c = forecast_chronos2_oracle_one(
            chronos2_pipeline_for_oracle,
            X_test_s,
            y_test_s,
            idx=idx,
            context_length=window,
            min_horizon=max(60, T - window),
            y_scaler=y_scaler,
            x_scaler=x_scaler,
            covariate_names=covariate_names,
        )

        med_c = np.asarray(med_c).squeeze()
        y_c = np.full((end,), np.nan, dtype=np.float32)
        h_use = min(len(med_c), end - window)
        y_c[window:window + h_use] = med_c[:h_use]

        plt.plot(
            x,
            y_c,
            linestyle=(0, (1, 2)),
            linewidth=2.3,
            label="chronos2_future_exog (median)",
        )

    plt.xlabel("Day index", fontsize=15)
    plt.ylabel("Target (Kg/Ha)", fontsize=15)
    if title:
        plt.title(title, fontsize=15)
    plt.legend(fontsize=13, loc="upper right")
    plt.tight_layout()
    plt.show()
