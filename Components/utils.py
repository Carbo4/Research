from typing import Tuple, List, Optional
from .dataset import OHLCVWindowDataset
from .MarketTransformer import MarketTransformer
from torch.utils.data import DataLoader, TensorDataset

import torch
import os

import matplotlib.pyplot as plt
import pandas as pd


"""Utility helpers: normalization, dataloader builders, and inference helpers.

This module contains commonly used helper functions for preparing datasets,
building DataLoaders and running simple inference loops that materialize
predictions into a pandas DataFrame.
"""


def _softplus_pos(u: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Numerically stable softplus that enforces a small positive floor.

    Args:
        u: Pre-activation tensor.
        eps: Small positive constant added to the softplus output.

    Returns:
        Tensor with strictly positive elements.
    """
    return torch.nn.functional.softplus(u) + eps


def _normalize_df(df: pd.DataFrame, numeric_cols: List[str], mean_vals: Optional[pd.Series] = None, std_vals: Optional[pd.Series] = None) -> Tuple[pd.DataFrame, pd.Series, pd.Series]:
    """Z-score normalize numeric columns of a DataFrame.

    If mean_vals/std_vals are provided they are used; otherwise computed from df.
    Returns (df_normalized, mean_vals, std_vals).
    """
    if mean_vals is None or std_vals is None:
        mean_vals = df[numeric_cols].mean()
        std_vals = df[numeric_cols].std()
    df_copy = df.copy()
    df_copy[numeric_cols] = (df_copy[numeric_cols] - mean_vals) / std_vals
    return df_copy, mean_vals, std_vals


def compute_gabor_metric(signal: torch.Tensor, fs: float = 1.0) -> float:
    """Compute a dataset-level Gabor metric using FFT power spectrum.

    Converts input to torch tensor, computes FFT, then computes variance of power
    spectrum along frequency and time-like axes. Returns sqrt(var_time * var_freq).
    """
    if signal.size == 0:
        return 0.0

    x: torch.Tensor = torch.from_numpy(signal.astype(torch.float32))

    fft_x: torch.Tensor = torch.fft.fft(x)
    power: torch.Tensor = (fft_x.real ** 2 + fft_x.imag ** 2)

    freqs: torch.Tensor = torch.fft.fftfreq(x.size(0), d=1.0/fs)
    times: torch.Tensor = torch.arange(x.size(0), dtype=torch.float32) / fs

    power_sum: float = float(power.sum().item())
    if power_sum <= 0.0:
        return 0.0

    power_norm: torch.Tensor = power / power_sum

    t_mean: float = float((power_norm * times).sum().item())
    f_mean: float = float((power_norm * freqs).sum().item())

    var_t: float = float((power_norm * (times - t_mean) ** 2).sum().item())
    var_f: float = float((power_norm * (freqs - f_mean) ** 2).sum().item())

    metric: float = float(torch.sqrt(max(var_t, 0.0) * max(var_f, 0.0)))
    return metric


def build_tensor_dataloader(datasets: List[OHLCVWindowDataset], batch_size: int = 64, max_windows: Optional[int] = None) -> DataLoader:
    """Materialize windows from multiple OHLCVWindowDataset instances into a TensorDataset.

    This iterates each dataset's internal windows and stacks the input windows and targets
    into tensors. Returns a PyTorch DataLoader wrapping a TensorDataset.
    """
    inputs, Os, Cs, Hs, Ls, g_vs, g_ocs = [], [], [], [], [], [], []

    for ds in datasets:
        for (start, end) in ds._windows:
            w = torch.tensor(ds.df.iloc[start:end][["open", "high", "low", "close", "volume"]].to_numpy(), dtype=torch.float32)
            inputs.append(w); last = w[-1]
            Os.append(float(last[0])); Hs.append(float(last[1])); Ls.append(float(last[2])); Cs.append(float(last[3]))
            g_vs.append(float(ds.gabor_volume)); g_ocs.append(float(ds.gabor_oc))
            if max_windows is not None and len(inputs) >= max_windows:
                break
        if max_windows is not None and len(inputs) >= max_windows:
            break

    if len(inputs) == 0:
        raise RuntimeError("No windows available for training. Check dataset/window_size settings.")

    X = torch.stack(inputs, axis=0).to(dtype=torch.float32)
    O, C, H, L, Gv, Goc = torch.tensor([Os, Cs, Hs, Ls, g_vs, g_ocs], dtype=torch.float32).unbind(0)

    tensor_ds = TensorDataset(X, O, C, H, L, Gv, Goc)
    dl = DataLoader(tensor_ds, batch_size=batch_size, shuffle=True)
    print(f"Available Samples: {len(tensor_ds)}; Batch Size: {batch_size}")
    return dl


def build_train_val_datasets(paths: List[str], window_size: int = 128, target_size: int = 8, stride: int = 32, split_date: Optional[str] = "2024-01-01", fs: float = 1.0) -> Tuple[List[OHLCVWindowDataset], List[OHLCVWindowDataset]]:
    """Load CSVs, split by date, normalize (train statistics applied to val),
    create OHLCVWindowDataset instances for train and validation sets.

    Returns (train_datasets, val_datasets).
    """
    train_datasets: List[OHLCVWindowDataset] = []
    val_datasets:   List[OHLCVWindowDataset] = []

    numeric_cols: List[str] = ["open", "high", "low", "close", "volume"]
    cutoff = pd.to_datetime(split_date) if split_date is not None else None

    for path in paths:
        df_raw: pd.DataFrame = pd.read_csv(path, header=0, names=["date", "open", "high", "low", "close", "volume"], parse_dates=[0])
        if cutoff is None:
            df_train = df_raw.copy()
            df_val = pd.DataFrame(columns=df_raw.columns)
        else:
            df_train = df_raw[df_raw["date"] < cutoff].copy()
            df_val = df_raw[df_raw["date"] >= cutoff].copy()

        if len(df_train) >= window_size:
            df_train_norm, mean_vals, std_vals = _normalize_df(df_train, numeric_cols)
            ds_train = OHLCVWindowDataset(df_train_norm, window_size=window_size, target_size=target_size, stride=stride)
            # compute gabor on normalized train df
            vol_arr = df_train_norm["volume"].to_numpy(dtype=float)
            oc_arr = torch.ravel(torch.column_stack([df_train_norm["open"].to_numpy(dtype=float), df_train_norm["close"].to_numpy(dtype=float)]))
            ds_train.gabor_volume = compute_gabor_metric(vol_arr, fs=fs)
            ds_train.gabor_oc = compute_gabor_metric(oc_arr, fs=fs)
            train_datasets.append(ds_train)

        if len(df_val) >= window_size:
            # apply train stats if available, else compute from val itself
            if 'mean_vals' in locals():
                df_val_norm, _, _ = _normalize_df(df_val, numeric_cols, mean_vals, std_vals)
            else:
                df_val_norm, _, _ = _normalize_df(df_val, numeric_cols)

            ds_val = OHLCVWindowDataset(df_val_norm, window_size=window_size, target_size=target_size, stride=stride)
            vol_arr = df_val_norm["volume"].to_numpy(dtype=float)
            oc_arr = torch.ravel(torch.column_stack([df_val_norm["open"].to_numpy(dtype=float), df_val_norm["close"].to_numpy(dtype=float)]))
            ds_val.gabor_volume = compute_gabor_metric(vol_arr, fs=fs)
            ds_val.gabor_oc = compute_gabor_metric(oc_arr, fs=fs)
            val_datasets.append(ds_val)

    return train_datasets, val_datasets



def infer_on_dataloader(model: MarketTransformer, dl: DataLoader, device: torch.device, output_dir: str = "./output", max_plots: int = 200) -> pd.DataFrame:
    """Run inference on a DataLoader and return a DataFrame with predictions and truths.

    Predictions use expected residual from half-normal for H/L: E[r] = sigma * sqrt(2/pi).
    """
    model.eval()
    os.makedirs(output_dir, exist_ok=True)
    rows = []

    with torch.no_grad():
        for batch_idx, batch in enumerate(dl):
            X, O, C, H, L, Gv, Goc = batch
            X = X.to(device); O = O.to(device); C = C.to(device); H = H.to(device); L = L.to(device); Gv = Gv.to(device); Goc = Goc.to(device)
            preds = model(X, Gv, Goc)

            mu_w = preds['mu_w']
            trend = preds['trend_coefs']
            a, b, c, d = trend.unbind(-1)
            t = torch.ones_like(a)
            z = b * (t ** 3) + c * (t ** 2) + d * t
            z = torch.clamp(z, min=-10.0, max=10.0)
            tau = a * torch.tanh(z)

            o_pred = mu_w
            c_pred = mu_w + tau

            feat_log_scales = preds['feat_log_scales']
            sigma_h = _softplus_pos(feat_log_scales[:, 2])
            sigma_l = _softplus_pos(feat_log_scales[:, 3])
            mean_rh = sigma_h * torch.sqrt(2.0 / torch.pi)
            mean_rl = sigma_l * torch.sqrt(2.0 / torch.pi)

            h_pred = torch.max(o_pred, c_pred) + mean_rh
            l_pred = torch.min(o_pred, c_pred) - mean_rl

            r_gate = preds['r_gate']

            for i in range(len(O)):
                rows.append({
                    'O_true': float(O[i].cpu().item()),
                    'C_true': float(C[i].cpu().item()),
                    'H_true': float(H[i].cpu().item()),
                    'L_true': float(L[i].cpu().item()),
                    'O_pred': float(o_pred[i].cpu().item()),
                    'C_pred': float(c_pred[i].cpu().item()),
                    'H_pred': float(h_pred[i].cpu().item()),
                    'L_pred': float(l_pred[i].cpu().item()),
                    'r_gate': float(r_gate[i].cpu().item()),
                })

            # optional quick plot for early batches
            if batch_idx < 1 and batch_idx * dl.batch_size < max_plots:
                plt.figure(figsize=(8, 4))
                plt.plot([i for i in range(min(len(O), max_plots))], [float(x) for x in O[:max_plots].cpu().numpy()], label='C_true')
                plt.plot([i for i in range(min(len(O), max_plots))], [float(x) for x in c_pred[:max_plots].cpu().numpy()], label='C_pred')
                plt.legend(); plt.title('Sample close true vs pred (first batch)')
                plt.tight_layout(); plt.savefig(os.path.join(output_dir, 'inference_sample_close.png')); plt.close()

    return pd.DataFrame(rows)