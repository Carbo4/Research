"""Visualization helpers for training and inference diagnostics.

Provides simple plotting utilities that materialize PNGs into an output
directory. These functions are intentionally lightweight and depend on
matplotlib and pandas.
"""

import os
import torch

import matplotlib.pyplot as plt
import pandas as pd


def visualize_training_diagnostics(history: dict, output_dir: str = "./output"):
    """Save training diagnostic plots from a `history` dictionary.

    Expected keys in `history` include `train_loss`, `val_loss` and optional
    `train_mae`/`val_mae`. Plots are saved into `output_dir`.
    """
    os.makedirs(output_dir, exist_ok=True)

    if 'train_loss' in history and 'val_loss' in history:
        plt.figure(figsize=(7,4)); plt.plot(history['train_loss'], label="train loss"); plt.plot(history['val_loss'], label='val loss')
        plt.legend(); plt.title("Loss over epochs"); plt.tight_layout(); plt.savefig(os.path.join(output_dir,'loss_curve.png')); plt.close()

    if 'train_mae' in history and 'val_mae' in history:
        plt.figure(figsize=(7,4)); plt.plot(history['train_mae'], label="train MAE"); plt.plot(history['val_mae'], label='val MAE')
        plt.legend(); plt.title("MAE over epochs"); plt.tight_layout(); plt.savefig(os.path.join(output_dir,'mae_curve.png')); plt.close()


def visualize_inference(df: pd.DataFrame, output_dir: str = "./output", max_pts: int = 5000):
    """Save a set of inference diagnostic plots from a predictions DataFrame.

    The function expects `df` to contain columns `C_true`, `C_pred`, `H_true`,
    `H_pred`, `L_true`, `L_pred` and `r_gate`.
    """
    os.makedirs(output_dir, exist_ok=True)

    dfc = df.copy()
    dfc = dfc.iloc[:max_pts]

    # diagnostics plots
    plt.figure(figsize=(6, 4))
    plt.hist(df['r_gate'].values, bins=20)
    plt.title('Router gate distribution'); plt.tight_layout(); plt.savefig(os.path.join(output_dir, 'gate_hist.png')); plt.close()

    resid = df['C_pred'].values - df['C_true'].values
    plt.figure(figsize=(6, 4)); plt.hist(resid, bins=40); plt.title('Close residuals'); plt.tight_layout(); plt.savefig(os.path.join(output_dir, 'close_residuals.png')); plt.close()


    plt.figure(figsize=(8,4)); plt.scatter(dfc['H_true'], dfc['H_pred'], s=4, alpha=.7); plt.plot([dfc['H_true'].min(),dfc['H_true'].max()],[dfc['H_true'].min(),dfc['H_true'].max()], '--')
    plt.title("High: True vs Predicted"); plt.tight_layout(); plt.savefig(os.path.join(output_dir,'scatter_high.png')); plt.close()

    plt.figure(figsize=(8,4)); plt.scatter(dfc['L_true'], dfc['L_pred'], s=4, alpha=.7); plt.plot([dfc['L_true'].min(),dfc['L_true'].max()],[dfc['L_true'].min(),dfc['L_true'].max()], '--')
    plt.title("Low: True vs Predicted"); plt.tight_layout(); plt.savefig(os.path.join(output_dir,'scatter_low.png')); plt.close()

    close_dates = torch.arange(len(dfc))
    plt.figure(figsize=(10,4)); plt.plot(close_dates, dfc['C_true'], label='C_true', linewidth=1.2)
    plt.plot(close_dates, dfc['C_pred'], label='C_pred', linewidth=1.2)
    plt.title("Close Price Prediction vs Truth"); plt.legend(); plt.tight_layout(); plt.savefig(os.path.join(output_dir,'close_vs_truth.png')); plt.close()