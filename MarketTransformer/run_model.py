"""
main.py
Loads OHLCV CSVs into memory, normalizes datasets, computes dataset-level Gabor metrics
using PyTorch FFT, and provides a ConcatDataset of sliding windows. Includes an MOE 
router that consumes per-dataset metadata.

This file implements a compact transformer-based model with sinusoidal positional
embeddings, a routed decoder block (simple two-expert interpolation gated by a
small router MLP), the three prediction heads described in the spec, and a training
loop with validation, diagnostics, and a simple inference/visualization pipeline.
"""

import os
from typing import List, Tuple, Optional, Dict

import math
import numpy as np
import pandas as pd

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset, TensorDataset

import matplotlib.pyplot as plt


# -----------------------------
# Data helpers (optimized)
# -----------------------------


def _normalize_df(df: pd.DataFrame, numeric_cols: List[str], 
                  mean_vals: Optional[pd.Series] = None, 
                  std_vals: Optional[pd.Series] = None) -> Tuple[pd.DataFrame, pd.Series, pd.Series]:
    """Z-score normalize numeric columns of a DataFrame."""
    if mean_vals is None or std_vals is None:
        mean_vals = df[numeric_cols].mean()
        std_vals = df[numeric_cols].std()
    
    df_copy = df.copy()
    df_copy[numeric_cols] = (df_copy[numeric_cols] - mean_vals) / std_vals
    return df_copy, mean_vals, std_vals


def compute_gabor_metric(signal: np.ndarray, fs: float = 1.0) -> float:
    """Compute a dataset-level Gabor metric using FFT power spectrum."""
    if signal.size == 0: 
        return 0.0
    
    x: torch.Tensor = torch.from_numpy(signal.astype(np.float32))
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
    
    return float(math.sqrt(max(var_t, 0.0) * max(var_f, 0.0)))


def _compute_gabor_metrics(df_norm: pd.DataFrame, fs: float = 1.0) -> Tuple[float, float]:
    """Helper to compute both Gabor metrics for a normalized dataframe."""
    vol_arr = df_norm["volume"].to_numpy(dtype=float)
    oc_arr = np.ravel(np.column_stack([
        df_norm["open"].to_numpy(dtype=float), 
        df_norm["close"].to_numpy(dtype=float)
    ]))
    return compute_gabor_metric(vol_arr, fs=fs), compute_gabor_metric(oc_arr, fs=fs)


class OHLCVWindowDataset(Dataset):
    """Sliding-window dataset built from a single normalized OHLCV DataFrame."""
    
    def __init__(self, df: pd.DataFrame, window_size: int = 64, 
                 target_size: int = 8, stride: int = 1):
        self.df = df
        self.window_size = int(window_size)
        self.target_size = int(target_size)
        self.stride = int(stride)
        self._windows = self._compute_windows()
        self.gabor_volume = 0.0
        self.gabor_oc = 0.0
    
    def _compute_windows(self) -> List[Tuple[int, int]]:
        n = len(self.df)
        if n < self.window_size: 
            return []
        return [(i, i + self.window_size) 
                for i in range(0, n - self.window_size + 1, self.stride)]
    
    def __len__(self) -> int: 
        return len(self._windows)
    
    def __getitem__(self, idx: int):
        start, end = self._windows[idx]
        w = self.df.iloc[start:end]
        arr = w[["open", "high", "low", "close", "volume"]].to_numpy(dtype=float)
        return arr.copy().astype(float), dict(zip("OHLC", arr[-1].tolist()))


# -----------------------------
# Dataset loading + splitting (optimized)
# -----------------------------


def _process_dataset(df: pd.DataFrame, numeric_cols: List[str], window_size: int, 
                     target_size: int, stride: int, fs: float = 1.0,
                     mean_vals: Optional[pd.Series] = None, 
                     std_vals: Optional[pd.Series] = None) -> Optional[OHLCVWindowDataset]:
    """Helper to process and create a dataset with Gabor metrics."""
    if len(df) < window_size:
        return None
    
    df_norm, _, _ = _normalize_df(df, numeric_cols, mean_vals, std_vals)
    ds = OHLCVWindowDataset(df_norm, window_size=window_size, 
                            target_size=target_size, stride=stride)
    ds.gabor_volume, ds.gabor_oc = _compute_gabor_metrics(df_norm, fs=fs)
    return ds


def build_train_val_datasets(paths: List[str], window_size: int = 128, 
                            target_size: int = 8, stride: int = 32, 
                            split_date: Optional[str] = "2024-01-01", 
                            fs: float = 1.0) -> Tuple[List[OHLCVWindowDataset], List[OHLCVWindowDataset]]:
    """Load CSVs, split by date, normalize, create OHLCVWindowDataset instances."""
    train_datasets, val_datasets = [], []
    numeric_cols = ["open", "high", "low", "close", "volume"]
    cutoff = pd.to_datetime(split_date) if split_date is not None else None
    
    for path in paths:
        df_raw = pd.read_csv(path, header=0, 
                           names=["date", "open", "high", "low", "close", "volume"], 
                           parse_dates=[0])
        
        if cutoff is None:
            df_train, df_val = df_raw.copy(), pd.DataFrame(columns=df_raw.columns)
        else:
            df_train = df_raw[df_raw["date"] < cutoff].copy()
            df_val = df_raw[df_raw["date"] >= cutoff].copy()
        
        # Process training dataset
        if ds_train := _process_dataset(df_train, numeric_cols, window_size, 
                                       target_size, stride, fs):
            train_datasets.append(ds_train)
            # Use train stats for validation normalization
            mean_vals = df_train[numeric_cols].mean()
            std_vals = df_train[numeric_cols].std()
        
        # Process validation dataset
        if ds_val := _process_dataset(df_val, numeric_cols, window_size, 
                                     target_size, stride, fs, mean_vals, std_vals):
            val_datasets.append(ds_val)
    
    return train_datasets, val_datasets


# -----------------------------
# Router used by decoder (sigmoid gating)
# -----------------------------


class RouterMLP(nn.Module):
    """Small MLP mapping [G_v, G_oc] -> r in [0,1]."""
    
    def __init__(self, hidden: int = 16, init_r: float = 0.5, 
                 init_w_std: float = 1e-3, noise_std: float = 0.05):
        super().__init__()
        self.fc1 = nn.Linear(2, hidden)
        self.act = nn.ReLU()
        self.fc2 = nn.Linear(hidden, 1)
        self.noise_std = float(noise_std)
        
        # Initialization
        nn.init.normal_(self.fc1.weight, mean=0.0, std=init_w_std)
        nn.init.zeros_(self.fc1.bias)
        nn.init.normal_(self.fc2.weight, mean=0.0, std=init_w_std)
        
        r_clamped = min(max(init_r, 1e-6), 1.0 - 1e-6)
        bias_val = math.log(r_clamped / (1.0 - r_clamped))
        with torch.no_grad():
            self.fc2.bias.fill_(bias_val)
    
    def forward(self, g_v: torch.Tensor, g_oc: torch.Tensor) -> torch.Tensor:
        x = torch.stack([g_v, g_oc], dim=-1).to(dtype=torch.float32)
        logits = self.fc2(self.act(self.fc1(x))).squeeze(-1)
        
        if self.training and self.noise_std > 0.0:
            logits = logits + torch.randn_like(logits) * self.noise_std
        
        return torch.sigmoid(logits)


# -----------------------------
# Sinusoidal positional embeddings
# -----------------------------


class SinusoidalPosEmb(nn.Module):
    """Sinusoidal positional embeddings (vectorized)."""
    
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim
    
    def forward(self, seq_len: int, device=None) -> torch.Tensor:
        pe = torch.zeros(seq_len, self.dim, device=device)
        position = torch.arange(0, seq_len, dtype=torch.float32, device=device).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, self.dim, 2, dtype=torch.float32, device=device) 
                           * (-math.log(10000.0) / self.dim))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        return pe


# -----------------------------
# Transformer-like encoder and routed decoder block
# -----------------------------


class EncoderBlock(nn.Module):
    """Unified encoder block: MHA + FFN + residuals."""
    
    def __init__(self, d_model: int = 128, nheads: int = 4, 
                 d_ff: int = 512, dropout: float = 0.1):
        super().__init__()
        self.attn = nn.MultiheadAttention(embed_dim=d_model, num_heads=nheads, 
                                         dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_ff), 
            nn.GELU(), 
            nn.Linear(d_ff, d_model)
        )
        self.norm2 = nn.LayerNorm(d_model)
    
    def forward(self, x: torch.Tensor, attn_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        y, _ = self.attn(x, x, x, attn_mask=attn_mask)
        x = self.norm1(x + y)
        x = self.norm2(x + self.ff(x))
        return x


class RoutedDecoderBlock(nn.Module):
    """Routed decoder block that interpolates two expert FFNs using routing weight r."""
    
    def __init__(self, d_model: int = 128, d_ff: int = 512):
        super().__init__()
        self.expert_a = nn.Sequential(
            nn.Linear(d_model, d_ff), 
            nn.GELU(), 
            nn.Linear(d_ff, d_model)
        )
        self.expert_b = nn.Sequential(
            nn.Linear(d_model, d_ff), 
            nn.GELU(), 
            nn.Linear(d_ff, d_model)
        )
        self.norm = nn.LayerNorm(d_model)
    
    def forward(self, x: torch.Tensor, r: torch.Tensor) -> torch.Tensor:
        r = r.view(-1, 1)
        out = r * self.expert_a(x) + (1.0 - r) * self.expert_b(x)
        return self.norm(x + out)


# -----------------------------
# Heads: mean/var, feat-var, trend (optimized)
# -----------------------------


def _create_head_network(d_model: int, output_dim: int) -> nn.Sequential:
    """Helper to create consistent head networks."""
    return nn.Sequential(
        nn.Linear(d_model, d_model), 
        nn.ReLU(), 
        nn.Linear(d_model, output_dim)
    )


class FFMeanVar(nn.Module):
    """Predicts window mean baseline and per-variable log-scales for O/C."""
    
    def __init__(self, d_model: int = 128):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(d_model, d_model), nn.ReLU())
        self.mu = nn.Linear(d_model, 1)           # mu_w
        self.s_o = nn.Linear(d_model, 1)          # pre-activation for log-scale of O
        self.s_c = nn.Linear(d_model, 1)          # pre-activation for log-scale of C
    
    def forward(self, h: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        x = self.net(h)
        return (self.mu(x).squeeze(-1), 
                self.s_o(x).squeeze(-1), 
                self.s_c(x).squeeze(-1))


class FFFeatVar(nn.Module):
    """Predicts feature-level variances (log-scale) for [O,C,H,L]."""
    
    def __init__(self, d_model: int = 128):
        super().__init__()
        self.net = _create_head_network(d_model, 4)
    
    def forward(self, h: torch.Tensor) -> torch.Tensor:
        return self.net(h)


class FFTrend(nn.Module):
    """Predict trend coefficients [a,b,c,d]."""
    
    def __init__(self, d_model: int = 128):
        super().__init__()
        self.net = _create_head_network(d_model, 4)
    
    def forward(self, h: torch.Tensor) -> torch.Tensor:
        return self.net(h)


# -----------------------------
# Full model
# -----------------------------


class MarketTransformer(nn.Module):
    """Compact model implementing encoder -> sinusoidal embeddings -> routed decoder -> heads."""
    
    def __init__(self, seq_len: int, d_model: int = 128, nheads: int = 4, 
                 n_layers: int = 3, d_ff: int = 512):
        super().__init__()
        self.seq_len = int(seq_len)
        self.d_model = d_model
        
        # Input projection
        self.in_proj = nn.Linear(5, d_model)
        self.pos_emb = SinusoidalPosEmb(d_model)
        
        # Encoder stack
        self.enc_blocks = nn.ModuleList([
            EncoderBlock(d_model=d_model, nheads=nheads, d_ff=d_ff) 
            for _ in range(n_layers)
        ])
        
        # Routed decoder
        self.router_mlp = RouterMLP(hidden=32, noise_std=.05)
        self.decoder = RoutedDecoderBlock(d_model=d_model, d_ff=d_ff)
        
        # Heads
        self.head_meanvar = FFMeanVar(d_model=d_model)
        self.head_featvar = FFFeatVar(d_model=d_model)
        self.head_trend = FFTrend(d_model=d_model)
    
    def _compute_trend_predictions(self, mu_w: torch.Tensor, 
                                 trend_coefs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Helper to compute trend-based predictions."""
        a, b, c, d = trend_coefs.unbind(-1)
        t = torch.ones_like(a)
        z = b * (t ** 3) + c * (t ** 2) + d * t
        z = torch.clamp(z, min=-10.0, max=10.0)
        tau = a * torch.tanh(z)
        return mu_w, mu_w + tau
    
    def forward(self, x: torch.Tensor, g_v: torch.Tensor, 
                g_oc: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Forward pass."""
        device = x.device
        b, t, f = x.shape
        assert t == self.seq_len, "Input sequence length mismatch"
        
        # Encoder processing
        h = self.in_proj(x) + self.pos_emb(self.seq_len, device=device).unsqueeze(0)
        for blk in self.enc_blocks:
            h = blk(h)
        
        # Routing and decoding
        h_pooled = h.mean(dim=1)
        r = self.router_mlp(g_v, g_oc)
        h_dec = self.decoder(h_pooled, r)
        
        # Generate predictions
        mu_w, s_o, s_c = self.head_meanvar(h_dec)
        o_pred, c_pred = self._compute_trend_predictions(mu_w, self.head_trend(h_dec))
        
        return {
            'o_pred_base': o_pred,
            'mu_w': mu_w,
            's_o': s_o,
            's_c': s_c,
            'feat_log_scales': self.head_featvar(h_dec),
            'trend_coefs': self.head_trend(h_dec),
            'r_gate': r,
        }


# -----------------------------
# Loss functions (optimized)
# -----------------------------


def _softplus_pos(u: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    return torch.nn.functional.softplus(u) + eps


def _dist_loss(dist_func, *args, reduction: str = 'none', **kwargs) -> torch.Tensor:
    """Helper for distribution loss functions."""
    loss = dist_func(*args, reduction='none', **kwargs)
    if reduction == 'mean': return loss.mean()
    if reduction == 'sum': return loss.sum()
    return loss


def gaussian_nll(y: torch.Tensor, mu: torch.Tensor, log_scale: torch.Tensor, 
                 reduction: str = 'none', eps: float = 1e-6) -> torch.Tensor:
    """Gaussian negative log-likelihood."""
    scale = _softplus_pos(log_scale, eps=eps)
    dist = torch.distributions.Normal(loc=mu, scale=scale)
    return _dist_loss(lambda: -dist.log_prob(y), reduction=reduction)


def half_normal_nll(residual: torch.Tensor, log_scale: torch.Tensor, 
                    reduction: str = 'none', eps: float = 1e-8) -> torch.Tensor:
    """Half-normal negative log-likelihood."""
    r_clamped = residual.clamp(min=0.0)
    scale = _softplus_pos(log_scale, eps=eps)
    term = (r_clamped.pow(2)) / (2.0 * scale.pow(2))
    loss = torch.log(scale) + term
    return _dist_loss(lambda: loss, reduction=reduction)


def hinge_penalties(o_pred: torch.Tensor, c_pred: torch.Tensor, 
                    h_true: torch.Tensor, l_true: torch.Tensor, 
                    lambda_h: float = 1.0, lambda_l: float = 1.0, 
                    reduction: str = 'none') -> torch.Tensor:
    """Hinge loss penalties for high/low constraints."""
    base_max, base_min = torch.max(o_pred, c_pred), torch.min(o_pred, c_pred)
    viol_h = torch.relu(base_max - h_true)
    viol_l = torch.relu(l_true - base_min)
    pen = lambda_h * viol_h.pow(2) + lambda_l * viol_l.pow(2)
    return _dist_loss(lambda: pen, reduction=reduction)


def stage1_loss_open_close(o_true: torch.Tensor, o_pred: torch.Tensor, s_o: torch.Tensor,
                          c_true: torch.Tensor, c_pred: torch.Tensor, s_c: torch.Tensor,
                          reduction: str = 'mean') -> torch.Tensor:
    """Stage 1 loss for open and close predictions."""
    nll = gaussian_nll(o_true, o_pred, s_o, reduction='none') + \
          gaussian_nll(c_true, c_pred, s_c, reduction='none')
    return _dist_loss(lambda: nll, reduction=reduction)


def stage2_loss_high_low(h_true: torch.Tensor, l_true: torch.Tensor, 
                         base_o: torch.Tensor, base_c: torch.Tensor, 
                         use_pred_base: bool, o_pred: torch.Tensor, c_pred: torch.Tensor,
                         log_scale_h: torch.Tensor, log_scale_l: torch.Tensor, 
                         option: str = 'half_normal', lambda_h: float = 1.0, 
                         lambda_l: float = 1.0, reduction: str = 'mean') -> torch.Tensor:
    """Stage 2 loss for high and low predictions."""
    if use_pred_base:
        base_upper, base_lower = torch.max(o_pred, c_pred), torch.min(o_pred, c_pred)
    else:
        base_upper, base_lower = torch.max(base_o, base_c), torch.min(base_o, base_c)
    
    r_h, r_l = h_true - base_upper, base_lower - l_true
    
    if option == 'half_normal':
        nll = half_normal_nll(r_h, log_scale_h, reduction='none') + \
              half_normal_nll(r_l, log_scale_l, reduction='none')
    else:
        raise ValueError("Unsupported Stage-2 option")
    
    total = nll + hinge_penalties(o_pred, c_pred, h_true, l_true, 
                                  lambda_h=lambda_h, lambda_l=lambda_l, reduction='none')
    return _dist_loss(lambda: total, reduction=reduction)


def total_loss(preds: Dict[str, torch.Tensor], targets: Dict[str, torch.Tensor], 
               config: Optional[Dict] = None) -> torch.Tensor:
    """Total loss combining stage 1 and stage 2 losses."""
    if config is None: 
        config = {}
    
    # Extract predictions and targets
    o_true, c_true, h_true, l_true = targets['O'], targets['C'], targets['H'], targets['L']
    mu_w, s_o, s_c = preds['mu_w'], preds['s_o'], preds['s_c']
    
    # Compute predictions
    o_pred, c_pred = mu_w, mu_w + preds['trend_coefs'][:, 0] * torch.tanh(
        torch.clamp(preds['trend_coefs'][:, 1] + preds['trend_coefs'][:, 2] + preds['trend_coefs'][:, 3], 
                   min=-10.0, max=10.0)
    )
    
    # Stage 1 loss
    l1 = stage1_loss_open_close(o_true, o_pred, s_o, c_true, c_pred, s_c, reduction='none')
    
    # Stage 2 loss
    log_scale_h, log_scale_l = preds['feat_log_scales'][:, 2], preds['feat_log_scales'][:, 3]
    l2 = stage2_loss_high_low(
        h_true, l_true, o_true, c_true, config.get('use_pred_base', True), 
        o_pred, c_pred, log_scale_h, log_scale_l, 
        option=config.get('stage2_option', 'half_normal'),
        lambda_h=config.get('lambda_h', 1.0),
        lambda_l=config.get('lambda_l', 1.0),
        reduction='none'
    )
    
    return (l1 + l2).mean()


# -----------------------------
# Utility: materialize windows into tensors for training (optimized)
# -----------------------------


def build_tensor_dataloader(datasets: List[OHLCVWindowDataset], 
                           batch_size: int = 64, 
                           max_windows: Optional[int] = None) -> DataLoader:
    """Materialize windows from multiple datasets into a TensorDataset."""
    inputs, targets, g_metrics = [], [], []
    
    for ds in datasets:
        for (start, end) in ds._windows:
            if max_windows is not None and len(inputs) >= max_windows:
                break
                
            w = ds.df.iloc[start:end][["open", "high", "low", "close", "volume"]]
            inputs.append(w.to_numpy(dtype=np.float32))
            last = w.iloc[-1].to_numpy(dtype=float)
            targets.append(last[:4])  # O, H, L, C
            g_metrics.append([ds.gabor_volume, ds.gabor_oc])
        
        if max_windows is not None and len(inputs) >= max_windows:
            break
    
    if len(inputs) == 0:
        raise RuntimeError("No windows available for training.")
    
    # Stack all data
    X = torch.tensor(np.stack(inputs, axis=0), dtype=torch.float32)
    O, H, L, C = torch.tensor(np.array(targets), dtype=torch.float32).T
    Gv, Goc = torch.tensor(np.array(g_metrics), dtype=torch.float32).T
    
    print(f"Available Samples: {len(X)}; Batch Size: {batch_size}")
    return DataLoader(TensorDataset(X, O, C, H, L, Gv, Goc), 
                     batch_size=batch_size, shuffle=True)


# -----------------------------
# Inference utilities and visualization (optimized)
# -----------------------------


def _compute_predictions(model: MarketTransformer, batch: Tuple) -> Dict[str, torch.Tensor]:
    """Helper to compute model predictions from a batch."""
    X, O, C, H, L, Gv, Goc = batch
    device = X.device
    
    X = X.to(device)
    preds = model(X, Gv.to(device), Goc.to(device))
    
    # Compute trend predictions
    mu_w = preds['mu_w']
    a, b, c, d = preds['trend_coefs'].unbind(-1)
    t = torch.ones_like(a)
    z = b * (t ** 3) + c * (t ** 2) + d * t
    z = torch.clamp(z, min=-10.0, max=10.0)
    tau = a * torch.tanh(z)
    
    o_pred, c_pred = mu_w, mu_w + tau
    
    # Compute high/low predictions
    sigma_h = _softplus_pos(preds['feat_log_scales'][:, 2])
    sigma_l = _softplus_pos(preds['feat_log_scales'][:, 3])
    mean_rh = sigma_h * math.sqrt(2.0 / math.pi)
    mean_rl = sigma_l * math.sqrt(2.0 / math.pi)
    
    h_pred = torch.max(o_pred, c_pred) + mean_rh
    l_pred = torch.min(o_pred, c_pred) - mean_rl
    
    return {
        'o_pred': o_pred, 'c_pred': c_pred, 'h_pred': h_pred, 'l_pred': l_pred,
        'r_gate': preds['r_gate'], 'O': O, 'C': C, 'H': H, 'L': L
    }


def infer_on_dataloader(model: MarketTransformer, dl: DataLoader, 
                       device: torch.device, output_dir: str = "./output", 
                       max_plots: int = 200) -> pd.DataFrame:
    """Run inference on a DataLoader and return predictions DataFrame."""
    model.eval()
    os.makedirs(output_dir, exist_ok=True)
    rows = []
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(dl):
            preds = _compute_predictions(model, batch)
            
            # Collect results
            batch_size = len(preds['O'])
            for i in range(batch_size):
                rows.append({
                    'O_true': float(preds['O'][i].item()),
                    'C_true': float(preds['C'][i].item()),
                    'H_true': float(preds['H'][i].item()),
                    'L_true': float(preds['L'][i].item()),
                    'O_pred': float(preds['o_pred'][i].item()),
                    'C_pred': float(preds['c_pred'][i].item()),
                    'H_pred': float(preds['h_pred'][i].item()),
                    'L_pred': float(preds['l_pred'][i].item()),
                    'r_gate': float(preds['r_gate'][i].item()),
                })
            
            # Optional plotting for first batch
            if batch_idx < 1 and batch_idx * dl.batch_size < max_plots:
                plt.figure(figsize=(8, 4))
                plt.plot(preds['C'][:max_plots].cpu().numpy(), label='C_true')
                plt.plot(preds['c_pred'][:max_plots].cpu().numpy(), label='C_pred')
                plt.legend()
                plt.title('Sample close true vs pred (first batch)')
                plt.tight_layout()
                plt.savefig(os.path.join(output_dir, 'inference_sample_close.png'))
                plt.close()
    
    return pd.DataFrame(rows)


# -----------------------------
# Visualization functions (optimized)
# -----------------------------


def _create_plot(x_data, y_data, title, xlabel="", ylabel="", 
                save_path=None, **kwargs):
    """Helper to create standardized plots."""
    plt.figure(figsize=kwargs.get('figsize', (8, 4)))
    plt.plot(x_data, y_data, **{k: v for k, v in kwargs.items() 
                                if k not in ['figsize', 'save_path']})
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    if 'legend' in kwargs:
        plt.legend()
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
    plt.close()


def visualize_training_diagnostics(history: dict, output_dir: str = "./output"):
    """Plot training diagnostics."""
    os.makedirs(output_dir, exist_ok=True)
    
    if 'train_loss' in history and 'val_loss' in history:
        _create_plot(range(len(history['train_loss'])), history['train_loss'], 
                    "Loss over epochs", "Epoch", "Loss",
                    label="train loss", color='blue',
                    save_path=os.path.join(output_dir, 'loss_curve.png'))
        plt.plot(history['val_loss'], label='val loss', color='orange')
        plt.legend()
        plt.savefig(os.path.join(output_dir, 'loss_curve.png'))
        plt.close()
    
    if 'train_mae' in history and 'val_mae' in history:
        _create_plot(range(len(history['train_mae'])), history['train_mae'],
                    "MAE over epochs", "Epoch", "MAE",
                    label="train MAE", color='blue',
                    save_path=os.path.join(output_dir, 'mae_curve.png'))
        plt.plot(history['val_mae'], label='val MAE', color='orange')
        plt.legend()
        plt.savefig(os.path.join(output_dir, 'mae_curve.png'))
        plt.close()


def visualize_inference(df: pd.DataFrame, output_dir: str = "./output", 
                       max_pts: int = 5000):
    """Create inference visualizations."""
    os.makedirs(output_dir, exist_ok=True)
    dfc = df.iloc[:max_pts].copy()
    
    # Diagnostic plots
    plt.figure(figsize=(6, 4))
    plt.hist(df['r_gate'].values, bins=20)
    plt.title('Router gate distribution')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'gate_hist.png'))
    plt.close()
    
    # Residual plots
    resid = df['C_pred'].values - df['C_true'].values
    plt.figure(figsize=(6, 4))
    plt.hist(resid, bins=40)
    plt.title('Close residuals')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'close_residuals.png'))
    plt.close()
    
    # Scatter plots
    fig, axes = plt.subplots(1, 2, figsize=(16, 4))
    for ax, (true_col, pred_col, title) in zip(axes, 
        [('H_true', 'H_pred', 'High: True vs Predicted'),
         ('L_true', 'L_pred', 'Low: True vs Predicted')]):
        ax.scatter(dfc[true_col], dfc[pred_col], s=4, alpha=.7)
        min_val, max_val = dfc[true_col].min(), dfc[true_col].max()
        ax.plot([min_val, max_val], [min_val, max_val], '--')
        ax.set_title(title)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'scatter_high_low.png'))
    plt.close()
    
    # Time series plot
    _create_plot(range(len(dfc)), dfc['C_true'], 
                "Close Price Prediction vs Truth",
                "Sample", "Close Price",
                label='C_true', linewidth=1.2,
                save_path=os.path.join(output_dir, 'close_vs_truth.png'))
    plt.plot(dfc['C_pred'], label='C_pred', linewidth=1.2)
    plt.legend()
    plt.savefig(os.path.join(output_dir, 'close_vs_truth.png'))
    plt.close()


# -----------------------------
# Training loop with validation + diagnostics (optimized)
# -----------------------------


def _compute_batch_loss(model: MarketTransformer, batch: Tuple, 
                       device: torch.device, config: Dict) -> Tuple[torch.Tensor, Dict]:
    """Compute loss and predictions for a batch."""
    X, O, C, H, L, Gv, Goc = (f.to(device) for f in batch)
    
    preds = model(X, Gv, Goc)
    preds_for_loss = {
        'mu_w': preds['mu_w'], 's_o': preds['s_o'], 's_c': preds['s_c'],
        'trend_coefs': preds['trend_coefs'], 'feat_log_scales': preds['feat_log_scales']
    }
    targets = {'O': O, 'C': C, 'H': H, 'L': L}
    
    loss = total_loss(preds_for_loss, targets, config)
    return loss, preds


def train_model(datasets_train: List[OHLCVWindowDataset], 
               datasets_val: List[OHLCVWindowDataset], 
               seq_len: int, epochs: int = 5, 
               batch_size: int = 64, 
               device: Optional[torch.device] = None) -> Tuple[nn.Module, Dict]:
    """Main training function with validation."""
    if device is None: 
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Create data loaders
    train_dl = build_tensor_dataloader(datasets_train, batch_size=batch_size)
    val_dl = build_tensor_dataloader(datasets_val, batch_size=batch_size) if datasets_val else None
    
    # Initialize model and optimizer
    model = MarketTransformer(seq_len=seq_len, d_model=128, 
                             nheads=4, n_layers=3, d_ff=512).to(device)
    optim = torch.optim.AdamW(model.parameters(), lr=1e-4)
    
    # Training history
    history = {
        'train_loss': [], 'val_loss': [], 'val_mae_close': [],
        'gate_mean': [], 'gate_std': [], 'gate_entropy': []
    }
    
    # Training loop
    for ep in range(1, epochs + 1):
        model.train()
        epoch_stats = _train_epoch(model, train_dl, optim, device, ep, epochs)
        
        # Update history
        for key in ['train_loss', 'gate_mean', 'gate_std', 'gate_entropy']:
            history[key].append(epoch_stats[key])
        
        # Decay router noise
        if hasattr(model, 'router_mlp') and model.router_mlp.noise_std > 0.0:
            model.router_mlp.noise_std = max(0.0, model.router_mlp.noise_std * 0.9)
        
        # Validation
        if val_dl:
            val_loss, val_mae = _validate_epoch(model, val_dl, device)
            history['val_loss'].append(val_loss)
            history['val_mae_close'].append(val_mae)
        
        print(f"Epoch {ep}/{epochs} train_loss={epoch_stats['train_loss']:.6f} "
              f"val_loss={val_loss:.6f} gate_mean={epoch_stats['gate_mean']:.4f} "
              f"gate_std={epoch_stats['gate_std']:.4f} "
              f"gate_entropy={epoch_stats['gate_entropy']:.4f} "
              f"val_mae_close={val_mae}")
    
    return model, history


def _train_epoch(model: nn.Module, train_dl: DataLoader, optim: torch.optim.Optimizer,
                device: torch.device, ep: int, epochs: int) -> Dict:
    """Train for one epoch."""
    total_loss_val, total_batches = 0.0, 0
    gate_vals, gate_entropies = [], []
    entropy_coeff, eps = 0.05, 1e-8
    
    for batch_idx, batch in enumerate(train_dl, 1):
        # Forward pass and loss computation
        loss, preds = _compute_batch_loss(model, batch, device, 
                                         {'stage2_option': 'half_normal', 
                                          'use_pred_base': True, 
                                          'lambda_h': 1.0, 'lambda_l': 1.0})
        
        # Add entropy regularizer
        if 'r_gate' in preds:
            r = preds['r_gate'].clamp(min=eps, max=1.0 - eps)
            entropy = -(r * torch.log(r) + (1.0 - r) * torch.log(1.0 - r))
            loss = loss - entropy_coeff * entropy.mean()
            gate_vals.append(float(r.mean().item()))
            gate_entropies.append(float(entropy.mean().item()))
        
        # Backward pass
        optim.zero_grad()
        loss.backward()
        
        # Gradient clipping and step
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optim.step()
        
        # Update statistics
        total_loss_val += loss.item()
        total_batches += 1
        
        # Logging
        if batch_idx % 50 == 0:
            lr = optim.param_groups[0]['lr']
            print(f"Epoch {ep} batch {batch_idx}: loss={loss.item():.6f} "
                  f"lr={lr:.6e} gate_mean={gate_vals[-1] if gate_vals else 0.0:.4f}")
    
    return {
        'train_loss': total_loss_val / max(1, total_batches),
        'gate_mean': np.mean(gate_vals) if gate_vals else 0.0,
        'gate_std': np.std(gate_vals) if gate_vals else 0.0,
        'gate_entropy': np.mean(gate_entropies) if gate_entropies else 0.0
    }


def _validate_epoch(model: nn.Module, val_dl: DataLoader, 
                   device: torch.device) -> Tuple[float, float]:
    """Validate for one epoch."""
    model.eval()
    val_total, val_batches, clos_resid = 0.0, 0, []
    
    with torch.no_grad():
        for batch in val_dl:
            loss, preds = _compute_batch_loss(model, batch, device,
                                             {'stage2_option': 'half_normal',
                                              'use_pred_base': True})
            val_total += loss.item()
            val_batches += 1
            
            # Compute close residuals
            a, b, c, d = preds['trend_coefs'].unbind(-1)
            tau = a * torch.tanh(torch.clamp(b + c + d, min=-10.0, max=10.0))
            c_pred = preds['mu_w'] + tau
            clos_resid.extend((c_pred - batch[2].to(device)).cpu().numpy().tolist())
    
    return (val_total / max(1, val_batches), 
            float(np.mean(np.abs(clos_resid))) if clos_resid else None)


# -----------------------------
# Runtime / entrypoint
# -----------------------------


class DataConfig:
    window: int = 256
    target: int = 32
    stride: int = 16
    batch: int = 64


def main():
    """Main entry point."""
    fp = "./data/PSE/"
    paths = [os.path.join(fp, f) for f in os.listdir(fp) if f.endswith(".csv")]
    
    train_ds, val_ds = build_train_val_datasets(
        paths, 
        window_size=DataConfig.window, 
        target_size=DataConfig.target, 
        stride=DataConfig.stride, 
        split_date="2024-01-01"
    )
    
    print(f"Train datasets: {len(train_ds)}; Val datasets: {len(val_ds)}")
    
    if not train_ds:
        print("No training datasets found; exiting")
        return
    
    # Train model
    model, history = train_model(
        train_ds, val_ds, 
        seq_len=DataConfig.window, 
        epochs=5, 
        batch_size=DataConfig.batch
    )
    
    # Inference and visualization
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if val_ds:
        val_dl = build_tensor_dataloader(val_ds, batch_size=DataConfig.batch, max_windows=5000)
        df_infer = infer_on_dataloader(model, val_dl, device)
        
        visualize_inference(df_infer)
        visualize_training_diagnostics(history)
        print(df_infer.describe().loc[['mean', 'std']][['C_true', 'C_pred', 'r_gate']])
    
    # Save model
    torch.save(model, "./MarketTransformer/MarketTransformer.pth")
    print("Training complete")


if __name__ == "__main__":
    main()