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

from typing import List, Tuple, Optional, Dict

import os
import math
import datetime as dt
import numpy as np
import pandas as pd

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset, TensorDataset

import matplotlib.pyplot as plt


# -----------------------------
# Data helpers (unchanged)
# -----------------------------


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


def compute_gabor_metric(signal: np.ndarray, fs: float = 1.0) -> float:
    """Compute a dataset-level Gabor metric using FFT power spectrum.

    Converts input to torch tensor, computes FFT, then computes variance of power
    spectrum along frequency and time-like axes. Returns sqrt(var_time * var_freq).
    """
    if signal.size == 0: return 0.0

    x: torch.Tensor = torch.from_numpy(signal.astype(np.float32))

    fft_x: torch.Tensor = torch.fft.fft(x)
    power: torch.Tensor = (fft_x.real ** 2 + fft_x.imag ** 2)

    freqs: torch.Tensor = torch.fft.fftfreq(x.size(0), d=1.0/fs)
    times: torch.Tensor = torch.arange(x.size(0), dtype=torch.float32) / fs

    power_sum: float = float(power.sum().item())
    if power_sum <= 0.0: return 0.0

    power_norm: torch.Tensor = power / power_sum

    t_mean: float = float((power_norm * times).sum().item())
    f_mean: float = float((power_norm * freqs).sum().item())

    var_t: float = float((power_norm * (times - t_mean) ** 2).sum().item())
    var_f: float = float((power_norm * (freqs - f_mean) ** 2).sum().item())

    metric: float = float(math.sqrt(max(var_t, 0.0) * max(var_f, 0.0)))
    return metric


class OHLCVWindowDataset(Dataset):
    """Sliding-window dataset built from a single normalized OHLCV DataFrame.

    Instances expose dataset-level metadata attributes: gabor_volume and gabor_oc.
    A static constructor performs loading, normalization, dataset creation, and
    computation of the Gabor metrics.
    """

    def __init__(self, df: pd.DataFrame, window_size: int = 64, target_size: int = 8, stride: int = 1):
        self.df:            pd.DataFrame           = df
        self.window_size:   int                    = int(window_size)
        self.target_size:   int                    = int(target_size)
        self.stride:        int                    = int(stride)
        self._windows:      List[Tuple[int, int]]  = self._compute_windows()
        self.gabor_volume:  float                  = 0.0
        self.gabor_oc:      float                  = 0.0

    def _compute_windows(self) -> List[Tuple[int, int]]:
        n: int = len(self.df)
        if n < self.window_size: return []
        return [(i, i + self.window_size) for i in range(0, n - self.window_size + 1, self.stride)]

    def __len__(self) -> int: return len(self._windows)

    def __getitem__(self, idx: int):
        start, end          = self._windows[idx]
        w:    pd.DataFrame  = self.df.iloc[start:end]
        arr: np.ndarray     = w[["open", "high", "low", "close", "volume"]].to_numpy(dtype=float)

        # inputs: full window, targets: last timestep's open,high,low,close
        inp: np.ndarray = arr.copy().astype(float)
        last: np.ndarray = arr[-1]

        out_dict = dict(zip("OHLC", last.tolist()))
        return inp, out_dict


# -----------------------------
# Dataset loading + splitting
# -----------------------------


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
            oc_arr = np.ravel(np.column_stack([df_train_norm["open"].to_numpy(dtype=float), df_train_norm["close"].to_numpy(dtype=float)]))
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
            oc_arr = np.ravel(np.column_stack([df_val_norm["open"].to_numpy(dtype=float), df_val_norm["close"].to_numpy(dtype=float)]))
            ds_val.gabor_volume = compute_gabor_metric(vol_arr, fs=fs)
            ds_val.gabor_oc = compute_gabor_metric(oc_arr, fs=fs)
            val_datasets.append(ds_val)

    return train_datasets, val_datasets


# -----------------------------
# Router used by decoder (sigmoid gating)
# -----------------------------


class RouterMLP(nn.Module):
    """Small MLP mapping [G_v, G_oc] -> r in [0,1].

    Improved initialization and optional noise + entropy regularizer to avoid
    gating collapse. By default initializes the final bias to logit(0.5)=0 and
    uses very small initial weights so the router starts near uniform routing.
    """

    def __init__(self, hidden: int = 16, init_r: float = 0.5, init_w_std: float = 1e-3, noise_std: float = 0.05):
        super().__init__()
        self.fc1 = nn.Linear(2, hidden)
        self.act = nn.ReLU()
        self.fc2 = nn.Linear(hidden, 1)
        self.noise_std = float(noise_std)

        # careful initialization: small weights, bias set to logit(init_r)
        nn.init.normal_(self.fc1.weight, mean=0.0, std=init_w_std)
        nn.init.zeros_(self.fc1.bias)
        nn.init.normal_(self.fc2.weight, mean=0.0, std=init_w_std)
        # convert init_r in (0,1) to logit for bias; guard numerical extremes
        r_clamped = min(max(init_r, 1e-6), 1.0 - 1e-6)
        bias_val = math.log(r_clamped / (1.0 - r_clamped))
        with torch.no_grad():
            self.fc2.bias.fill_(bias_val)

    def forward(self, g_v: torch.Tensor, g_oc: torch.Tensor) -> torch.Tensor:
        x = torch.stack([g_v, g_oc], dim=-1).to(dtype=torch.float32)
        h = self.act(self.fc1(x))
        logits = self.fc2(h).squeeze(-1)
        # optional small gaussian noise to encourage exploration early in training
        if self.training and self.noise_std > 0.0:
            logits = logits + torch.randn_like(logits) * self.noise_std
        r = torch.sigmoid(logits)
        return r


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
        div_term = torch.exp(torch.arange(0, self.dim, 2, dtype=torch.float32, device=device) * (-math.log(10000.0) / self.dim))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        return pe


# -----------------------------
# Transformer-like encoder and routed decoder block
# -----------------------------


class EncoderBlock(nn.Module):
    """Unified encoder block: MHA + FFN + residuals."""

    def __init__(self, d_model: int = 128, nheads: int = 4, d_ff: int = 512, dropout: float = 0.1):
        super().__init__()
        self.attn = nn.MultiheadAttention(embed_dim=d_model, num_heads=nheads, dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(nn.Linear(d_model, d_ff), nn.GELU(), nn.Linear(d_ff, d_model))
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor, attn_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        y, _ = self.attn(x, x, x, attn_mask=attn_mask)
        x = self.norm1(x + y)
        z = self.ff(x)
        x = self.norm2(x + z)
        return x


class RoutedDecoderBlock(nn.Module):
    """Routed decoder block that interpolates two expert FFNs using routing weight r.

    The block accepts a pooled encoder representation (batch, d_model) and returns
    a decoded embedding of the same size.
    """

    def __init__(self, d_model: int = 128, d_ff: int = 512):
        super().__init__()
        self.expert_a = nn.Sequential(nn.Linear(d_model, d_ff), nn.GELU(), nn.Linear(d_ff, d_model))
        self.expert_b = nn.Sequential(nn.Linear(d_model, d_ff), nn.GELU(), nn.Linear(d_ff, d_model))
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor, r: torch.Tensor) -> torch.Tensor:
        # x: (batch, d_model), r: (batch,) in [0,1]
        a = self.expert_a(x)
        b = self.expert_b(x)
        r = r.view(-1, 1)
        out = r * a + (1.0 - r) * b
        return self.norm(x + out)


# -----------------------------
# Heads: mean/var, feat-var, trend
# -----------------------------


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
        mu_w = self.mu(x).squeeze(-1)
        s_o  = self.s_o(x).squeeze(-1)
        s_c  = self.s_c(x).squeeze(-1)
        return mu_w, s_o, s_c


class FFFeatVar(nn.Module):
    """Predicts feature-level variances (log-scale) for [O,C,H,L]."""

    def __init__(self, d_model: int = 128):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(d_model, d_model), nn.ReLU(), nn.Linear(d_model, 4))

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        return self.net(h)  # returns 4-d pre-activations (log-scale)


class FFTrend(nn.Module):
    """Predict trend coefficients [a,b,c,d]."""

    def __init__(self, d_model: int = 128):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(d_model, d_model), nn.ReLU(), nn.Linear(d_model, 4))

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        return self.net(h)


# -----------------------------
# Full model
# -----------------------------


class MarketTransformer(nn.Module):
    """Compact model implementing encoder -> sinusoidal embeddings -> routed decoder -> heads."""

    def __init__(self, seq_len: int, d_model: int = 128, nheads: int = 4, n_layers: int = 3, d_ff: int = 512):
        super().__init__()
        self.seq_len = int(seq_len)
        self.d_model = d_model

        # input projection from 5 features -> d_model
        self.in_proj = nn.Linear(5, d_model)
        self.pos_emb = SinusoidalPosEmb(d_model)

        # encoder stack
        self.enc_blocks = nn.ModuleList([EncoderBlock(d_model=d_model, nheads=nheads, d_ff=d_ff) for _ in range(n_layers)])

        # routed decoder
        self.router_mlp = RouterMLP(hidden=32, noise_std=.05)
        self.decoder = RoutedDecoderBlock(d_model=d_model, d_ff=d_ff)

        # heads
        self.head_meanvar = FFMeanVar(d_model=d_model)
        self.head_featvar = FFFeatVar(d_model=d_model)
        self.head_trend   = FFTrend(d_model=d_model)

    def forward(self, x: torch.Tensor, g_v: torch.Tensor, g_oc: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Forward pass.

        x: (batch, seq_len, 5) raw features
        g_v, g_oc: (batch,) dataset/window-level metrics used to compute gate r
        """
        device = x.device
        b, t, f = x.shape
        assert t == self.seq_len, "Input sequence length mismatch"

        h = self.in_proj(x)
        pos = self.pos_emb(self.seq_len, device=device)
        h = h + pos.unsqueeze(0)

        for blk in self.enc_blocks:
            h = blk(h)

        # pool across time (mean) to produce window embedding
        h_pooled = h.mean(dim=1)

        # routing weight per sample
        r = self.router_mlp(g_v, g_oc)
        h_dec = self.decoder(h_pooled, r)

        mu_w, s_o, s_c = self.head_meanvar(h_dec)
        feat_log_scales = self.head_featvar(h_dec)
        trend_coefs = self.head_trend(h_dec)

        return {
            'o_pred_base': mu_w,               # baseline for open predictions
            'mu_w': mu_w,                      # same name used in spec
            's_o': s_o,                        # pre-activation for O
            's_c': s_c,                        # pre-activation for C
            'feat_log_scales': feat_log_scales,
            'trend_coefs': trend_coefs,
            'r_gate': r,
        }


# -----------------------------
# Loss functions (reuse earlier implementations)
# -----------------------------


def _softplus_pos(u: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    return torch.nn.functional.softplus(u) + eps


def gaussian_nll(y: torch.Tensor, mu: torch.Tensor, log_scale: torch.Tensor, reduction: str = 'none', eps: float = 1e-6) -> torch.Tensor:
    scale: torch.Tensor = _softplus_pos(log_scale, eps=eps)
    dist = torch.distributions.Normal(loc=mu, scale=scale)
    nll: torch.Tensor = -dist.log_prob(y)
    if reduction == 'mean': return nll.mean()
    if reduction == 'sum':  return nll.sum()
    return nll


def half_normal_nll(residual: torch.Tensor, log_scale: torch.Tensor, reduction: str = 'none', eps: float = 1e-8) -> torch.Tensor:
    r_clamped: torch.Tensor = residual.clamp(min=0.0)
    scale:     torch.Tensor = _softplus_pos(log_scale, eps=eps)
    term:      torch.Tensor = (r_clamped.pow(2)) / (2.0 * scale.pow(2))
    nll:       torch.Tensor = torch.log(scale) + term
    if reduction == 'mean': return nll.mean()
    if reduction == 'sum':  return nll.sum()
    return nll


def hinge_penalties(o_pred: torch.Tensor, c_pred: torch.Tensor, h_true: torch.Tensor, l_true: torch.Tensor, lambda_h: float = 1.0, lambda_l: float = 1.0, reduction: str = 'none') -> torch.Tensor:
    base_max: torch.Tensor = torch.max(o_pred, c_pred)
    base_min: torch.Tensor = torch.min(o_pred, c_pred)
    viol_h: torch.Tensor = torch.relu(base_max - h_true)
    viol_l: torch.Tensor = torch.relu(l_true - base_min)
    p_h: torch.Tensor = lambda_h * viol_h.pow(2)
    p_l: torch.Tensor = lambda_l * viol_l.pow(2)
    pen: torch.Tensor = p_h + p_l
    if reduction == 'mean': return pen.mean()
    if reduction == 'sum':  return pen.sum()
    return pen


def stage1_loss_open_close(o_true: torch.Tensor, o_pred: torch.Tensor, s_o: torch.Tensor, c_true: torch.Tensor, c_pred: torch.Tensor, s_c: torch.Tensor, reduction: str = 'mean') -> torch.Tensor:
    nll_o: torch.Tensor = gaussian_nll(o_true, o_pred, s_o, reduction='none')
    nll_c: torch.Tensor = gaussian_nll(c_true, c_pred, s_c, reduction='none')
    nll:   torch.Tensor = nll_o + nll_c
    if reduction == 'mean': return nll.mean()
    if reduction == 'sum':  return nll.sum()
    return nll


def stage2_loss_high_low(h_true: torch.Tensor, l_true: torch.Tensor, base_o: torch.Tensor, base_c: torch.Tensor, use_pred_base: bool, o_pred: torch.Tensor, c_pred: torch.Tensor, log_scale_h: torch.Tensor, log_scale_l: torch.Tensor, option: str = 'half_normal', lambda_h: float = 1.0, lambda_l: float = 1.0, reduction: str = 'mean') -> torch.Tensor:
    if use_pred_base:
        base_upper: torch.Tensor = torch.max(o_pred, c_pred)
        base_lower: torch.Tensor = torch.min(o_pred, c_pred)
    else:
        base_upper: torch.Tensor = torch.max(base_o, base_c)
        base_lower: torch.Tensor = torch.min(base_o, base_c)

    r_h: torch.Tensor = h_true - base_upper
    r_l: torch.Tensor = base_lower - l_true

    if option == 'half_normal':
        nll_h: torch.Tensor = half_normal_nll(r_h, log_scale_h, reduction='none')
        nll_l: torch.Tensor = half_normal_nll(r_l, log_scale_l, reduction='none')
    else:
        raise ValueError("Unsupported Stage-2 option in this simplified training loop")

    nll: torch.Tensor = nll_h + nll_l
    hinge: torch.Tensor = hinge_penalties(o_pred, c_pred, h_true, l_true, lambda_h=lambda_h, lambda_l=lambda_l, reduction='none')
    total: torch.Tensor = nll + hinge
    if reduction == 'mean': return total.mean()
    if reduction == 'sum':  return total.sum()
    return total


def total_loss(preds: Dict[str, torch.Tensor], targets: Dict[str, torch.Tensor], config: Optional[Dict] = None) -> torch.Tensor:
    if config is None: config = {}
    stage2_option: str = config.get('stage2_option', 'half_normal')
    use_pred_base: bool = config.get('use_pred_base', True)
    lambda_h: float = float(config.get('lambda_h', 1.0))
    lambda_l: float = float(config.get('lambda_l', 1.0))

    o_true: torch.Tensor = targets['O']
    c_true: torch.Tensor = targets['C']
    h_true: torch.Tensor = targets['H']
    l_true: torch.Tensor = targets['L']

    # predictions
    mu_w:   torch.Tensor = preds['mu_w']
    s_o:    torch.Tensor = preds['s_o']
    s_c:    torch.Tensor = preds['s_c']

    # close is mu_w + tau(t_close)
    trend_coefs: torch.Tensor = preds['trend_coefs']
    a, b, c, d = trend_coefs.unbind(-1)
    t = torch.ones_like(a)
    z = b * (t ** 3) + c * (t ** 2) + d * t
    z = torch.clamp(z, min=-10.0, max=10.0)
    tau = a * torch.tanh(z)

    o_pred, c_pred = mu_w, mu_w + tau

    l1 = stage1_loss_open_close(o_true, o_pred, s_o, c_true, c_pred, s_c, reduction='none')

    # choose half-normal stage2 residual scales from feat_log_scales (indexes 0->O,1->C,2->H,3->L)
    feat_log_scales = preds['feat_log_scales']
    log_scale_h = feat_log_scales[:, 2]
    log_scale_l = feat_log_scales[:, 3]

    l2 = stage2_loss_high_low(
        h_true, l_true, o_true, c_true, use_pred_base, o_pred, c_pred, log_scale_h, log_scale_l, 
        option      = stage2_option, 
        lambda_h    = lambda_h, 
        lambda_l    = lambda_l, 
        reduction   = 'none'
    )

    total = l1 + l2
    return total.mean()


# -----------------------------
# Utility: materialize windows into tensors for training
# -----------------------------


def build_tensor_dataloader(datasets: List[OHLCVWindowDataset], batch_size: int = 64, max_windows: Optional[int] = None) -> DataLoader:
    """Materialize windows from multiple OHLCVWindowDataset instances into a TensorDataset.

    This iterates each dataset's internal windows and stacks the input windows and targets
    into tensors. Returns a PyTorch DataLoader wrapping a TensorDataset.
    """
    inputs, Os, Cs, Hs, Ls, g_vs, g_ocs = [], [], [], [], [], [], []

    for ds in datasets:
        for (start, end) in ds._windows:
            w = ds.df.iloc[start:end][["open", "high", "low", "close", "volume"]].to_numpy(dtype=np.float32)
            inputs.append(w)
            last = w[-1]
            Os.append(float(last[0])); Hs.append(float(last[1])); Ls.append(float(last[2])); Cs.append(float(last[3]))
            g_vs.append(float(ds.gabor_volume)); g_ocs.append(float(ds.gabor_oc))
            if max_windows is not None and len(inputs) >= max_windows: break
        if max_windows is not None and len(inputs) >= max_windows: break

    if len(inputs) == 0:
        raise RuntimeError("No windows available for training. Check dataset/window_size settings.")

    X = torch.tensor(np.stack(inputs, axis=0), dtype=torch.float32)
    O, C, H, L, Gv, Goc = torch.tensor([Os, Cs, Hs, Ls, g_vs, g_ocs], dtype=torch.float32).unbind(0)

    tensor_ds = TensorDataset(X, O, C, H, L, Gv, Goc)
    dl = DataLoader(tensor_ds, batch_size=batch_size, shuffle=True)
    print(f"Available Samples: {len(tensor_ds)}; Batch Size: {batch_size}")
    return dl


# -----------------------------
# Inference utilities and visualization
# -----------------------------


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
            mean_rh = sigma_h * math.sqrt(2.0 / math.pi)
            mean_rl = sigma_l * math.sqrt(2.0 / math.pi)

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


# ==== Advanced Visual Diagnostics ====
# Loss + MAE curves, feature variance spread, trend visualization and joint comparison

def visualize_training_diagnostics(history: dict, output_dir: str = "./output"):
    os.makedirs(output_dir, exist_ok=True)

    if 'train_loss' in history and 'val_loss' in history:
        plt.figure(figsize=(7,4)); plt.plot(history['train_loss'], label="train loss"); plt.plot(history['val_loss'], label='val loss')
        plt.legend(); plt.title("Loss over epochs"); plt.tight_layout(); plt.savefig(os.path.join(output_dir,'loss_curve.png')); plt.close()

    if 'train_mae' in history and 'val_mae' in history:
        plt.figure(figsize=(7,4)); plt.plot(history['train_mae'], label="train MAE"); plt.plot(history['val_mae'], label='val MAE')
        plt.legend(); plt.title("MAE over epochs"); plt.tight_layout(); plt.savefig(os.path.join(output_dir,'mae_curve.png')); plt.close()


def visualize_inference(df: pd.DataFrame, output_dir: str = "./output", max_pts: int = 5000):
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

    close_dates = np.arange(len(dfc))
    plt.figure(figsize=(10,4)); plt.plot(close_dates, dfc['C_true'], label='C_true', linewidth=1.2)
    plt.plot(close_dates, dfc['C_pred'], label='C_pred', linewidth=1.2)
    plt.title("Close Price Prediction vs Truth"); plt.legend(); plt.tight_layout(); plt.savefig(os.path.join(output_dir,'close_vs_truth.png')); plt.close()




# -----------------------------
# Training loop with validation + diagnostics
# -----------------------------


def train_model(datasets_train: List[OHLCVWindowDataset], datasets_val: List[OHLCVWindowDataset], seq_len: int, epochs: int = 5, batch_size: int = 64, device: Optional[torch.device] = None):
    if device is None: device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    train_dl: DataLoader = build_tensor_dataloader(datasets_train, batch_size=batch_size)
    val_dl: Optional[DataLoader] = None
    if datasets_val: val_dl = build_tensor_dataloader(datasets_val, batch_size=batch_size)

    # create model with router noise enabled initially
    model: MarketTransformer = MarketTransformer(seq_len=seq_len, d_model=128, nheads=4, n_layers=3, d_ff=512).to(device)
    optim = torch.optim.AdamW(model.parameters(), lr=1e-4)

    history = {'train_loss': [], 'val_loss': [], 'val_mae_close': [], 'gate_mean': [], 'gate_std': [], 'gate_entropy': []}

    entropy_coeff = 0.05
    eps = 1e-8

    for ep in range(1, epochs + 1):
        model.train()
        total_loss_val: float = 0.0
        total_batches: int = 0
        gate_vals = []
        gate_entropies = []

        for batch_idx, batch in enumerate(train_dl):
            X, O, C, H, L, Gv, Goc = (f.to(device) for f in batch)

            preds = model(X, Gv, Goc)
            preds_for_loss = {
                'mu_w': preds['mu_w'], 's_o': preds['s_o'], 's_c': preds['s_c'], 'trend_coefs': preds['trend_coefs'], 'feat_log_scales': preds['feat_log_scales']
            }
            targets = {'O': O, 'C': C, 'H': H, 'L': L}

            loss = total_loss(preds_for_loss, targets, config={'stage2_option': 'half_normal', 'use_pred_base': True, 'lambda_h': 1.0, 'lambda_l': 1.0})

            # add entropy regularizer on router gates to encourage non-collapsing routing
            r = preds.get('r_gate', None)
            if r is not None:
                r_clamped = r.clamp(min=eps, max=1.0 - eps)
                entropy = -(r_clamped * torch.log(r_clamped) + (1.0 - r_clamped) * torch.log(1.0 - r_clamped))
                entropy_mean = entropy.mean()
                entropy_loss = -float(entropy_coeff) * entropy_mean
                # add entropy regularizer to loss (negative -> encourages higher entropy)
                loss = loss + entropy_loss
            else:
                entropy_mean = torch.tensor(0.0)

            optim.zero_grad(); loss.backward()

            # gradient diagnostics
            grad_norm = 0.0
            for p in model.parameters():
                if p.grad is not None:
                    grad_norm += float(p.grad.data.norm(2).cpu().item() ** 2)
            grad_norm = math.sqrt(grad_norm)

            optim.step()

            total_loss_val += float(loss.detach().cpu().item())
            total_batches += 1

            if r is not None:
                r_mean = float(r.mean().detach().cpu().item())
                r_std = float(r.std().detach().cpu().item())
                gate_vals.append(r_mean)
                gate_entropies.append(float(entropy_mean.detach().cpu().item()))
            else:
                gate_vals.append(0.0); gate_entropies.append(0.0)

            if (batch_idx + 1) % 50 == 0:
                lr = optim.param_groups[0]['lr']
                print(f"Epoch {ep} batch {batch_idx+1}: loss={loss.item():.6f} grad_norm={grad_norm:.4f} lr={lr:.6e} gate_mean={gate_vals[-1]:.4f} gate_entropy={gate_entropies[-1]:.4f}")

        train_avg = total_loss_val / max(1, total_batches)
        history['train_loss'].append(train_avg)
        history['gate_mean'].append(np.mean(gate_vals) if gate_vals else 0.0)
        history['gate_std'].append(np.std(gate_vals) if gate_vals else 0.0)
        history['gate_entropy'].append(np.mean(gate_entropies) if gate_entropies else 0.0)

        # decay router noise to let gating sharpen later in training
        if hasattr(model, 'router_mlp') and getattr(model.router_mlp, 'noise_std', 0.0) > 0.0:
            model.router_mlp.noise_std = max(0.0, model.router_mlp.noise_std * 0.9)

        # validation
        val_loss = None
        val_mae_close = None
        if val_dl is not None:
            model.eval(); val_total = 0.0; val_batches = 0; clos_resid = []
            with torch.no_grad():
                for batch in val_dl:
                    X, O, C, H, L, Gv, Goc = (f.to(device) for f in batch)
                    preds = model(X, Gv, Goc)
                    preds_for_loss = {'mu_w': preds['mu_w'], 's_o': preds['s_o'], 's_c': preds['s_c'], 'trend_coefs': preds['trend_coefs'], 'feat_log_scales': preds['feat_log_scales']}
                    targets = {'O': O, 'C': C, 'H': H, 'L': L}
                    l = total_loss(preds_for_loss, targets, config={'stage2_option': 'half_normal', 'use_pred_base': True})
                    val_total += float(l.cpu().item()); val_batches += 1

                    # collect close residuals
                    a, b, c, d = preds['trend_coefs'].unbind(-1)
                    t = torch.ones_like(a)
                    z = b * (t ** 3) + c * (t ** 2) + d * t
                    z = torch.clamp(z, min=-10.0, max=10.0)
                    tau = a * torch.tanh(z)
                    o_pred = preds['mu_w']; c_pred = preds['mu_w'] + tau
                    clos_resid.extend((c_pred - C).detach().cpu().numpy().tolist())

            val_loss = val_total / max(1, val_batches)
            val_mae_close = float(np.mean(np.abs(np.array(clos_resid)))) if len(clos_resid) else None
            history['val_loss'].append(val_loss); history['val_mae_close'].append(val_mae_close)

        print(f"Epoch {ep}/{epochs} train_loss={train_avg:.6f} val_loss={float(val_loss):.6f} gate_mean={history['gate_mean'][-1]:.4f} gate_std={history['gate_std'][-1]:.4f} gate_entropy={history['gate_entropy'][-1]:.4f} val_mae_close={val_mae_close}")

    return model, history


# -----------------------------
# Runtime / entrypoint
# -----------------------------


class DataConfig:
    window: int = 256
    target: int = 32
    stride: int = 16
    batch: int = 64


def main(*args, **kwargs):
    fp = "./data/PSE/"
    paths = [fp+f for f in os.listdir(fp) if f.endswith(".csv")]

    train_ds, val_ds = build_train_val_datasets(paths, window_size=DataConfig.window, target_size=DataConfig.target, stride=DataConfig.stride, split_date="2024-01-01")

    print(f"Train datasets: {len(train_ds)}; Val datasets: {len(val_ds)}")

    if len(train_ds) == 0:
        print("No training datasets found; exiting")
        return

    # model, history = train_model (train_ds, val_ds, seq_len=DataConfig.window, epochs=5, batch_size=DataConfig.batch)
    model = torch.load("./MarketTransformer/MarketTransformer.pth", weights_only=False)

    # run inference on validation set if available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if val_ds:
        val_dl = build_tensor_dataloader(val_ds, batch_size=DataConfig.batch, max_windows=5000)
        df_infer = infer_on_dataloader(model, val_dl, device)
        visualize_inference(df_infer)
        # visualize_training_diagnostics(history)
        # quick numeric diagnostics
        print(df_infer.describe().loc[['mean','std']][['C_true','C_pred','r_gate']])
    # torch.save(model, "./MarketTransformer/MarketTransformer.pth")
    # print("Training complete")


if __name__ == "__main__":
    main()