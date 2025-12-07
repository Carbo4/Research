"""Model components for the MarketTransformer architecture.

This module contains small, well-documented neural building blocks used in
the project: routing MLPs, encoder blocks, a routed decoder block and the
composite `MarketTransformer` model that wires them together.
"""

from .positional_embedding import SinusoidalPosEmb
from typing import Dict, Optional, Tuple

import torch.nn as nn
import torch


class RouterMLP(nn.Module):
    """Small MLP mapping [G_v, G_oc] -> r in [0,1].

    The router maps two scalar dataset/window metrics (e.g. Gabor metrics)
    to a scalar routing weight in [0,1] via a tiny MLP. Small-weight
    initialization and optional gaussian noise encourage exploration and
    avoid early collapse to deterministic routing.
    """

    def __init__(self, hidden: int = 16, init_r: float = 0.5, init_w_std: float = 1e-3, noise_std: float = 0.05):
        super().__init__()
        self.fc1        = nn.Linear(2, hidden)
        self.act        = nn.ReLU()
        self.fc2        = nn.Linear(hidden, 1)
        self.noise_std  = float(noise_std)

        # careful initialization: small weights, bias set to logit(init_r)
        nn.init.normal_(self.fc1.weight, mean=0.0, std=init_w_std)
        nn.init.zeros_(self.fc1.bias)
        nn.init.normal_(self.fc2.weight, mean=0.0, std=init_w_std)
        # convert init_r in (0,1) to logit for bias; guard numerical extremes
        r_clamped = min(max(init_r, 1e-6), 1.0 - 1e-6)
        bias_val = torch.log(r_clamped / (1.0 - r_clamped))
        with torch.no_grad():
            self.fc2.bias.fill_(bias_val)

    def forward(self, g_v: torch.Tensor, g_oc: torch.Tensor) -> torch.Tensor:
        """Compute routing weight `r` for a batch.

        Args:
            g_v: Tensor of shape (batch,) representing volume-based metric.
            g_oc: Tensor of shape (batch,) representing open/close metric.

        Returns:
            Tensor of shape (batch,) with values in [0,1].
        """
        x = torch.stack([g_v, g_oc], dim=-1).to(dtype=torch.float32)
        h = self.act(self.fc1(x))
        logits = self.fc2(h).squeeze(-1)
        # optional small gaussian noise to encourage exploration early in training
        if self.training and self.noise_std > 0.0:
            logits = logits + torch.randn_like(logits) * self.noise_std
        r = torch.sigmoid(logits)
        return r


class EncoderBlock(nn.Module):
    """Unified encoder block: multi-head attention, residuals and FFN.

    This block mirrors a transformer encoder layer with pre/post layernorm
    and a simple feed-forward network activated by GELU.
    """

    def __init__(self, d_model: int = 128, nheads: int = 4, d_ff: int = 512, dropout: float = 0.1):
        super().__init__()
        self.attn   = nn.MultiheadAttention(embed_dim=d_model, num_heads=nheads, dropout=dropout, batch_first=True)
        self.norm1  = nn.LayerNorm(d_model)
        self.ff     = nn.Sequential(nn.Linear(d_model, d_ff), nn.GELU(), nn.Linear(d_ff, d_model))
        self.norm2  = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor, attn_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Apply attention and feed-forward layers to `x`.

        Args:
            x: Tensor shaped (batch, seq_len, d_model).
            attn_mask: Optional attention mask forwarded to PyTorch MHA.

        Returns:
            Tensor of same shape as input.
        """
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
        """Decode a pooled embedding `x` using routing weight `r`.

        Args:
            x: Tensor of shape (batch, d_model).
            r: Tensor of shape (batch,) with routing weights in [0,1].

        Returns:
            Decoded tensor of shape (batch, d_model).
        """
        # x: (batch, d_model), r: (batch,) in [0,1]
        a = self.expert_a(x)
        b = self.expert_b(x)
        r = r.view(-1, 1)
        out = r * a + (1.0 - r) * b
        return self.norm(x + out)
    
    
class FFMeanVar(nn.Module):
    """Predicts window mean baseline and per-variable log-scales for O/C.

    The head returns `(mu_w, s_o, s_c)` where `s_o` and `s_c` are pre-activation
    values that are later transformed into positive scales via a softplus.
    """

    def __init__(self, d_model: int = 128):
        super().__init__()
        self.net    = nn.Sequential(nn.Linear(d_model, d_model), nn.ReLU())
        self.mu     = nn.Linear(d_model, 1)           # mu_w
        self.s_o    = nn.Linear(d_model, 1)          # pre-activation for log-scale of O
        self.s_c    = nn.Linear(d_model, 1)          # pre-activation for log-scale of C

    def forward(self, h: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return `(mu_w, s_o, s_c)` for a batch of embeddings `h`.

        Args:
            h: Tensor of shape (batch, d_model).

        Returns:
            Tuple of 1-D tensors (mu_w, s_o, s_c) each shaped (batch,).
        """
        x = self.net(h)
        mu_w    = self.mu(x).squeeze(-1)
        s_o     = self.s_o(x).squeeze(-1)
        s_c     = self.s_c(x).squeeze(-1)
        return mu_w, s_o, s_c


class FFFeatVar(nn.Module):
    """Predicts feature-level variances (log-scale) for [O,C,H,L]."""

    def __init__(self, d_model: int = 128):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(d_model, d_model), nn.ReLU(), nn.Linear(d_model, 4))

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        """Return 4-d pre-activations corresponding to log-scales for [O,C,H,L]."""
        return self.net(h)  # returns 4-d pre-activations (log-scale)


class FFTrend(nn.Module):
    """Predict trend coefficients [a,b,c,d]."""

    def __init__(self, d_model: int = 128):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(d_model, d_model), nn.ReLU(), nn.Linear(d_model, 4))

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        """Return trend coefficients tensor of shape (batch, 4)."""
        return self.net(h)
    
    
class MarketTransformer(nn.Module):
    """Compact model implementing encoder -> sinusoidal embeddings -> routed decoder -> heads.

    This model consumes a window of OHLCV features and returns a dictionary of
    predictions and diagnostics used by the training loop and inference helpers.
    """

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

        Args:
            x: Tensor shaped `(batch, seq_len, 5)` containing raw OHLCV features.
            g_v, g_oc: Tensors of shape `(batch,)` containing dataset/window-level metrics
                       used to compute the routing gate `r`.

        Returns:
            Dictionary with prediction tensors and diagnostics. Keys include:
                - 'mu_w', 's_o', 's_c' : mean and pre-scale activations for O/C
                - 'feat_log_scales'    : pre-activations for per-feature scales [O,C,H,L]
                - 'trend_coefs'        : trend coefficients used to compute close offset
                - 'r_gate'             : routing weight per sample
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