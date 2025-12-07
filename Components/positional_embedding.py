"""Positional embedding helpers.

Provides a vectorized sinusoidal positional embedding implementation
compatible with transformer-style models.
"""

import torch
import torch.nn as nn

import numpy as np



class SinusoidalPosEmb(nn.Module):
    """Sinusoidal positional embeddings (vectorized).

    Usage:
        pos = SinusoidalPosEmb(dim)(seq_len, device=device)
        # pos has shape (seq_len, dim)
    """

    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, seq_len: int, device=None) -> torch.Tensor:
        """Return sinusoidal positional encodings for a sequence length.

        Args:
            seq_len: Number of positions to encode.
            device: Optional torch device to place the tensor on.

        Returns:
            Tensor of shape (seq_len, dim) with sinusoidal embeddings.
        """
        pe = torch.zeros(seq_len, self.dim, device=device)
        position    = torch.arange(0, seq_len, dtype=torch.float32, device=device).unsqueeze(1)
        div_term    = torch.exp(torch.arange(0, self.dim, 2, dtype=torch.float32, device=device) * (-np.log(10000.0) / self.dim))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        return pe