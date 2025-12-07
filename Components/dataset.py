from typing import List, Tuple
from torch.utils.data import Dataset

import torch
import pandas as pd


"""Dataset utilities for sliding-window OHLCV data.

This module provides `OHLCVWindowDataset`, a lightweight PyTorch
`Dataset` that materializes sliding windows from a single normalized
OHLCV DataFrame. Each dataset exposes simple metadata (`gabor_volume`,
`gabor_oc`) that can be used by models or training scripts.
"""


class OHLCVWindowDataset(Dataset):
    """Sliding-window dataset built from a single normalized OHLCV DataFrame.

    Attributes:
        df (pd.DataFrame): Normalized DataFrame with columns ["open","high","low","close","volume"].
        window_size (int): Number of timesteps in each input window.
        target_size (int): Number of timesteps in prediction target (unused in this simplified dataset).
        stride (int): Step size between consecutive windows.
        _windows (List[Tuple[int,int]]): Cached list of (start, end) indices for windows.
        gabor_volume (float): Optional dataset-level metric computed externally.
        gabor_oc (float): Optional dataset-level metric computed externally.

    The dataset returns a tuple `(inputs, target_dict)` from `__getitem__`:
        - inputs: numpy-like array shaped (window_size, 5) of features [O,H,L,C,V]
        - target_dict: mapping with keys `'O'`,`'H'`,`'L'`,`'C'` for the last timestep
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
        """Compute start/end index tuples for sliding windows.

        Returns an empty list when the DataFrame is shorter than `window_size`.
        """
        n: int = len(self.df)
        if n < self.window_size:
            return []
        return [(i, i + self.window_size) for i in range(0, n - self.window_size + 1, self.stride)]

    def __len__(self) -> int:
        """Return number of available windows."""
        return len(self._windows)

    def __getitem__(self, idx: int):
        """Return the input window and dictionary of last-timestep OHLC values.

        Args:
            idx (int): Index of the window to return.

        Returns:
            Tuple[numpy.ndarray, dict]: (inputs, target_dict) where `inputs` is
            an array of shape (window_size, 5) and `target_dict` maps 'O','H','L','C'
            to the corresponding float values from the final timestep.
        """
        start, end          = self._windows[idx]
        w:    pd.DataFrame  = self.df.iloc[start:end]
        arr: torch.tensor     = w[["open", "high", "low", "close", "volume"]].to_numpy(dtype=float)

        # inputs: full window, targets: last timestep's open,high,low,close
        inp: torch.tensor = arr.copy().astype(float)
        last: torch.tensor = arr[-1]

        out_dict = dict(zip("OHLC", last.tolist()))
        return inp, out_dict