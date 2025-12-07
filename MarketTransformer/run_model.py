"""
MarketTransformer run script module.
This module provides a simple command-line entry point to train, load, and run
inference with a MarketTransformer model on market CSV datasets. It encapsulates
dataset construction, model training (or loading), batching for inference, and
visual diagnostic plotting.
Primary components and behavior
- DataConfig:
    A lightweight configuration container defining defaults used by the script:
      - window  (int): input sequence length (default 256)
      - target  (int): prediction horizon / target length (default 32)
      - stride  (int): sliding-window step when creating sequences (default 16)
      - batch   (int): default batch size for training/inference (default 64)
      - max_windows (int): maximum number of windows to include in a dataloader (default 5000)
- main(path=None):
    Orchestrates the end-to-end workflow:
      1. Detects device (CUDA if available, otherwise CPU).
      2. Collects CSV file paths from ./data/PSE/ (only files ending with .csv).
      3. Builds training and validation datasets using build_train_val_datasets()
         with DataConfig.window, DataConfig.target, DataConfig.stride and a
         fixed split_date of "2024-01-01".
      4. If no datasets are found, exits early.
      5. If path is not provided (or None), trains a MarketTransformer model via
         train_model(..., seq_len=DataConfig.window, epochs=5, batch_size=DataConfig.batch),
         saves the trained model to "./MarketTransformer/MarketTransformer.pth", and
         captures training history for diagnostics.
      6. If path is provided, loads the model from the supplied path.
      7. If a validation dataset exists, creates a DataLoader with
         build_tensor_dataloader(..., batch_size=DataConfig.batch, max_windows=DataConfig.max_windows),
         runs inference with infer_on_dataloader(model, val_dl, device), and
         visualizes results with visualize_inference and visualize_training_diagnostics.
      8. Prints a brief numeric summary of key inference columns ('C_true', 'C_pred', 'r_gate').
Parameters
- path (str|None): Optional path to a saved MarketTransformer model. If provided,
  the script will load the model from this path and skip training. If None, the
  script trains a new model and saves it to the default path.
Side effects and file I/O
- Reads CSV files from ./data/PSE/
- When training, writes model to ./MarketTransformer/MarketTransformer.pth
- Calls plotting helpers to produce visual diagnostic outputs (these may write
  files or open interactive windows depending on implementation).
Return value
- None. All work is performed for side effects (training, saving, plotting, printing).
Usage examples
- Train a new model and run inference on validation data:
    main()
- Load an existing model and run inference:
Notes and assumptions
- Expects a directory ./data/PSE/ populated with CSV market data files.
- build_train_val_datasets, build_tensor_dataloader, train_model, infer_on_dataloader,
  visualize_inference, and visualize_training_diagnostics are provided by the
  Components package and must be importable.
- Uses a hard-coded split_date "2024-01-01" for train/validation splitting.
- Default training uses 5 epochs; adjust train_model call if different behavior
  is required (this script calls train_model with fixed args).
- The script prefers GPU if torch.cuda.is_available() returns True.
Typical entry point
- The module is executable as a script; calling the file directly runs:
"""

from Components.MarketTransformer import MarketTransformer
from Components.utils import build_train_val_datasets, build_tensor_dataloader, infer_on_dataloader
from Components.visuals import visualize_inference, visualize_training_diagnostics
from Components.trainer import train_model

from torch.utils.data import DataLoader
from pandas import DataFrame

import torch
import os



class DataConfig:
    
    window  : int = 256
    target  : int = 32
    stride  : int = 16
    batch   : int = 64
    
    max_windows : int = 5000


def main(path = None):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model   : MarketTransformer
    history : dict
    
    fp      : str       = "./data/PSE/"
    paths   : list[str] = [fp+f for f in os.listdir(fp) if f.endswith(".csv")]

    train_ds, val_ds = build_train_val_datasets( 
        paths, 
        window_size = DataConfig.window, 
        target_size = DataConfig.target, 
        stride      = DataConfig.stride, 
        split_date  = "2024-01-01"
    )

    print(f"Train datasets: {len(train_ds)}; Val datasets: {len(val_ds)}")
    if len(train_ds) == 0 : return print("No training datasets found; exiting")
    
    if not path: 
        model, history = train_model(train_ds, val_ds, seq_len=DataConfig.window, epochs=15, batch_size=DataConfig.batch)
        torch.save(model, "./MarketTransformer/MarketTransformer.pth")
        print("Training complete")
    
    else : model = torch.load(path, weights_only=False)

    
    # run inference on validation set if available
    if not val_ds: return
    
    val_dl : DataLoader = build_tensor_dataloader(
        val_ds, 
        batch_size  = DataConfig.batch, 
        max_windows = DataConfig.max_windows
    )
    df_infer : DataFrame = infer_on_dataloader(model, val_dl, device)
    visualize_inference(df_infer); visualize_training_diagnostics(history)
    
    # quick numeric diagnostics
    print(df_infer.describe().loc[['mean','std']][['C_true','C_pred','r_gate']])
    


# if __name__ == "__main__" : main(path="./MarketTransformer/MarketTransformer.pth")
if __name__ == "__main__" : main()