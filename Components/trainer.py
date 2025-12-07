"""Training loop utilities for MarketTransformer.

This module exposes `train_model`, a convenience function that constructs a
model, optimizer and runs a simple training and optional validation loop.
"""

from typing import List, Optional
from .dataset import OHLCVWindowDataset
from .MarketTransformer import MarketTransformer
from .utils import build_tensor_dataloader
from loss import total_loss
from torch.utils.data import DataLoader

import torch


def train_model(datasets_train: List[OHLCVWindowDataset], datasets_val: List[OHLCVWindowDataset], seq_len: int, epochs: int = 5, batch_size: int = 64, device: Optional[torch.device] = None):
    """Train a MarketTransformer on provided windowed datasets.

    Args:
        datasets_train: List of `OHLCVWindowDataset` instances used for training.
        datasets_val: Optional list of datasets used for validation.
        seq_len: Expected sequence length for the model and training windows.
        epochs: Number of training epochs.
        batch_size: Batch size for DataLoader.
        device: Torch device to use; if None the function selects CUDA if available.

    Returns:
        Tuple `(model, history)` where `model` is the trained `MarketTransformer`
        instance and `history` is a dict of training/validation metrics.
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    train_dl: DataLoader = build_tensor_dataloader(datasets_train, batch_size=batch_size)
    val_dl: Optional[DataLoader] = None
    if datasets_val:
        val_dl = build_tensor_dataloader(datasets_val, batch_size=batch_size)

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
            grad_norm = torch.sqrt(grad_norm)

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
        history['gate_mean'].append(torch.mean(gate_vals) if gate_vals else 0.0)
        history['gate_std'].append(torch.std(gate_vals) if gate_vals else 0.0)
        history['gate_entropy'].append(torch.mean(gate_entropies) if gate_entropies else 0.0)

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
            val_mae_close = float(torch.mean(torch.abs(torch.array(clos_resid)))) if len(clos_resid) else None
            history['val_loss'].append(val_loss); history['val_mae_close'].append(val_mae_close)

        print(f"Epoch {ep}/{epochs} train_loss={train_avg:.6f} val_loss={float(val_loss):.6f} gate_mean={history['gate_mean'][-1]:.4f} gate_std={history['gate_std'][-1]:.4f} gate_entropy={history['gate_entropy'][-1]:.4f} val_mae_close={val_mae_close}")

    return model, history