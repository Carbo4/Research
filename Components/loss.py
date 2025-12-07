from typing import Dict, Optional
from .utils import _softplus_pos

import torch


"""Loss functions used for training MarketTransformer models.

This module implements negative log-likelihood objectives for Gaussian
and Half-Normal residual models, hinge penalties to enforce high/low
consistency with open/close baselines, and helper compound losses used
in the training loop.
"""


def gaussian_nll(y: torch.Tensor, mu: torch.Tensor, log_scale: torch.Tensor, reduction: str = 'none', eps: float = 1e-6) -> torch.Tensor:
    """Gaussian negative log-likelihood.

    Args:
        y: Observations.
        mu: Predicted mean.
        log_scale: Pre-activation log-scale values (softplus applied internally).
        reduction: One of ['none','mean','sum'] to reduce the output.
        eps: Small additive constant to keep scales positive.

    Returns:
        Tensor of NLL values (reduced according to `reduction`).
    """
    scale: torch.Tensor = _softplus_pos(log_scale, eps=eps)
    dist = torch.distributions.Normal(loc=mu, scale=scale)
    nll: torch.Tensor = -dist.log_prob(y)
    if reduction == 'mean':
        return nll.mean()
    if reduction == 'sum':
        return nll.sum()
    return nll


def half_normal_nll(residual: torch.Tensor, log_scale: torch.Tensor, reduction: str = 'none', eps: float = 1e-8) -> torch.Tensor:
    """Negative log-likelihood for a half-normal distributed residual.

    The function assumes residuals representing violations that should be
    non-negative; values are clamped to zero before computing the NLL.
    """
    r_clamped: torch.Tensor = residual.clamp(min=0.0)
    scale:     torch.Tensor = _softplus_pos(log_scale, eps=eps)
    term:      torch.Tensor = (r_clamped.pow(2)) / (2.0 * scale.pow(2))
    nll:       torch.Tensor = torch.log(scale) + term
    if reduction == 'mean':
        return nll.mean()
    if reduction == 'sum':
        return nll.sum()
    return nll


def hinge_penalties(o_pred: torch.Tensor, c_pred: torch.Tensor, h_true: torch.Tensor, l_true: torch.Tensor, lambda_h: float = 1.0, lambda_l: float = 1.0, reduction: str = 'none') -> torch.Tensor:
    """Quadratic hinge penalties for violations of H/L bounds.

    If predicted open/close produce a baseline interval, this penalty
    penalizes when the true high is below the predicted interval or the true
    low is above it. Penalties are squared and scaled by `lambda_h`/`lambda_l`.
    """
    base_max:   torch.Tensor = torch.max(o_pred, c_pred)
    base_min:   torch.Tensor = torch.min(o_pred, c_pred)
    viol_h:     torch.Tensor = torch.relu(base_max - h_true)
    viol_l:     torch.Tensor = torch.relu(l_true - base_min)
    p_h:        torch.Tensor = lambda_h * viol_h.pow(2)
    p_l:        torch.Tensor = lambda_l * viol_l.pow(2)
    pen:        torch.Tensor = p_h + p_l
    if reduction == 'mean':
        return pen.mean()
    if reduction == 'sum':
        return pen.sum()
    return pen


def stage1_loss_open_close(o_true: torch.Tensor, o_pred: torch.Tensor, s_o: torch.Tensor, c_true: torch.Tensor, c_pred: torch.Tensor, s_c: torch.Tensor, reduction: str = 'mean') -> torch.Tensor:
    """Stage-1 loss combining Gaussian NLL for open and close.

    The stage 1 objective models open and close using independent Gaussian
    likelihoods and returns their sum.
    """
    nll_o: torch.Tensor = gaussian_nll(o_true, o_pred, s_o, reduction='none')
    nll_c: torch.Tensor = gaussian_nll(c_true, c_pred, s_c, reduction='none')
    nll:   torch.Tensor = nll_o + nll_c
    if reduction == 'mean':
        return nll.mean()
    if reduction == 'sum':
        return nll.sum()
    return nll


def stage2_loss_high_low(h_true: torch.Tensor, l_true: torch.Tensor, base_o: torch.Tensor, base_c: torch.Tensor, use_pred_base: bool, o_pred: torch.Tensor, c_pred: torch.Tensor, log_scale_h: torch.Tensor, log_scale_l: torch.Tensor, option: str = 'half_normal', lambda_h: float = 1.0, lambda_l: float = 1.0, reduction: str = 'mean') -> torch.Tensor:
    """Stage-2 objective enforcing high/low consistency and modeling residuals.

    Depending on `use_pred_base` the baseline interval is either computed from
    predicted O/C or from provided base_o/base_c. Residuals for H and L are
    modeled with a half-normal NLL by default and combined with hinge penalties.
    """
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
    if reduction == 'mean':
        return total.mean()
    if reduction == 'sum':
        return total.sum()
    return total


def total_loss(preds: Dict[str, torch.Tensor], targets: Dict[str, torch.Tensor], config: Optional[Dict] = None) -> torch.Tensor:
    """High-level loss combining stage-1 and stage-2 terms.

    Args:
        preds: Dictionary of predicted tensors required by the stages (mu_w, s_o, s_c, feat_log_scales, trend_coefs).
        targets: Dictionary with keys 'O','C','H','L'.
        config: Optional configuration mapping for stage2 options and penalties.

    Returns:
        Scalar tensor representing the mean loss over the batch.
    """
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