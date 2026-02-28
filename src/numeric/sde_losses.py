"""
SDE loss functions for the data-driven latent SDE pipeline.

Stage 2: tangential_drift_loss — trains drift_net only (frozen AE, detached z).
Stage 3: ambient_diffusion_loss — trains diffusion_net only (frozen AE, detached z).

Conventions:
- q = curvature_drift_explicit_full(d2phi, Sigma_z) is already halved (0.5 * Tr(Σ H)).
- Never use torch.linalg.pinv; use regularized_metric_inverse for stable pseudoinverse.
- Pre-project Lambda to tangent before pulling back to latent space.
- Symmetrize all computed symmetric matrices.
"""
from torch import Tensor

from .geometry import curvature_drift_explicit_full, regularized_metric_inverse


def tangential_drift_loss(
    decoder,
    drift_net,
    z: Tensor,
    v: Tensor,
    Lambda: Tensor,
    dphi: Tensor = None,
    d2phi: Tensor = None,
) -> Tensor:
    """
    Tangential drift matching loss (Stage 2 — trains drift_net only).

    L = ||P_hat (Dphi * b_z - (v - q))||^2

    Projects the full residual onto tangent space so the normal K residual
    is excluded. Uses P_hat on the residual (not pre-applied to each side)
    to avoid relying on P_hat being an exact projector (eps regularization).

    Args:
        decoder: Frozen decoder network (requires_grad=False on params).
        drift_net: DriftNet to train.
        z: Detached latent points, shape (B, d).
        v: Observed ambient drift (velocity), shape (B, D).
        Lambda: Observed ambient covariance, shape (B, D, D).
        dphi: Optional precomputed Jacobian, shape (B, D, d).
        d2phi: Optional precomputed Hessian, shape (B, D, d, d).

    Returns:
        Scalar loss value.
    """
    if dphi is None:
        dphi = decoder.jacobian_network(z)     # (B, D, d)
    if d2phi is None:
        d2phi = decoder.hessian_network(z)     # (B, D, d, d)

    g = dphi.mT @ dphi                    # (B, d, d)
    ginv = regularized_metric_inverse(g)   # (B, d, d)
    pinv = ginv @ dphi.mT                  # (B, d, D) — stable pseudoinverse

    P_hat = dphi @ pinv                    # (B, D, D)
    P_hat = 0.5 * (P_hat + P_hat.mT)      # symmetrize projector

    Lambda_tan = P_hat @ Lambda @ P_hat    # pre-project to tangent
    Lambda_tan = 0.5 * (Lambda_tan + Lambda_tan.mT)

    Sigma_z_obs = pinv @ Lambda_tan @ pinv.mT  # (B, d, d)
    Sigma_z_obs = 0.5 * (Sigma_z_obs + Sigma_z_obs.mT)

    q = curvature_drift_explicit_full(d2phi, Sigma_z_obs)  # already halved, (B, D)

    b_z = drift_net(z)                     # (B, d)
    dphi_bz = (dphi @ b_z.unsqueeze(-1)).squeeze(-1)  # Dphi * b_z, (B, D)

    residual = dphi_bz - (v - q)           # full ambient residual
    tan_res = (P_hat @ residual.unsqueeze(-1)).squeeze(-1)  # project to tangent

    D = tan_res.shape[-1]
    return (tan_res ** 2).sum(-1).mean() / D


def ambient_diffusion_loss(
    diffusion_net,
    z: Tensor,
    Lambda: Tensor,
    dphi: Tensor = None,
    decoder=None,
    v: Tensor = None,
    d2phi: Tensor = None,
    lambda_K: float = 0.0,
) -> Tensor:
    """
    Ambient covariance matching loss (Stage 3 — trains diffusion_net only).

    L = ||Dphi Sigma_z Dphi^T - Lambda||^2_F + lambda_K * ||(I-P_hat)(v - q)||^2

    When lambda_K=0 (default), only covariance matching is used.
    When lambda_K>0, adds K identity regularization that penalizes
    the normal component of (v - q), encouraging the learned diffusion
    to satisfy the geometric K identity.

    Args:
        diffusion_net: DiffusionNet to train.
        z: Detached latent points, shape (B, d).
        Lambda: Observed ambient covariance, shape (B, D, D).
        dphi: Precomputed Jacobian, shape (B, D, d). If None, computed from decoder.
        decoder: Decoder network (only needed if dphi is None).
        v: Observed ambient drift, shape (B, D). Required if lambda_K > 0.
        d2phi: Precomputed Hessian, shape (B, D, d, d). Required if lambda_K > 0.
        lambda_K: Weight for K identity regularization. Default 0 (off).

    Returns:
        Scalar loss value.
    """
    import torch

    if dphi is None:
        dphi = decoder.jacobian_network(z)     # (B, D, d)

    sigma = diffusion_net(z)               # (B, d, d)
    Sigma_z = sigma @ sigma.mT             # (B, d, d)
    Sigma_z = 0.5 * (Sigma_z + Sigma_z.mT)

    Lambda_pred = dphi @ Sigma_z @ dphi.mT  # (B, D, D)
    D = Lambda.shape[-1]
    cov_loss = ((Lambda_pred - Lambda) ** 2).sum((-1, -2)).mean() / (D * D)

    if lambda_K == 0.0:
        return cov_loss

    # K identity regularization: ||(I - P_hat)(v - q)||^2
    if d2phi is None:
        d2phi = decoder.hessian_network(z)

    g = dphi.mT @ dphi
    ginv = regularized_metric_inverse(g)
    P_hat = dphi @ ginv @ dphi.mT
    P_hat = 0.5 * (P_hat + P_hat.mT)

    q = curvature_drift_explicit_full(d2phi, Sigma_z)  # already halved

    D = dphi.shape[1]
    I_mat = torch.eye(D, device=z.device, dtype=z.dtype).unsqueeze(0)
    N_hat = I_mat - P_hat
    normal_res = (N_hat @ (v - q).unsqueeze(-1)).squeeze(-1)
    K_loss = (normal_res ** 2).sum(-1).mean() / D

    return cov_loss + lambda_K * K_loss
