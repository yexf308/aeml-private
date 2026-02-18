"""
SDE loss functions for the data-driven latent SDE pipeline.

Stage 2: tangential_drift_loss — trains drift_net only (frozen AE, detached z).
Stage 3: tangent_diffusion_loss — trains diffusion_net only (frozen AE, detached z).

Conventions:
- q = curvature_drift_explicit_full(d2phi, Sigma_z) is already halved (0.5 * Tr(Σ H)).
- Never use torch.linalg.pinv; use regularized_metric_inverse for stable pseudoinverse.
- Pre-project Lambda to tangent before pulling back to latent space.
- Symmetrize all computed symmetric matrices.
"""
import torch
from torch import Tensor

from .geometry import curvature_drift_explicit_full, regularized_metric_inverse


def tangential_drift_loss(
    decoder,
    drift_net,
    z: Tensor,
    v: Tensor,
    Lambda: Tensor,
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

    Returns:
        Scalar loss value.
    """
    dphi = decoder.jacobian_network(z)     # (B, D, d)
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

    return (tan_res ** 2).sum(-1).mean()


def tangent_diffusion_loss(
    decoder,
    diffusion_net,
    z: Tensor,
    v: Tensor,
    Lambda: Tensor,
    lambda_K: float = 0.1,
) -> Tensor:
    """
    Tangent-projected covariance matching + K regularization (Stage 3 — trains diffusion_net only).

    L = ||Lambda_pred - Lambda_tan||^2_F + lambda_K * ||(I - P_hat)(v - q)||^2

    Compares predicted ambient covariance against tangent-projected observed covariance,
    so normal covariance noise doesn't leak gradient. Includes K identity as regularization.

    Args:
        decoder: Frozen decoder network (requires_grad=False on params).
        diffusion_net: DiffusionNet to train.
        z: Detached latent points, shape (B, d).
        v: Observed ambient drift (velocity), shape (B, D).
        Lambda: Observed ambient covariance, shape (B, D, D).
        lambda_K: Weight for K identity regularization.

    Returns:
        Scalar loss value.
    """
    dphi = decoder.jacobian_network(z)     # (B, D, d)
    d2phi = decoder.hessian_network(z)     # (B, D, d, d)

    sigma = diffusion_net(z)               # (B, d, d)
    Sigma_z = sigma @ sigma.mT             # (B, d, d)
    Sigma_z = 0.5 * (Sigma_z + Sigma_z.mT)

    # Tangent-projected covariance matching
    g = dphi.mT @ dphi                    # (B, d, d)
    ginv = regularized_metric_inverse(g)   # (B, d, d)

    P_hat = dphi @ ginv @ dphi.mT         # (B, D, D)
    P_hat = 0.5 * (P_hat + P_hat.mT)

    Lambda_tan = P_hat @ Lambda @ P_hat    # pre-project target
    Lambda_tan = 0.5 * (Lambda_tan + Lambda_tan.mT)

    Lambda_pred = dphi @ Sigma_z @ dphi.mT  # (B, D, D)
    cov_loss = ((Lambda_pred - Lambda_tan) ** 2).sum((-1, -2)).mean()

    # K identity regularization
    D = dphi.shape[1]
    I_mat = torch.eye(D, device=z.device, dtype=z.dtype).unsqueeze(0)
    N_hat = I_mat - P_hat

    q = curvature_drift_explicit_full(d2phi, Sigma_z)  # already halved
    normal_residual = (N_hat @ (v - q).unsqueeze(-1)).squeeze(-1)
    K_loss = (normal_residual ** 2).sum(-1).mean()

    return cov_loss + lambda_K * K_loss
