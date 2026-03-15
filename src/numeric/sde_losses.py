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
    q_override: Tensor = None,
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
        q_override: Optional precomputed curvature correction, shape (B, D).
            If provided, used instead of computing q from d2phi.

    Returns:
        Scalar loss value.
    """
    if dphi is None:
        dphi = decoder.jacobian_network(z)     # (B, D, d)

    g = dphi.mT @ dphi                    # (B, d, d)
    ginv = regularized_metric_inverse(g)   # (B, d, d)
    pinv = ginv @ dphi.mT                  # (B, d, D) — stable pseudoinverse

    P_hat = dphi @ pinv                    # (B, D, D)
    P_hat = 0.5 * (P_hat + P_hat.mT)      # symmetrize projector

    if q_override is not None:
        q = q_override
    else:
        if d2phi is None:
            d2phi = decoder.hessian_network(z)     # (B, D, d, d)

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


def encoder_pullback_drift_loss(
    encoder,
    decoder,
    drift_net,
    z: Tensor,
    x: Tensor,
    v: Tensor,
    Lambda: Tensor,
    dphi: Tensor = None,
    d2phi: Tensor = None,
    dpi: Tensor = None,
) -> Tensor:
    """
    Encoder-pullback drift loss using D²(π∘φ)=0 identity (Stage 2).

    Uses the identity D²(π∘φ) = 0 to avoid the encoder Hessian:
        b_z_target = Dπ · (v - q_φ)
    where q_φ = ½ Σ_z : d²φ is the decoder-side Itô correction and
    Dπ is the encoder Jacobian (not the decoder pseudoinverse).

    NOTE: The identity D²(π∘φ)=0 only holds when π∘φ = id exactly.
    For learned AEs, ||D²(π∘φ)|| ≈ 0.9-1.3 (first-order but not
    second-order inverse consistency), so this is a poor approximation.
    Prefer precompute_enc_pull_target + train_stage2_regression.

    The loss is metric-weighted MSE in latent space:
        L = (b_z - b_target)^T g (b_z - b_target) / D

    Args:
        encoder: Frozen encoder network.
        decoder: Frozen decoder network.
        drift_net: DriftNet to train.
        z: Detached latent points, shape (B, d).
        x: Ambient samples, shape (B, D).
        v: Observed ambient drift, shape (B, D).
        Lambda: Observed ambient covariance, shape (B, D, D).
        dphi: Optional precomputed decoder Jacobian, shape (B, D, d).
        d2phi: Optional precomputed decoder Hessian, shape (B, D, d, d).
        dpi: Optional precomputed encoder Jacobian, shape (B, d, D).

    Returns:
        Scalar loss value.
    """
    import torch

    if dphi is None:
        dphi = decoder.jacobian_network(z)
    if d2phi is None:
        d2phi = decoder.hessian_network(z)
    if dpi is None:
        dpi = torch.func.vmap(torch.func.jacrev(encoder))(x).detach()

    g = dphi.mT @ dphi                        # (B, d, d)
    ginv = regularized_metric_inverse(g)       # (B, d, d)
    pinv = ginv @ dphi.mT                      # (B, d, D)

    # Pull back Lambda to latent via pseudoinverse (required by D²(π∘φ)=0 identity)
    Sigma_z = pinv @ Lambda @ pinv.mT          # (B, d, d)
    Sigma_z = 0.5 * (Sigma_z + Sigma_z.mT)

    # Decoder-side Itô correction: q_φ = ½ Σ_z : d²φ
    q_phi = curvature_drift_explicit_full(d2phi, Sigma_z)  # (B, D), already halved

    # Target: b_z = Dπ · (v - q_φ)
    b_z_target = (dpi @ (v - q_phi).unsqueeze(-1)).squeeze(-1)  # (B, d)

    b_z = drift_net(z)                         # (B, d)
    residual = b_z - b_z_target                # (B, d)

    # Metric-weighted MSE: r^T g r / D
    D = dphi.shape[1]
    cost = (residual.unsqueeze(-2) @ g @ residual.unsqueeze(-1)).squeeze(-1).squeeze(-1)
    return cost.mean() / D


def latent_drift_regression_loss(
    drift_net,
    z: Tensor,
    b_z_target: Tensor,
    g: Tensor = None,
    ambient_dim: int = 1,
) -> Tensor:
    """
    Metric-weighted latent drift regression loss.

    L = mean_i [ (b_z - b_target)^T g (b_z - b_target) ] / D

    Normalised by ambient dimension D so that lambda_smooth has consistent
    effective weight relative to drift_smoothness_loss (which also divides by D).

    When g is None, uses plain Euclidean MSE (equivalent to g = I).

    Args:
        drift_net: DriftNet to train.
        z: Detached latent points, shape (B, d).
        b_z_target: Precomputed target latent drift, shape (B, d).
        g: Optional metric tensor, shape (B, d, d). If None, Euclidean MSE.
        ambient_dim: Ambient dimension D for normalisation (default 1 = no normalisation).

    Returns:
        Scalar loss value.
    """
    b_z = drift_net(z)                     # (B, d)
    residual = b_z - b_z_target            # (B, d)

    if g is not None:
        # Metric-weighted: r^T g r
        cost = (residual.unsqueeze(-2) @ g @ residual.unsqueeze(-1)).squeeze(-1).squeeze(-1)
    else:
        cost = (residual ** 2).sum(-1)

    return cost.mean() / ambient_dim


def drift_smoothness_loss(
    decoder,
    drift_net,
    z_aug: Tensor,
    use_metric: bool = True,
) -> Tensor:
    """
    Latent Jacobian penalty for drift_net smoothness.

    When use_metric=True (default):
        L = E_z[ tr(J_bz^T g J_bz g^{-1}) / D ]
    When use_metric=False (Euclidean ablation):
        L = E_z[ ||J_bz||_F^2 / D ]

    where J_bz = d(b_z)/dz (d x d), g = Dphi^T Dphi is the metric from
    the frozen decoder, and D is the ambient dimension.

    Normalised by D (not d) so that lambda_smooth has consistent effective
    weight relative to tangential_drift_loss (which also divides by D).

    Args:
        decoder: Frozen decoder (requires_grad=False on params).
        drift_net: DriftNet being trained.
        z_aug: Augmented latent points, shape (B, d).
        use_metric: If True, use metric-weighted penalty; if False, plain Frobenius.

    Returns:
        Scalar loss (non-negative).
    """
    import torch

    # Exact Jacobian of drift_net: J_bz[i,j] = d(b_z)_i / dz_j
    J_bz = torch.func.vmap(torch.func.jacrev(drift_net))(z_aug)  # (B, d, d)

    if use_metric:
        # Metric tensor from frozen decoder
        with torch.no_grad():
            dphi = decoder.jacobian_network(z_aug)          # (B, D, d)
            g = dphi.mT @ dphi                              # (B, d, d)
            ginv = regularized_metric_inverse(g)             # (B, d, d)

        D = dphi.shape[1]  # ambient dimension

        # tr(J_bz^T g J_bz g^{-1}) / D
        M = J_bz.mT @ g @ J_bz @ ginv                       # (B, d, d)
        return torch.diagonal(M, dim1=-2, dim2=-1).sum(-1).mean() / D
    else:
        # Plain Euclidean: ||J_bz||_F^2 / D
        # Need D from decoder; fall back to d if decoder unavailable
        with torch.no_grad():
            dphi = decoder.jacobian_network(z_aug)
        D = dphi.shape[1]
        return (J_bz ** 2).sum((-1, -2)).mean() / D


def diffusion_smoothness_loss(
    decoder,
    diffusion_net,
    z_aug: Tensor,
) -> Tensor:
    """
    Jacobian penalty on the latent covariance Sigma(z) = sigma sigma^T.

    Smooths the *identifiable* covariance tensor (not the raw Cholesky factor)
    to avoid gauge dependence.  Uses vech (unique entries of the symmetric 2x2)
    and normalises by D^2 to match ambient_diffusion_loss scaling.

        L = E_z[ ||J_{vech(Sigma)}||_F^2 ] / D^2

    Args:
        decoder: Frozen decoder (for D extraction).
        diffusion_net: DiffusionNet being trained.
        z_aug: Augmented latent points, shape (B, d).

    Returns:
        Scalar loss (non-negative).
    """
    import torch

    # Use functional_call to avoid in-place index_put_ in DiffusionNet.forward
    # which is incompatible with vmap.  Instead, directly call the inner MLP
    # and reconstruct Σ = σ σ^T functionally.
    d = z_aug.shape[-1]
    tril_rows = torch.tril_indices(d, d, device=z_aug.device)[0]
    tril_cols = torch.tril_indices(d, d, device=z_aug.device)[1]

    params = dict(diffusion_net.named_parameters())
    buffers = dict(diffusion_net.named_buffers())

    def _vech_cov(z_single):
        """z -> vech(Sigma) for a single point, purely functional."""
        raw = torch.func.functional_call(
            diffusion_net.net, {k.removeprefix("net."): v for k, v in {**params, **buffers}.items() if k.startswith("net.")},
            (z_single.unsqueeze(0),),
        ).squeeze(0)  # (n_tril,)
        # Build lower-triangular sigma without in-place ops
        sigma = torch.zeros(d, d, device=z_single.device, dtype=z_single.dtype)
        sigma = sigma.index_put((tril_rows, tril_cols), raw)  # out-of-place
        Sigma = sigma @ sigma.T
        # vech
        return Sigma[tril_rows, tril_cols]

    # Jacobian of vech(Sigma) w.r.t. z: shape (B, n_vech, d)
    J = torch.func.vmap(torch.func.jacrev(_vech_cov))(z_aug)

    with torch.no_grad():
        dphi = decoder.jacobian_network(z_aug)
    D = dphi.shape[1]

    return (J ** 2).sum((-1, -2)).mean() / (D * D)


def latent_diffusion_loss(
    diffusion_net,
    z: Tensor,
    Lambda: Tensor,
    dphi: Tensor,
) -> Tensor:
    """
    Latent-coordinate covariance matching loss (Stage 3 alternative).

    Instead of pushing Sigma_z up to ambient and matching D×D matrices,
    pull the observed Lambda down to latent and match d×d matrices:

        Sigma_z_obs = g^{-1} Dphi^T P Lambda P Dphi g^{-1}
        L = ||sigma sigma^T - Sigma_z_obs||^2_F

    This is rank-aware: the loss operates on d×d (e.g. 2×2) regardless of D,
    eliminating the normal-direction chart error that inflates the ambient loss
    at high D.

    Args:
        diffusion_net: DiffusionNet to train.
        z: Detached latent points, shape (B, d).
        Lambda: Observed ambient covariance, shape (B, D, D).
        dphi: Precomputed Jacobian, shape (B, D, d).

    Returns:
        Scalar loss value.
    """
    import torch

    g = dphi.mT @ dphi                              # (B, d, d)
    ginv = regularized_metric_inverse(g)             # (B, d, d)
    pinv = ginv @ dphi.mT                            # (B, d, D)

    P_hat = dphi @ pinv                              # (B, D, D)
    P_hat = 0.5 * (P_hat + P_hat.mT)

    Lambda_tan = P_hat @ Lambda @ P_hat
    Lambda_tan = 0.5 * (Lambda_tan + Lambda_tan.mT)

    Sigma_z_obs = pinv @ Lambda_tan @ pinv.mT        # (B, d, d)
    Sigma_z_obs = 0.5 * (Sigma_z_obs + Sigma_z_obs.mT)

    sigma = diffusion_net(z)                          # (B, d, d)
    Sigma_z = sigma @ sigma.mT
    Sigma_z = 0.5 * (Sigma_z + Sigma_z.mT)

    return ((Sigma_z - Sigma_z_obs) ** 2).sum((-1, -2)).mean()


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
