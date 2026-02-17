import torch
import pandas as pd
from dataclasses import dataclass
from typing import Tuple
from .autoencoders import AutoEncoder
from .geometry import (
    transform_covariance,
    ambient_quadratic_variation_drift,
    curvature_drift_explicit,
    curvature_drift_explicit_full,
    curvature_drift_hessian_free,
    curvature_drift_hessian_free_full,
    regularized_metric_inverse,
)

# To add a new loss regularization, simply add a weight to the weight class, then modify TotalLoss
@dataclass
class LossWeights:
    tangent_bundle: float = 0.
    contractive: float = 0.
    normal_decoder_jacobian: float = 0.
    decoder_contraction: float = 0.
    diffeo: float = 0.
    curvature: float = 0.
    curvature_full: float = 0.
    drift: float = 0.

# Pointwise loss functions
def fro_norm_sq(matrix):
    return torch.linalg.matrix_norm(matrix, ord="fro")**2

def fro_distance_sq(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return fro_norm_sq(a-b)

def l2_loss(x_hat, x):
    return torch.linalg.vector_norm(x_hat - x, ord=2, dim=1) ** 2

def tangent_bundle_loss(phat, p):
    """Original O(D²) tangent bundle loss using full projection matrices."""
    return fro_distance_sq(phat, p) * 0.5


def tangent_bundle_loss_efficient(dphi: torch.Tensor, U_d: torch.Tensor) -> torch.Tensor:
    """
    Efficient O(Dd²) tangent bundle loss using the trace identity.

    Uses the identity: ½‖P̂ - P‖²_F = d - Tr(P̂ P)

    where P̂ = ∇φ g⁻¹ ∇φᵀ (model projection) and P = U_d U_dᵀ (empirical projection).

    By the cyclic property of trace:
        Tr(P̂ P) = Tr(∇φ g⁻¹ ∇φᵀ U_d U_dᵀ) = Tr(g⁻¹ ∇φᵀ U_d U_dᵀ ∇φ)

    Let C = ∇φᵀ U_d (d×d matrix), then:
        Tr(P̂ P) = Tr(g⁻¹ C Cᵀ)

    This reduces complexity from O(D²d) to O(Dd²).

    Args:
        dphi: Decoder Jacobian, shape (B, D, d)
        U_d: Top-d eigenvectors of empirical projection, shape (B, D, d)

    Returns:
        loss: Per-sample tangent bundle loss, shape (B,)
    """
    d = dphi.shape[-1]

    # Compute metric tensor g = ∇φᵀ ∇φ
    g = torch.bmm(dphi.transpose(-1, -2), dphi)  # (B, d, d)
    ginv = regularized_metric_inverse(g)  # (B, d, d)

    # C = ∇φᵀ @ U_d: (B, d, D) @ (B, D, d) -> (B, d, d)
    C = torch.bmm(dphi.transpose(-1, -2), U_d)

    # Tr(g⁻¹ C Cᵀ) = Tr(g⁻¹ @ C @ Cᵀ)
    # Compute C @ Cᵀ first
    CCT = torch.bmm(C, C.transpose(-1, -2))  # (B, d, d)

    # Tr(g⁻¹ @ CCT) = sum of elementwise g⁻¹ * CCTᵀ = sum of g⁻¹ * CCT (symmetric)
    trace = torch.sum(ginv * CCT, dim=(-1, -2))

    # Loss = d - Tr(P̂ P)
    return d - trace

def contraction_loss(jacobian: torch.Tensor) -> torch.Tensor:
    return fro_norm_sq(jacobian)

def normal_decoder_jacobian(normal_proj: torch.Tensor, decoder_jacobian: torch.Tensor):
    return fro_norm_sq(torch.bmm(normal_proj, decoder_jacobian))

def diffeomorphism_penalty(dpi: torch.Tensor, dphi: torch.Tensor):
    d = dpi.size(1)
    identity = torch.eye(d, device=dpi.device, dtype=dpi.dtype).unsqueeze(0)
    return fro_distance_sq(torch.bmm(dpi, dphi), identity)


# Individual losses/penalties
def empirical_l2_risk(x_hat, x):
    return torch.mean(l2_loss(x_hat, x))

def empirical_tangent_bundle_risk(phat, p):
    return torch.mean(tangent_bundle_loss(phat, p))


def empirical_tangent_bundle_risk_efficient(dphi: torch.Tensor, U_d: torch.Tensor) -> torch.Tensor:
    """Efficient mean tangent bundle risk using trace formula."""
    return torch.mean(tangent_bundle_loss_efficient(dphi, U_d))

def empirical_diffeo_penalty(dpi: torch.Tensor, dphi: torch.Tensor) -> torch.Tensor:
    return torch.mean(diffeomorphism_penalty(dpi, dphi))

def empirical_contraction_penalty(jacobian: torch.Tensor):
    return torch.mean(contraction_loss(jacobian))

def empirical_normal_decoder_jacobian(normal_proj, decoder_jacobian):
    return torch.mean(normal_decoder_jacobian(normal_proj, decoder_jacobian))

# Loss function to decide which to compute based off weights.
def autoencoder_loss(model: AutoEncoder,
                     targets: Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
                     loss_weights: LossWeights,
                     tangent_basis: torch.Tensor = None,
                     hessians: torch.Tensor = None,
                     local_cov_true: torch.Tensor = None):
    """
    Compute total autoencoder loss with optional regularization penalties.

    Args:
        model: AutoEncoder model
        targets: Tuple of (x, mu, cov, p) tensors
        loss_weights: Weight configuration for each penalty term
        tangent_basis: Optional (B, D, d) tensor of top-d eigenvectors of P.
                       If provided, uses efficient O(Dd²) tangent bundle loss
                       instead of O(D²d) version.
        hessians: Optional (B, D, d, d) tensor of true surface Hessians.
                  Required when curvature_full > 0 or drift > 0.
        local_cov_true: Optional (B, d, d) tensor of true local covariance.
                        Required when curvature_full > 0 or drift > 0.

    Returns:
        total_loss: Scalar loss value
    """
    x, mu, cov, p = targets

    if loss_weights.curvature_full > 0. or loss_weights.drift > 0.:
        if hessians is None or local_cov_true is None:
            raise ValueError(
                "curvature_full/drift > 0 requires both `hessians` and `local_cov_true` arguments"
            )

    D = p.size(2)
    normal_proj = torch.eye(D, device=p.device, dtype=p.dtype).unsqueeze(0) - p

    # Check if we should use efficient tangent bundle loss
    # Only beneficial when D > ~100 (overhead dominates for small D)
    use_efficient_tangent = (
        tangent_basis is not None
        and loss_weights.tangent_bundle > 0.0
        and D > 100  # Threshold where efficient formula beats standard
    )

    need_encoder_jacobian = loss_weights.contractive > 0. or loss_weights.diffeo > 0. or loss_weights.curvature > 0.
    # Need decoder Jacobian for efficient tangent loss too
    need_decoder_jacobian = (
        loss_weights.normal_decoder_jacobian > 0.
        or loss_weights.decoder_contraction > 0.
        or loss_weights.diffeo > 0.
        or loss_weights.curvature > 0.
        or loss_weights.curvature_full > 0.
        or loss_weights.drift > 0.
        or use_efficient_tangent
    )
    # Only need full projection matrix if NOT using efficient tangent loss
    need_orthogonal_projection = (loss_weights.tangent_bundle > 0. and not use_efficient_tangent) or loss_weights.curvature > 0.

    # Decide whether to use Hessian-free JVP for curvature computation
    # JVP is faster when D > 200 and d <= 3
    d = model.intrinsic_dim
    use_hessian_free_curvature = (
        loss_weights.curvature > 0.0
        and D > 200
        and d <= 3
    )
    use_hessian_free_curvature_full = (
        loss_weights.curvature_full > 0.0
        and D > 200
        and d <= 3
    )
    need_decoder_hessian = (
        (loss_weights.curvature > 0. and not use_hessian_free_curvature)
        or (loss_weights.curvature_full > 0. and not use_hessian_free_curvature_full)
    )

    # We always need to encode/decode
    z = model.encoder(x)
    xhat = model.decoder(z)

    # Preparing input without redundancies
    if need_encoder_jacobian:
        dpi = model.jacobian_encoder(x)
    if need_decoder_jacobian:
        dphi = model.jacobian_decoder(z)
    if need_orthogonal_projection:
        if need_decoder_jacobian:
            g = torch.bmm(dphi.mT, dphi)
            ginv = regularized_metric_inverse(g)
            phat = torch.bmm(dphi, torch.bmm(ginv, dphi.mT))
        else:
            phat = model.orthogonal_projection(z)
    # Shared computation for curvature losses
    # local_cov_z = pulled-back ambient covariance in z-coordinates
    if loss_weights.curvature > 0. or loss_weights.curvature_full > 0.:
        penrose = torch.linalg.pinv(dphi)
        local_cov_z = transform_covariance(cov, penrose)

    if loss_weights.curvature > 0.:
        normal_drift_true = torch.bmm(normal_proj, mu.unsqueeze(-1)).squeeze(-1)
        nhat = torch.eye(p.size(1), device=p.device, dtype=p.dtype).unsqueeze(0) - phat

        if use_hessian_free_curvature:
            normal_drift_model = curvature_drift_hessian_free(
                model.decoder, z, local_cov_z, nhat
            )
        else:
            d2phi = model.hessian_decoder(z)
            normal_drift_model = curvature_drift_explicit(d2phi, local_cov_z, nhat)

    if loss_weights.curvature_full > 0.:
        # True target: full Ito correction in ambient space from true local covariance
        ito_true = 0.5 * ambient_quadratic_variation_drift(local_cov_true, hessians)

        # Detach z so Kf gradients only reach the decoder, not the encoder.
        # This prevents the curvature objective from degrading reconstruction.
        z_det = z.detach()
        dphi_det = model.jacobian_decoder(z_det)
        penrose_det = torch.linalg.pinv(dphi_det)
        local_cov_z_det = transform_covariance(cov, penrose_det)

        if use_hessian_free_curvature_full:
            ito_model = curvature_drift_hessian_free_full(model.decoder, z_det, local_cov_z_det)
        else:
            d2phi_det = model.hessian_decoder(z_det)
            ito_model = curvature_drift_explicit_full(d2phi_det, local_cov_z_det)

    if loss_weights.drift > 0.:
        # Unified ambient drift matching: ||P̂·(mu - q_true) + q̂ - mu||²
        # Directly optimizes the full ambient drift round-trip through the model.
        # When P̂ ≈ P, reduces to Kf (||q̂ - q_true||²).
        # When q̂ ≈ q_true, penalizes tangent space misalignment.
        z_det = z.detach()
        dphi_det = model.jacobian_decoder(z_det)
        d2phi_det = model.hessian_decoder(z_det)
        g_det = torch.bmm(dphi_det.mT, dphi_det)
        ginv_det = regularized_metric_inverse(g_det)
        phat_det = torch.bmm(dphi_det, torch.bmm(ginv_det, dphi_det.mT))

        penrose_det = torch.linalg.pinv(dphi_det)
        local_cov_z_det = transform_covariance(cov, penrose_det)

        q_true = 0.5 * ambient_quadratic_variation_drift(local_cov_true, hessians)
        q_model = curvature_drift_explicit_full(d2phi_det, local_cov_z_det)

        tangential_true = mu - q_true
        mu_model = torch.bmm(phat_det, tangential_true.unsqueeze(-1)).squeeze(-1) + q_model

    # Accumulating losses
    total_loss = empirical_l2_risk(xhat, x)
    if loss_weights.tangent_bundle > 0.0:
        if use_efficient_tangent:
            # Use efficient O(Dd²) trace-based computation
            # Computes Tr(g⁻¹ ∇φᵀ U_d U_dᵀ ∇φ) without forming D×D matrices
            total_loss += loss_weights.tangent_bundle * empirical_tangent_bundle_risk_efficient(dphi, tangent_basis)
        else:
            # Use standard O(D²d) computation
            total_loss += loss_weights.tangent_bundle * empirical_tangent_bundle_risk(phat, p)
    if loss_weights.contractive > 0.0:
        total_loss += loss_weights.contractive * empirical_contraction_penalty(dpi)
    if loss_weights.normal_decoder_jacobian > 0.0:
        total_loss += loss_weights.normal_decoder_jacobian * empirical_normal_decoder_jacobian(normal_proj, dphi)
    if loss_weights.decoder_contraction > 0.0:
        total_loss += loss_weights.decoder_contraction * empirical_contraction_penalty(dphi)
    if loss_weights.diffeo > 0.0:
        total_loss += loss_weights.diffeo * empirical_diffeo_penalty(dpi, dphi)
    if loss_weights.curvature > 0.0:
        total_loss += loss_weights.curvature * empirical_l2_risk(normal_drift_model, normal_drift_true)
    if loss_weights.curvature_full > 0.0:
        total_loss += loss_weights.curvature_full * empirical_l2_risk(ito_model, ito_true)
    if loss_weights.drift > 0.0:
        total_loss += loss_weights.drift * empirical_l2_risk(mu_model, mu)
    return total_loss
