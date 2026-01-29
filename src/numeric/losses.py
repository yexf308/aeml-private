import torch
import pandas as pd
from dataclasses import dataclass
from typing import Tuple
from .autoencoders import AutoEncoder
from .geometry import transform_covariance, ambient_quadratic_variation_drift

# To add a new loss regularization, simply add a weight to the weight class, then modify TotalLoss
@dataclass
class LossWeights:
    tangent_bundle: float = 0.
    contractive: float = 0.
    normal_decoder_jacobian: float = 0.
    decoder_contraction: float = 0.
    diffeo: float = 0.
    curvature: float = 0.

# Pointwise loss functions
def fro_norm_sq(matrix):
    return torch.linalg.matrix_norm(matrix, ord="fro")**2

def fro_distance_sq(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return fro_norm_sq(a-b)

def l2_loss(x_hat, x):
    return torch.linalg.vector_norm(x_hat - x, ord=2, dim=1) ** 2

def tangent_bundle_loss(phat, p):
    return fro_distance_sq(phat, p) * 0.5

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

def empirical_diffeo_penalty(dpi: torch.Tensor, dphi: torch.Tensor) -> torch.Tensor:
    return torch.mean(diffeomorphism_penalty(dpi, dphi))

def empirical_contraction_penalty(jacobian: torch.Tensor):
    return torch.mean(contraction_loss(jacobian))

def empirical_normal_decoder_jacobian(normal_proj, decoder_jacobian):
    return torch.mean(normal_decoder_jacobian(normal_proj, decoder_jacobian))

# Loss function to decide which to compute based off weights. 
def autoencoder_loss(model: AutoEncoder, 
                     targets: Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor], 
                     loss_weights: LossWeights):
    x, mu, cov, p = targets
    D = p.size(2)
    normal_proj = torch.eye(D, device=p.device, dtype=p.dtype).unsqueeze(0) - p

    
    need_encoder_jacobian = loss_weights.contractive > 0. or loss_weights.diffeo > 0. or loss_weights.curvature > 0.
    need_decoder_jacobian = loss_weights.normal_decoder_jacobian > 0. or loss_weights.decoder_contraction > 0. or loss_weights.diffeo > 0. or loss_weights.curvature > 0.
    need_orthogonal_projection = loss_weights.tangent_bundle > 0. or loss_weights.curvature > 0.
    need_decoder_hessian = loss_weights.curvature > 0.
    
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
            ginv = torch.linalg.inv(g)
            phat = torch.bmm(dphi, torch.bmm(ginv, dphi.mT))
        else:
            phat = model.orthogonal_projection(z)
    if need_decoder_hessian:
        normal_drift_true = torch.bmm(normal_proj, mu.unsqueeze(-1)).squeeze(-1)
        # compute nhat from phat via nhat=I-phat just like we did to compute normal_proj from p
        nhat = torch.eye(p.size(1), device=p.device, dtype=p.dtype).unsqueeze(0) - phat
        d2phi = model.hessian_decoder(z)
        # TODO: either use Dpi or use g^{-1} Dphi^T penrose via an option
        penrose = torch.linalg.pinv(dphi)
        local_cov = transform_covariance(cov, penrose)
        q = ambient_quadratic_variation_drift(local_cov, d2phi)
        normal_drift_model = 0.5 * torch.bmm(nhat, q.unsqueeze(-1)).squeeze(-1)
        

    
    # Accumulating losses
    total_loss = empirical_l2_risk(xhat, x)
    if loss_weights.tangent_bundle > 0.0:
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
    return total_loss
