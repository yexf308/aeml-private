"""
High-dimensional manifold embeddings via Fourier augmentation.

Given a base surface f: R² → R (e.g. paraboloid), construct:
  φ(u,v) = (u, v, f(u,v), cos(ω₁·z)/√K, sin(ω₁·z)/√K, ..., cos(ωK·z)/√K, sin(ωK·z)/√K)

where z = (u,v), ω_k ~ N(0,I) with fixed seed, K = (D-3)/2, D = 3 + 2K.

Properties:
  - Metric: g(z) = g_base(z) + (1/K) Σ_k [sin²/cos² terms] · ωₖωₖᵀ  (non-constant)
  - Itô correction: each Fourier component has H_k = -(cos/sin)(ωₖ·z)/√K · ωₖωₖᵀ
  - Bypass SymPy: all geometry computed numerically via torch
"""

import torch
import numpy as np
from dataclasses import dataclass
from typing import Callable

from src.numeric.datasets import DatasetBatch
from src.numeric.geometry import (
    regularized_metric_inverse,
    ambient_quadratic_variation_drift,
)


# Numeric base surface functions: f(u, v) -> z_scalar (batched)
def paraboloid_f(u, v):
    return u ** 2 + v ** 2


def hyperbolic_paraboloid_f(u, v):
    return u ** 2 - v ** 2


def sinusoidal_f(u, v):
    return torch.sin(u + v)


BASE_SURFACES = {
    "paraboloid": paraboloid_f,
    "hyperbolic_paraboloid": hyperbolic_paraboloid_f,
    "sinusoidal": sinusoidal_f,
}


@dataclass
class FourierAugmentedSurface:
    """A 2D surface embedded in R^D via Fourier augmentation."""

    base_name: str  # key into BASE_SURFACES
    D: int  # ambient dimension (must be odd >= 3)
    omega_seed: int = 42  # seed for random frequencies

    def __post_init__(self):
        assert self.D >= 3 and (self.D - 3) % 2 == 0, (
            f"D must be odd >= 3, got {self.D}"
        )
        self.K = (self.D - 3) // 2  # number of Fourier pairs
        self.base_f = BASE_SURFACES[self.base_name]
        # Generate fixed random frequencies ω_k ~ N(0, I), shape (K, 2)
        rng = np.random.RandomState(self.omega_seed)
        self.omegas = torch.tensor(rng.randn(self.K, 2), dtype=torch.float32)

    def chart(self, uv: torch.Tensor) -> torch.Tensor:
        """φ(u,v) -> R^D. Input: (B, 2), output: (B, D)."""
        device = uv.device
        omegas = self.omegas.to(device)
        u, v = uv[:, 0], uv[:, 1]
        base_z = self.base_f(u, v)  # (B,)

        parts = [u.unsqueeze(-1), v.unsqueeze(-1), base_z.unsqueeze(-1)]

        if self.K > 0:
            omega_z = uv @ omegas.T  # (B, K)
            scale = 1.0 / np.sqrt(self.K)
            parts.append(torch.cos(omega_z) * scale)  # (B, K)
            parts.append(torch.sin(omega_z) * scale)  # (B, K)

        return torch.cat(parts, dim=-1)  # (B, D)

    def jacobian(self, uv: torch.Tensor) -> torch.Tensor:
        """∂φ/∂z, shape (B, D, 2). Analytically computed."""
        device = uv.device
        omegas = self.omegas.to(device)
        B = uv.shape[0]

        jac = torch.zeros(B, self.D, 2, device=device)
        jac[:, 0, 0] = 1.0  # du/du
        jac[:, 1, 1] = 1.0  # dv/dv

        # Base surface partials via autograd
        uv_ad = uv.detach().requires_grad_(True)
        f_val = self.base_f(uv_ad[:, 0], uv_ad[:, 1])
        grad_f = torch.autograd.grad(f_val.sum(), uv_ad, create_graph=False)[0]
        jac[:, 2, :] = grad_f

        if self.K > 0:
            omega_z = uv @ omegas.T  # (B, K)
            scale = 1.0 / np.sqrt(self.K)
            sin_oz = torch.sin(omega_z)
            cos_oz = torch.cos(omega_z)
            # d(cos(ω·z))/dz = -sin(ω·z) · ω
            jac[:, 3 : 3 + self.K, :] = (
                -sin_oz.unsqueeze(-1) * omegas.unsqueeze(0) * scale
            )
            # d(sin(ω·z))/dz = cos(ω·z) · ω
            jac[:, 3 + self.K :, :] = (
                cos_oz.unsqueeze(-1) * omegas.unsqueeze(0) * scale
            )

        return jac

    def hessians(self, uv: torch.Tensor) -> torch.Tensor:
        """∂²φ_i/∂z_j∂z_k, shape (B, D, 2, 2). Analytically computed."""
        device = uv.device
        omegas = self.omegas.to(device)
        B = uv.shape[0]

        H = torch.zeros(B, self.D, 2, 2, device=device)

        # Base surface Hessian via autograd (double backward)
        uv_ad = uv.detach().requires_grad_(True)
        f_val = self.base_f(uv_ad[:, 0], uv_ad[:, 1])
        grad_f = torch.autograd.grad(f_val.sum(), uv_ad, create_graph=True)[0]
        for j in range(2):
            grad2 = torch.autograd.grad(
                grad_f[:, j].sum(), uv_ad,
                create_graph=False, retain_graph=(j < 1),
            )[0]
            H[:, 2, j, :] = grad2

        if self.K > 0:
            omega_z = uv @ omegas.T  # (B, K)
            scale = 1.0 / np.sqrt(self.K)
            cos_oz = torch.cos(omega_z)
            sin_oz = torch.sin(omega_z)

            # ωωᵀ: (K, 2, 2)
            omega_outer = torch.bmm(
                omegas.unsqueeze(-1),  # (K, 2, 1)
                omegas.unsqueeze(-2),  # (K, 1, 2)
            )

            # Cos block Hessian: -cos(ω·z) * ωωᵀ / √K
            H[:, 3 : 3 + self.K, :, :] = (
                -cos_oz.unsqueeze(-1).unsqueeze(-1)
                * omega_outer.unsqueeze(0)
                * scale
            )
            # Sin block Hessian: -sin(ω·z) * ωωᵀ / √K
            H[:, 3 + self.K :, :, :] = (
                -sin_oz.unsqueeze(-1).unsqueeze(-1)
                * omega_outer.unsqueeze(0)
                * scale
            )

        return H


def sample_from_highd_manifold(
    surface: FourierAugmentedSurface,
    local_drift_fn: Callable,  # (B, 2) -> (B, 2) batched
    local_diffusion_fn: Callable,  # (B, 2) -> (B, 2, 2) batched
    bounds: list,  # [(u_lo, u_hi), (v_lo, v_hi)]
    n_samples: int,
    seed: int = 42,
    device: str = "cpu",
) -> DatasetBatch:
    """Sample training data from a Fourier-augmented surface.

    All geometric quantities computed numerically (no SymPy).
    """
    torch.manual_seed(seed)
    np.random.seed(seed)

    u_lo, u_hi = bounds[0]
    v_lo, v_hi = bounds[1]
    uv = torch.rand(n_samples, 2, device=device)
    uv[:, 0] = uv[:, 0] * (u_hi - u_lo) + u_lo
    uv[:, 1] = uv[:, 1] * (v_hi - v_lo) + v_lo

    x = surface.chart(uv)  # (B, D)
    J = surface.jacobian(uv)  # (B, D, 2)

    # Metric and projection
    g = torch.bmm(J.transpose(-1, -2), J)  # (B, 2, 2)
    ginv = regularized_metric_inverse(g)  # (B, 2, 2)
    P = torch.bmm(J, torch.bmm(ginv, J.transpose(-1, -2)))  # (B, D, D)

    # Local drift and diffusion (batched)
    mu_local = local_drift_fn(uv)  # (B, 2)
    sigma_local = local_diffusion_fn(uv)  # (B, 2, 2)

    # Local covariance: Σ_z = σ σᵀ
    local_cov = torch.bmm(sigma_local, sigma_local.transpose(-1, -2))  # (B, 2, 2)

    # Hessians
    hess = surface.hessians(uv)  # (B, D, 2, 2)

    # Itô correction: q_r = Σ_{ij} Σ_z_{ij} H_{r,ij}
    # Using einsum: q = einsum("bij, nrij -> nr", local_cov, hess)
    q = ambient_quadratic_variation_drift(local_cov, hess)  # (B, D)

    # Full ambient drift: μ = J · μ_local + (1/2) q
    mu_tangential = torch.bmm(J, mu_local.unsqueeze(-1)).squeeze(-1)  # (B, D)
    mu = mu_tangential + 0.5 * q  # (B, D)

    # Ambient covariance: Λ = J · Σ_z · Jᵀ
    cov = torch.bmm(J, torch.bmm(local_cov, J.transpose(-1, -2)))  # (B, D, D)

    return DatasetBatch(
        samples=x,
        local_samples=uv,
        weights=torch.ones(n_samples, device=device),
        mu=mu,
        cov=cov,
        p=P,
        hessians=hess,
        local_cov=local_cov,
    )


def create_highd_lambdified_sde(
    surface: FourierAugmentedSurface,
    local_drift_fn: Callable,  # (B, 2) -> (B, 2) batched
    local_diffusion_fn: Callable,  # (B, 2) -> (B, 2, 2) batched
):
    """Create a LambdifiedSDE for evaluate_pipeline without SymPy.

    Returns a LambdifiedSDE with batch torch-callable functions matching
    the interface expected by simulate_ground_truth and evaluate_pipeline.
    """
    from experiments.trajectory_fidelity_study import LambdifiedSDE

    def chart_batch(uv: torch.Tensor) -> torch.Tensor:
        return surface.chart(uv)

    return LambdifiedSDE(
        local_drift=local_drift_fn,
        local_diffusion=local_diffusion_fn,
        chart=chart_batch,
        ambient_drift=None,
        ambient_covariance=None,
    )
