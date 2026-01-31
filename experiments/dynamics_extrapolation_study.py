"""
Dynamics Extrapolation Study for AEML

Tests whether curvature penalty helps match DYNAMICS (drift & covariance) outside training region.

Unlike reconstruction extrapolation, this measures:
1. Drift matching: ||∇φ·μ + ½q - b|| (tangent + Ito correction = ambient drift)
2. Covariance matching: ||∇φ·Λ^l·∇φᵀ - Λ|| (transformed local cov = ambient cov)
3. Tangent alignment: ||P_learned - P_true||
4. Curvature drift: ||(I-P)b - ½II:Λ||

Usage:
    python -m experiments.dynamics_extrapolation_study --surface paraboloid
"""

import argparse
import torch
import numpy as np
import pandas as pd
import sympy as sp
from dataclasses import dataclass
from typing import Dict, List, Tuple

from src.numeric.autoencoders import AutoEncoder
from src.numeric.datagen import sample_from_manifold
from src.numeric.datasets import DatasetBatch
from src.numeric.losses import LossWeights
from src.numeric.geometry import (
    transform_covariance,
    orthogonal_projection_from_jacobian,
)
from src.numeric.training import ModelConfig, MultiModelTrainer, TrainingConfig
from src.symbolic.manifold_sdes import ManifoldSDE
from src.symbolic.riemannian import RiemannianManifold
from src.symbolic.surfaces import (
    paraboloid, hyperbolic_paraboloid, monkey_saddle,
    sinusoidal, plane, surface
)

SURFACE_MAP = {
    "paraboloid": paraboloid,
    "hyperbolic_paraboloid": hyperbolic_paraboloid,
    "monkey_saddle": monkey_saddle,
    "sinusoidal": sinusoidal,
    "plane": plane,
}

PENALTY_CONFIGS = {
    "baseline": LossWeights(),
    "T": LossWeights(tangent_bundle=1.0),
    "K": LossWeights(curvature=1.0),
    "T+K": LossWeights(tangent_bundle=1.0, curvature=1.0),
    "T+F": LossWeights(tangent_bundle=1.0, diffeo=1.0),
    "T+F+K": LossWeights(tangent_bundle=1.0, diffeo=1.0, curvature=1.0),
}


def create_manifold_sde(surface_name: str):
    """Create manifold SDE with non-trivial dynamics."""
    u, v = sp.symbols("u v", real=True)
    local_coord, chart = surface(SURFACE_MAP[surface_name], u, v)

    manifold = RiemannianManifold(local_coord, chart)

    # Non-trivial dynamics (same as in runner.py)
    local_drift = sp.Matrix([-v, u])
    local_diffusion = sp.Matrix([[1 + u**2/4, u + v], [0, 1 + v**2/4]])

    manifold_sde = ManifoldSDE(
        manifold,
        local_drift=local_drift,
        local_diffusion=local_diffusion,
    )

    return manifold_sde


def sample_ring_region(
    manifold_sde: ManifoldSDE,
    inner_bound: float,
    outer_bound: float,
    n_samples: int,
    device: str = "cpu",
    seed: int = 42,
) -> DatasetBatch:
    """Sample from ring region: [-outer, outer] \ [-inner, inner]."""
    torch.manual_seed(seed)
    np.random.seed(seed)

    oversample_factor = 4
    full_dataset = sample_from_manifold(
        manifold_sde,
        [(-outer_bound, outer_bound), (-outer_bound, outer_bound)],
        n_samples=n_samples * oversample_factor,
        seed=seed,
        device=device,
    )

    local = full_dataset.local_samples
    in_outer = (local[:, 0].abs() <= outer_bound) & (local[:, 1].abs() <= outer_bound)
    in_inner = (local[:, 0].abs() <= inner_bound) & (local[:, 1].abs() <= inner_bound)
    in_ring = in_outer & ~in_inner

    ring_indices = torch.where(in_ring)[0]

    if len(ring_indices) < n_samples:
        indices = ring_indices
    else:
        perm = torch.randperm(len(ring_indices))[:n_samples]
        indices = ring_indices[perm]

    return DatasetBatch(
        samples=full_dataset.samples[indices],
        local_samples=full_dataset.local_samples[indices],
        weights=full_dataset.weights[indices],
        mu=full_dataset.mu[indices],
        cov=full_dataset.cov[indices],
        p=full_dataset.p[indices],
        hessians=full_dataset.hessians[indices],
    )


def compute_dynamics_metrics(model: AutoEncoder, dataset: DatasetBatch) -> Dict[str, float]:
    """
    Compute dynamics matching metrics.

    Returns:
        - reconstruction: ||x - φ(ψ(x))||²
        - tangent: ||P_learned - P_true||²_F
        - drift_matching: ||∇φ·b^l + ½q - b||² (ambient drift error)
        - cov_matching: ||∇φ·Λ^l·∇φᵀ - Λ||²_F (covariance transformation error)
        - curvature_drift: ||(I-P)b - ½II:Λ||² (normal drift = curvature term)
    """
    model.eval()
    device = dataset.samples.device

    with torch.no_grad():
        x = dataset.samples  # (B, D)
        z = model.encoder(x)  # (B, d)
        x_hat = model.decoder(z)  # (B, D)

        B, D = x.shape
        d = z.shape[1]

        # Reconstruction error
        recon_error = ((x_hat - x) ** 2).sum(dim=-1).mean().item()

    # Need gradients for Jacobian computation
    model.eval()
    x = dataset.samples.clone().requires_grad_(True)
    z = model.encoder(x)

    # Compute decoder Jacobian: ∇φ (B, D, d)
    dphi = model.decoder_jacobian(z)

    # Learned tangent projector: P = ∇φ (∇φᵀ∇φ)⁻¹ ∇φᵀ
    dphi_T = dphi.transpose(-1, -2)  # (B, d, D)
    gram = torch.bmm(dphi_T, dphi)  # (B, d, d)
    gram_inv = torch.linalg.inv(gram)  # (B, d, d)
    P_learned = torch.bmm(torch.bmm(dphi, gram_inv), dphi_T)  # (B, D, D)

    # True tangent projector
    P_true = dataset.p  # (B, D, D)

    # Tangent alignment error
    tangent_error = ((P_learned - P_true) ** 2).sum(dim=(-1, -2)).mean().item() / 2

    # Transform local covariance to ambient: ∇φ · Λ^l · ∇φᵀ
    # We need local covariance - compute from ambient via pseudo-inverse
    # Λ^l = (∇φᵀ∇φ)⁻¹ ∇φᵀ Λ ∇φ (∇φᵀ∇φ)⁻¹
    Lambda_ambient = dataset.cov  # (B, D, D) - true ambient covariance

    # Transformed covariance: ∇φ · Λ^l_learned · ∇φᵀ
    # If encoder-decoder is diffeomorphism: Λ^l_learned ≈ ∇ψ · Λ · ∇ψᵀ
    # Then ∇φ · Λ^l · ∇φᵀ should ≈ Λ (if spans match)
    Lambda_transformed = torch.bmm(torch.bmm(dphi, gram_inv),
                                   torch.bmm(dphi_T, torch.bmm(Lambda_ambient,
                                   torch.bmm(dphi, torch.bmm(gram_inv, dphi_T)))))

    # Simpler: check if P·Λ·P ≈ Λ (covariance lies in tangent space)
    P_Lambda_P = torch.bmm(torch.bmm(P_learned, Lambda_ambient), P_learned)
    cov_tangent_error = ((P_Lambda_P - Lambda_ambient) ** 2).sum(dim=(-1, -2)).mean().item()

    # Ambient drift
    mu_ambient = dataset.mu  # (B, D) - true ambient drift

    # Normal projector
    I = torch.eye(D, device=device).unsqueeze(0).expand(B, -1, -1)
    N_learned = I - P_learned  # (B, D, D)

    # Normal component of drift: (I-P)·b
    normal_drift = torch.bmm(N_learned, mu_ambient.unsqueeze(-1)).squeeze(-1)  # (B, D)

    # Compute curvature drift error: ||(I-P)b||²
    # The normal drift should be small if manifold is learned correctly
    # For true manifold: (I-P)b = ½ II:Λ (curvature term)
    # We compare learned normal drift magnitude
    curvature_drift_error = (normal_drift ** 2).sum(dim=-1).mean().item()

    # Also compute using true projector for reference
    N_true = I - P_true
    normal_drift_true = torch.bmm(N_true, mu_ambient.unsqueeze(-1)).squeeze(-1)
    curvature_drift_true = (normal_drift_true ** 2).sum(dim=-1).mean().item()

    # Tangent component of drift matching: P·b ≈ ∇φ·b^l
    tangent_drift = torch.bmm(P_learned, mu_ambient.unsqueeze(-1)).squeeze(-1)  # (B, D)
    # True tangent drift from P_true
    tangent_drift_true = torch.bmm(P_true, mu_ambient.unsqueeze(-1)).squeeze(-1)
    drift_tangent_error = ((tangent_drift - tangent_drift_true) ** 2).sum(dim=-1).mean().item()

    return {
        "reconstruction": recon_error,
        "tangent": tangent_error,
        "cov_tangent": cov_tangent_error,
        "drift_tangent": drift_tangent_error,
        "normal_drift_learned": curvature_drift_error,
        "normal_drift_true": curvature_drift_true,
    }


def run_dynamics_extrapolation_study(
    surface_name: str,
    train_bound: float = 1.0,
    max_extrap_dist: float = 0.5,
    dist_step: float = 0.1,
    n_train: int = 2000,
    n_test: int = 500,
    epochs: int = 500,
    device: str = "cuda",
    seed: int = 42,
) -> pd.DataFrame:
    """Run dynamics extrapolation study."""

    print(f"\n{'='*60}")
    print(f"Dynamics Extrapolation Study: {surface_name}")
    print(f"Train region: [-{train_bound}, {train_bound}]")
    print(f"Testing DYNAMICS matching (drift, covariance, curvature)")
    print(f"{'='*60}")

    # Create manifold with non-trivial dynamics
    manifold_sde = create_manifold_sde(surface_name)
    print("Using non-trivial SDE dynamics (not RBM)")

    # Sample training data
    print("\nSampling training data...")
    train_data = sample_from_manifold(
        manifold_sde,
        [(-train_bound, train_bound), (-train_bound, train_bound)],
        n_samples=n_train,
        seed=seed,
        device=device,
    )

    # Create test datasets
    distances = [0.0] + list(np.arange(dist_step, max_extrap_dist + dist_step/2, dist_step))
    test_datasets = {}

    print("\nSampling test data at different distances...")
    for dist in distances:
        if dist == 0.0:
            test_datasets[dist] = sample_from_manifold(
                manifold_sde,
                [(-train_bound, train_bound), (-train_bound, train_bound)],
                n_samples=n_test,
                seed=seed + 1000,
                device=device,
            )
        else:
            inner = train_bound + dist - dist_step
            outer = train_bound + dist
            test_datasets[dist] = sample_ring_region(
                manifold_sde, inner, outer, n_test, device, seed + int(dist * 1000)
            )
        print(f"  dist={dist:.1f}: {len(test_datasets[dist].samples)} samples")

    results = []

    for penalty_name, loss_weights in PENALTY_CONFIGS.items():
        print(f"\n--- Training: {penalty_name} ---")

        trainer = MultiModelTrainer(TrainingConfig(
            epochs=epochs,
            n_samples=n_train,
            input_dim=3,
            hidden_dim=64,
            latent_dim=2,
            learning_rate=0.005,
            batch_size=32,
            test_size=0.03,
            print_interval=epochs // 5,
            device=device,
        ))

        trainer.add_model(ModelConfig(name=penalty_name, loss_weights=loss_weights))

        data_loader = trainer.create_data_loader(train_data)
        for epoch in range(epochs):
            losses = trainer.train_epoch(data_loader)
            if (epoch + 1) % (epochs // 5) == 0:
                print(f"  Epoch {epoch+1}: {losses[penalty_name]:.6f}")

        model = trainer.models[penalty_name]

        for dist, test_data in test_datasets.items():
            metrics = compute_dynamics_metrics(model, test_data)

            result = {
                "surface": surface_name,
                "penalty": penalty_name,
                "distance": dist,
                **metrics,
            }
            results.append(result)

            print(f"  dist={dist:.1f}: recon={metrics['reconstruction']:.4f}, "
                  f"tangent={metrics['tangent']:.4f}, normal_drift={metrics['normal_drift_learned']:.4f}")

    return pd.DataFrame(results)


def main():
    parser = argparse.ArgumentParser(description="AEML Dynamics Extrapolation Study")
    parser.add_argument("--surface", type=str, default="paraboloid",
                       choices=list(SURFACE_MAP.keys()))
    parser.add_argument("--train_bound", type=float, default=1.0)
    parser.add_argument("--max_dist", type=float, default=0.5)
    parser.add_argument("--dist_step", type=float, default=0.1)
    parser.add_argument("--n_train", type=int, default=2000)
    parser.add_argument("--n_test", type=int, default=500)
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=str, default="dynamics_extrapolation_results.csv")

    args = parser.parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    df = run_dynamics_extrapolation_study(
        surface_name=args.surface,
        train_bound=args.train_bound,
        max_extrap_dist=args.max_dist,
        dist_step=args.dist_step,
        n_train=args.n_train,
        n_test=args.n_test,
        epochs=args.epochs,
        device=device,
        seed=args.seed,
    )

    df.to_csv(args.output, index=False)
    print(f"\nResults saved to {args.output}")

    # Print summary
    print("\n" + "=" * 70)
    print("DYNAMICS EXTRAPOLATION SUMMARY")
    print("=" * 70)

    for metric in ['reconstruction', 'tangent', 'normal_drift_learned', 'cov_tangent']:
        print(f"\n{metric.upper()} at dist=0.5:")
        dist_05 = df[df['distance'] == 0.5].pivot(
            index='surface', columns='penalty', values=metric
        )
        if not dist_05.empty:
            print(dist_05.round(4))


if __name__ == "__main__":
    main()
