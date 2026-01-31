"""
Extrapolation Study for AEML

Tests whether curvature penalty helps with extrapolation beyond training region.

Train on [-1, 1] x [-1, 1], then test reconstruction at increasing distances:
- dist=0.0: interpolation (within training region)
- dist=0.2: [-1.2, 1.2] \ [-1, 1]
- dist=0.4: [-1.4, 1.4] \ [-1.2, 1.2]
- etc.

Usage:
    python -m experiments.extrapolation_study --surface paraboloid
"""

import argparse
import torch
import numpy as np
import pandas as pd
import sympy as sp
from dataclasses import dataclass
from typing import Dict, List, Tuple

from src.numeric.datagen import sample_from_manifold
from src.numeric.datasets import DatasetBatch
from src.numeric.losses import LossWeights, l2_loss
from src.numeric.training import ModelConfig, MultiModelTrainer, TrainingConfig, train_models
from src.symbolic.manifold_sdes import ManifoldSDE
from src.symbolic.riemannian import RiemannianManifold
from src.symbolic.surfaces import (
    paraboloid, hyperbolic_paraboloid, monkey_saddle,
    gaussian_bump, sinusoidal, plane, surface
)

SURFACE_MAP = {
    "paraboloid": paraboloid,
    "hyperbolic_paraboloid": hyperbolic_paraboloid,
    "monkey_saddle": monkey_saddle,
    "sinusoidal": sinusoidal,
    "plane": plane,
}

# Penalty configs to compare
PENALTY_CONFIGS = {
    "baseline": LossWeights(),
    "T": LossWeights(tangent_bundle=1.0),
    "T+K": LossWeights(tangent_bundle=1.0, curvature=1.0),
    "T+F": LossWeights(tangent_bundle=1.0, diffeo=1.0),
    "T+F+K": LossWeights(tangent_bundle=1.0, diffeo=1.0, curvature=1.0),
    "K": LossWeights(curvature=1.0),
}


def create_manifold_sde(surface_name: str):
    """Create manifold SDE for given surface."""
    u, v = sp.symbols("u v", real=True)
    local_coord, chart = surface(SURFACE_MAP[surface_name], u, v)

    manifold = RiemannianManifold(local_coord, chart)
    # Use simple RBM (no drift/diffusion) for cleaner extrapolation test
    manifold_sde = ManifoldSDE(manifold, local_drift=None, local_diffusion=None)

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

    # Oversample and filter to ring region
    oversample_factor = 4
    full_dataset = sample_from_manifold(
        manifold_sde,
        [(-outer_bound, outer_bound), (-outer_bound, outer_bound)],
        n_samples=n_samples * oversample_factor,
        seed=seed,
        device=device,
    )

    # Filter to ring region
    local = full_dataset.local_samples
    in_outer = (local[:, 0].abs() <= outer_bound) & (local[:, 1].abs() <= outer_bound)
    in_inner = (local[:, 0].abs() <= inner_bound) & (local[:, 1].abs() <= inner_bound)
    in_ring = in_outer & ~in_inner

    ring_indices = torch.where(in_ring)[0]

    if len(ring_indices) < n_samples:
        # Use what we have
        indices = ring_indices
    else:
        # Subsample
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


def evaluate_reconstruction(model, dataset: DatasetBatch) -> float:
    """Compute mean reconstruction error."""
    model.eval()
    with torch.no_grad():
        x = dataset.samples
        x_hat = model(x)
        errors = ((x_hat - x) ** 2).sum(dim=-1).sqrt()  # L2 error per sample
        return errors.mean().item()


def run_extrapolation_study(
    surface_name: str,
    train_bound: float = 1.0,
    max_extrap_dist: float = 1.0,
    dist_step: float = 0.2,
    n_train: int = 2000,
    n_test: int = 500,
    epochs: int = 1000,
    device: str = "cuda",
    seed: int = 42,
) -> pd.DataFrame:
    """Run extrapolation study."""

    print(f"\n{'='*60}")
    print(f"Extrapolation Study: {surface_name}")
    print(f"Train region: [-{train_bound}, {train_bound}]")
    print(f"Max extrapolation distance: {max_extrap_dist}")
    print(f"{'='*60}")

    # Create manifold
    manifold_sde = create_manifold_sde(surface_name)

    # Sample training data
    print("\nSampling training data...")
    train_data = sample_from_manifold(
        manifold_sde,
        [(-train_bound, train_bound), (-train_bound, train_bound)],
        n_samples=n_train,
        seed=seed,
        device=device,
    )

    # Create test datasets at different extrapolation distances
    distances = [0.0] + list(np.arange(dist_step, max_extrap_dist + dist_step/2, dist_step))
    test_datasets = {}

    print("\nSampling test data at different distances...")
    for dist in distances:
        if dist == 0.0:
            # Interpolation test (within training region)
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

    # Train models with different penalties
    results = []

    for penalty_name, loss_weights in PENALTY_CONFIGS.items():
        print(f"\n--- Training: {penalty_name} ---")

        # Create trainer
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

        # Train
        data_loader = trainer.create_data_loader(train_data)
        for epoch in range(epochs):
            losses = trainer.train_epoch(data_loader)
            if (epoch + 1) % (epochs // 5) == 0:
                print(f"  Epoch {epoch+1}: {losses[penalty_name]:.6f}")

        # Evaluate at each distance
        model = trainer.models[penalty_name]
        for dist, test_data in test_datasets.items():
            recon_error = evaluate_reconstruction(model, test_data)
            results.append({
                "surface": surface_name,
                "penalty": penalty_name,
                "distance": dist,
                "reconstruction_error": recon_error,
                "n_test_samples": len(test_data.samples),
            })
            print(f"  dist={dist:.1f}: recon_error={recon_error:.6f}")

    return pd.DataFrame(results)


def plot_results(df: pd.DataFrame, output_path: str = None):
    """Plot extrapolation results."""
    try:
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(10, 6))

        for penalty in df['penalty'].unique():
            subset = df[df['penalty'] == penalty]
            ax.plot(subset['distance'], subset['reconstruction_error'],
                   marker='o', label=penalty, linewidth=2, markersize=8)

        ax.set_xlabel('Extrapolation Distance', fontsize=12)
        ax.set_ylabel('Reconstruction Error (L2)', fontsize=12)
        ax.set_title(f"Extrapolation Performance: {df['surface'].iloc[0]}", fontsize=14)
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.set_yscale('log')

        if output_path:
            plt.savefig(output_path, dpi=150, bbox_inches='tight')
            print(f"\nPlot saved to {output_path}")
        plt.show()

    except ImportError:
        print("matplotlib not available for plotting")


def main():
    parser = argparse.ArgumentParser(description="AEML Extrapolation Study")
    parser.add_argument("--surface", type=str, default="paraboloid",
                       choices=list(SURFACE_MAP.keys()))
    parser.add_argument("--train_bound", type=float, default=1.0)
    parser.add_argument("--max_dist", type=float, default=1.0)
    parser.add_argument("--dist_step", type=float, default=0.2)
    parser.add_argument("--n_train", type=int, default=2000)
    parser.add_argument("--n_test", type=int, default=500)
    parser.add_argument("--epochs", type=int, default=1000)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=str, default="extrapolation_results.csv")
    parser.add_argument("--plot", action="store_true")

    args = parser.parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    df = run_extrapolation_study(
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

    # Save results
    df.to_csv(args.output, index=False)
    print(f"\nResults saved to {args.output}")

    # Print summary
    print("\n" + "="*60)
    print("EXTRAPOLATION SUMMARY")
    print("="*60)
    pivot = df.pivot(index='distance', columns='penalty', values='reconstruction_error')
    print(pivot.round(6))

    if args.plot:
        plot_results(df, args.output.replace('.csv', '.png'))


if __name__ == "__main__":
    main()
