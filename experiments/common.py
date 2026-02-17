"""
Shared utilities for AEML experiment scripts.

Provides common surface maps, penalty configs, and data sampling helpers
used across extrapolation and dynamics studies.
"""

import numpy as np
import torch

from src.numeric.datagen import sample_from_manifold
from src.numeric.datasets import DatasetBatch
from src.numeric.losses import LossWeights
from src.symbolic.manifold_sdes import ManifoldSDE
from src.symbolic.surfaces import (
    paraboloid, hyperbolic_paraboloid, monkey_saddle,
    sinusoidal, plane,
)


# Canonical surface map for experiment scripts (keys use underscores).
# For the runner's map (keys with spaces), import _SURFACE_MAP from runner.py.
SURFACE_MAP = {
    "paraboloid": paraboloid,
    "hyperbolic_paraboloid": hyperbolic_paraboloid,
    "monkey_saddle": monkey_saddle,
    "sinusoidal": sinusoidal,
    "plane": plane,
}

# Penalty configs shared by extrapolation studies
PENALTY_CONFIGS = {
    "baseline": LossWeights(),
    "T": LossWeights(tangent_bundle=1.0),
    "K": LossWeights(curvature=1.0),
    "T+K": LossWeights(tangent_bundle=1.0, curvature=1.0),
    "T+F": LossWeights(tangent_bundle=1.0, diffeo=1.0),
    "T+F+K": LossWeights(tangent_bundle=1.0, diffeo=1.0, curvature=1.0),
    "T+Kf":     LossWeights(tangent_bundle=1.0, curvature_full=1.0),
    "T+F+Kf":   LossWeights(tangent_bundle=1.0, diffeo=1.0, curvature_full=1.0),
}


def sample_ring_region(
    manifold_sde: ManifoldSDE,
    inner_bound: float,
    outer_bound: float,
    n_samples: int,
    device: str = "cpu",
    seed: int = 42,
) -> DatasetBatch:
    """Sample from ring region: [-outer, outer] \\ [-inner, inner]."""
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
        local_cov=full_dataset.local_cov[indices] if full_dataset.local_cov is not None else None,
    )


def create_test_datasets(
    manifold_sde: ManifoldSDE,
    train_bound: float,
    distances: list,
    dist_step: float,
    n_test: int,
    device: str,
    seed: int,
) -> dict:
    """Create test datasets at different extrapolation distances.

    Args:
        manifold_sde: The manifold SDE to sample from.
        train_bound: Boundary of the training region.
        distances: List of extrapolation distances (0.0 = interpolation).
        dist_step: Step size between distance rings.
        n_test: Number of test samples per distance.
        device: Torch device.
        seed: Random seed.

    Returns:
        Dict mapping distance -> DatasetBatch.
    """
    test_datasets = {}
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
    return test_datasets
