"""
Unified multi-seed experiment runner for the paper.

Three subcommands:
  ablation        -- AE ablation across 8 penalty configs (1 surface, default paraboloid)
  extrapolation   -- Reconstruction extrapolation across distances (multiple surfaces)
  dynamics        -- Dynamics extrapolation across distances (1 surface, default paraboloid)

All experiments use N=20 sparse training data, two-phase AE training when K is active,
and 10 seeds by default for statistical rigor.

Usage:
    python -m experiments.paper_experiments ablation --n-seeds 10 --output paper_ablation.csv
    python -m experiments.paper_experiments extrapolation --n-seeds 10 --output paper_extrapolation.csv
    python -m experiments.paper_experiments dynamics --n-seeds 10 --output paper_dynamics.csv
"""

import argparse
import time
from dataclasses import asdict

import numpy as np
import pandas as pd
import sympy as sp
import torch

from src.numeric.datagen import sample_from_manifold
from src.numeric.losses import LossWeights
from src.numeric.performance_stats import compute_losses_per_sample
from src.numeric.training import MultiModelTrainer, TrainingConfig, TrainingPhase

from src.symbolic.manifold_sdes import ManifoldSDE
from src.symbolic.riemannian import RiemannianManifold
from src.symbolic.surfaces import surface

from experiments.common import (
    SURFACE_MAP,
    PENALTY_CONFIGS,
    make_model_config,
    create_test_datasets,
)
from experiments.dynamics_extrapolation_study import compute_dynamics_metrics

# ──────────────────────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────────────────────

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
N_TRAIN = 20
TRAIN_BOUND = 1.0
EPOCHS_AE = 500
BATCH_SIZE = 32
LR = 0.005

# 8-config ablation grid (K weight = 0.1)
ABLATION_CONFIGS = {
    "baseline": LossWeights(),
    "T": LossWeights(tangent_bundle=1.0),
    "K": LossWeights(curvature=0.1),
    "F": LossWeights(diffeo=1.0),
    "T+K": LossWeights(tangent_bundle=1.0, curvature=0.1),
    "T+F": LossWeights(tangent_bundle=1.0, diffeo=1.0),
    "F+K": LossWeights(diffeo=1.0, curvature=0.1),
    "T+F+K": LossWeights(tangent_bundle=1.0, diffeo=1.0, curvature=0.1),
}


# ──────────────────────────────────────────────────────────────
# Manifold SDE factory
# ──────────────────────────────────────────────────────────────

def create_manifold_sde(surface_name, with_dynamics=False):
    u, v = sp.symbols("u v", real=True)
    local_coord, chart = surface(SURFACE_MAP[surface_name], u, v)
    manifold = RiemannianManifold(local_coord, chart)
    if with_dynamics:
        local_drift = sp.Matrix([-v, u])
        local_diffusion = sp.Matrix([[1 + u**2 / 4, u + v], [0, 1 + v**2 / 4]])
    else:
        local_drift, local_diffusion = None, None
    return ManifoldSDE(manifold, local_drift=local_drift, local_diffusion=local_diffusion)


# ──────────────────────────────────────────────────────────────
# Two-phase AE training
# ──────────────────────────────────────────────────────────────

def train_ae_with_twophase(train_data, model_name, lw, epochs):
    """Train AE with two-phase schedule when curvature is active."""
    trainer = MultiModelTrainer(TrainingConfig(
        epochs=epochs,
        n_samples=N_TRAIN,
        input_dim=3,
        hidden_dim=64,
        latent_dim=2,
        learning_rate=LR,
        batch_size=BATCH_SIZE,
        test_size=0.03,
        print_interval=max(1, epochs // 5),
        device=DEVICE,
    ))
    trainer.add_model(make_model_config(model_name, lw, hidden_dims=[64, 64]))
    loader = trainer.create_data_loader(train_data)

    has_K = lw.curvature > 0
    if has_K:
        # Build warmup weights (same but curvature=0)
        warmup_kwargs = {k: v for k, v in asdict(lw).items() if k != "curvature" and v > 0}
        warmup_lw = LossWeights(**warmup_kwargs)

        phase1_epochs = epochs // 2
        phase2_epochs = epochs - phase1_epochs
        schedule = [
            TrainingPhase(epochs=phase1_epochs, loss_weights=warmup_lw, name="warmup"),
            TrainingPhase(epochs=phase2_epochs, loss_weights=lw, name="finetune"),
        ]
        trainer.train_with_schedule(loader, model_name, schedule,
                                    print_interval=max(1, epochs // 5))
    else:
        for epoch in range(epochs):
            losses = trainer.train_epoch(loader)
            if (epoch + 1) % max(1, epochs // 5) == 0:
                print(f"  Epoch {epoch+1}: loss={losses[model_name]:.6f}")

    model = trainer.models[model_name]
    model.eval()
    return model


# ──────────────────────────────────────────────────────────────
# Paired t-test
# ──────────────────────────────────────────────────────────────

def paired_ttest(vals_a, vals_b):
    """Paired t-test: is mean(a - b) significantly different from 0?"""
    diffs = np.array(vals_a) - np.array(vals_b)
    n = len(diffs)
    if n < 2:
        return float("nan"), float("nan")
    mean_d = diffs.mean()
    std_d = diffs.std(ddof=1)
    if std_d < 1e-15:
        return mean_d, 0.0
    t_stat = mean_d / (std_d / np.sqrt(n))
    from scipy import stats as sp_stats
    p_val = sp_stats.t.sf(abs(t_stat), df=n - 1) * 2  # two-sided
    return mean_d, p_val


# ──────────────────────────────────────────────────────────────
# ABLATION STUDY
# ──────────────────────────────────────────────────────────────

def run_ablation_seed(surface_name, config_name, lw, seed, epochs):
    """Run one ablation seed: train AE, evaluate on 500-point test set."""
    torch.manual_seed(seed)
    np.random.seed(seed)

    manifold_sde = create_manifold_sde(surface_name, with_dynamics=True)
    train_data = sample_from_manifold(
        manifold_sde,
        [(-TRAIN_BOUND, TRAIN_BOUND), (-TRAIN_BOUND, TRAIN_BOUND)],
        n_samples=N_TRAIN, seed=seed, device=DEVICE,
    )

    model = train_ae_with_twophase(train_data, config_name, lw, epochs)

    # Evaluate on fresh 500-point test set
    test_data = sample_from_manifold(
        manifold_sde,
        [(-TRAIN_BOUND, TRAIN_BOUND), (-TRAIN_BOUND, TRAIN_BOUND)],
        n_samples=500, seed=seed + 10000, device=DEVICE,
    )
    losses_df = compute_losses_per_sample(model, test_data)

    return {
        "surface": surface_name,
        "config": config_name,
        "seed": seed,
        "reconstruction": losses_df["reconstruction"].mean(),
        "tangent": losses_df["tangent"].mean(),
        "curvature": losses_df["curvature"].mean(),
    }


def run_ablation(surface_name, seeds, epochs, output):
    """Full ablation study across all configs and seeds."""
    print(f"\n{'='*70}")
    print(f"ABLATION STUDY: {surface_name}")
    print(f"Seeds: {seeds}")
    print(f"Configs: {list(ABLATION_CONFIGS.keys())}")
    print(f"{'='*70}")

    rows = []
    for seed in seeds:
        for cfg_name, lw in ABLATION_CONFIGS.items():
            print(f"\n  {surface_name} | {cfg_name} | seed={seed}")
            row = run_ablation_seed(surface_name, cfg_name, lw, seed, epochs)
            rows.append(row)
            print(f"    recon={row['reconstruction']:.6f}  tangent={row['tangent']:.6f}  "
                  f"curvature={row['curvature']:.6f}")

    df = pd.DataFrame(rows)
    if output:
        df.to_csv(output, index=False)
        print(f"\nSaved to {output}")

    # Summary
    print(f"\n{'='*70}")
    print("ABLATION SUMMARY (mean +/- std)")
    print(f"{'='*70}")
    metrics = ["reconstruction", "tangent", "curvature"]
    print(f"{'config':>10s}  {'n':>3s}  ", end="")
    for m in metrics:
        print(f"{m:>14s} {'':>14s}  ", end="")
    print()
    print("-" * 100)

    for cfg_name in ABLATION_CONFIGS:
        subset = df[df["config"] == cfg_name]
        n = len(subset)
        print(f"{cfg_name:>10s}  {n:>3d}  ", end="")
        for m in metrics:
            vals = subset[m].values
            print(f"{vals.mean():>14.6f} +/-{vals.std():>10.6f}  ", end="")
        print()

    return df


# ──────────────────────────────────────────────────────────────
# EXTRAPOLATION STUDY
# ──────────────────────────────────────────────────────────────

def run_extrapolation_seed(surface_name, config_name, lw, seed, epochs, distances, dist_step):
    """Run one extrapolation seed: train AE, eval at each distance."""
    torch.manual_seed(seed)
    np.random.seed(seed)

    manifold_sde = create_manifold_sde(surface_name, with_dynamics=False)
    train_data = sample_from_manifold(
        manifold_sde,
        [(-TRAIN_BOUND, TRAIN_BOUND), (-TRAIN_BOUND, TRAIN_BOUND)],
        n_samples=N_TRAIN, seed=seed, device=DEVICE,
    )

    model = train_ae_with_twophase(train_data, config_name, lw, epochs)

    # Test datasets at each distance
    test_datasets = create_test_datasets(
        manifold_sde, TRAIN_BOUND, distances, dist_step, 500, DEVICE, seed + 10000,
    )

    rows = []
    for dist, test_data in test_datasets.items():
        losses_df = compute_losses_per_sample(model, test_data)
        rows.append({
            "surface": surface_name,
            "config": config_name,
            "seed": seed,
            "distance": dist,
            "reconstruction": losses_df["reconstruction"].mean(),
            "tangent": losses_df["tangent"].mean(),
            "curvature": losses_df["curvature"].mean(),
        })
    return rows


def run_extrapolation(surfaces, seeds, epochs, output):
    """Full extrapolation study across surfaces, configs, seeds."""
    distances = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]
    dist_step = 0.1

    print(f"\n{'='*70}")
    print("EXTRAPOLATION STUDY")
    print(f"Surfaces: {surfaces}")
    print(f"Seeds: {seeds}")
    print(f"Configs: {list(PENALTY_CONFIGS.keys())}")
    print(f"Distances: {distances}")
    print(f"{'='*70}")

    all_rows = []
    for surf in surfaces:
        for seed in seeds:
            for cfg_name, lw in PENALTY_CONFIGS.items():
                print(f"\n  {surf} | {cfg_name} | seed={seed}")
                rows = run_extrapolation_seed(surf, cfg_name, lw, seed, epochs, distances, dist_step)
                all_rows.extend(rows)
                # Print interpolation and max extrapolation
                r0 = [r for r in rows if r["distance"] == 0.0][0]
                r5 = [r for r in rows if r["distance"] == 0.5][0]
                print(f"    dist=0.0: recon={r0['reconstruction']:.6f}")
                print(f"    dist=0.5: recon={r5['reconstruction']:.6f}")

    df = pd.DataFrame(all_rows)
    if output:
        df.to_csv(output, index=False)
        print(f"\nSaved to {output}")

    # Summary at dist=0.0 and dist=0.5
    print(f"\n{'='*70}")
    print("EXTRAPOLATION SUMMARY (mean +/- std)")
    print(f"{'='*70}")
    for dist_val in [0.0, 0.5]:
        print(f"\n  Distance = {dist_val}")
        sub = df[df["distance"] == dist_val]
        print(f"  {'surface':>25s}  {'config':>8s}  {'n':>3s}  {'reconstruction':>14s}")
        print("  " + "-" * 60)
        for surf in surfaces:
            for cfg_name in PENALTY_CONFIGS:
                vals = sub[(sub["surface"] == surf) & (sub["config"] == cfg_name)]["reconstruction"]
                if len(vals) > 0:
                    print(f"  {surf:>25s}  {cfg_name:>8s}  {len(vals):>3d}  "
                          f"{vals.mean():>10.6f} +/-{vals.std():>8.6f}")

    return df


# ──────────────────────────────────────────────────────────────
# DYNAMICS STUDY
# ──────────────────────────────────────────────────────────────

def run_dynamics_extrap_seed(surface_name, config_name, lw, seed, epochs, distances, dist_step):
    """Run one dynamics extrapolation seed: train AE, eval dynamics at each distance."""
    torch.manual_seed(seed)
    np.random.seed(seed)

    manifold_sde = create_manifold_sde(surface_name, with_dynamics=True)
    train_data = sample_from_manifold(
        manifold_sde,
        [(-TRAIN_BOUND, TRAIN_BOUND), (-TRAIN_BOUND, TRAIN_BOUND)],
        n_samples=N_TRAIN, seed=seed, device=DEVICE,
    )

    model = train_ae_with_twophase(train_data, config_name, lw, epochs)

    # Test datasets at each distance
    test_datasets = create_test_datasets(
        manifold_sde, TRAIN_BOUND, distances, dist_step, 500, DEVICE, seed + 10000,
    )

    rows = []
    for dist, test_data in test_datasets.items():
        metrics = compute_dynamics_metrics(model, test_data)
        rows.append({
            "surface": surface_name,
            "config": config_name,
            "seed": seed,
            "distance": dist,
            **metrics,
        })
    return rows


def run_dynamics(surface_name, seeds, epochs, output):
    """Full dynamics extrapolation study across configs and seeds."""
    distances = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]
    dist_step = 0.1

    print(f"\n{'='*70}")
    print(f"DYNAMICS STUDY: {surface_name}")
    print(f"Seeds: {seeds}")
    print(f"Configs: {list(PENALTY_CONFIGS.keys())}")
    print(f"Distances: {distances}")
    print(f"{'='*70}")

    all_rows = []
    for seed in seeds:
        for cfg_name, lw in PENALTY_CONFIGS.items():
            print(f"\n  {surface_name} | {cfg_name} | seed={seed}")
            rows = run_dynamics_extrap_seed(surface_name, cfg_name, lw, seed, epochs, distances, dist_step)
            all_rows.extend(rows)
            r0 = [r for r in rows if r["distance"] == 0.0][0]
            r5 = [r for r in rows if r["distance"] == 0.5][0]
            print(f"    dist=0.0: tangent={r0['tangent']:.6f}  normal_drift={r0['normal_drift_learned']:.6f}")
            print(f"    dist=0.5: tangent={r5['tangent']:.6f}  normal_drift={r5['normal_drift_learned']:.6f}")

    df = pd.DataFrame(all_rows)
    if output:
        df.to_csv(output, index=False)
        print(f"\nSaved to {output}")

    # Summary
    print(f"\n{'='*70}")
    print("DYNAMICS SUMMARY (mean +/- std)")
    print(f"{'='*70}")
    dyn_metrics = ["reconstruction", "tangent", "cov_tangent", "drift_tangent",
                   "normal_drift_learned", "normal_drift_true"]
    for dist_val in [0.0, 0.5]:
        print(f"\n  Distance = {dist_val}")
        sub = df[df["distance"] == dist_val]
        print(f"  {'config':>8s}  {'n':>3s}  ", end="")
        for m in dyn_metrics:
            print(f"{m[:12]:>14s}  ", end="")
        print()
        print("  " + "-" * 110)
        for cfg_name in PENALTY_CONFIGS:
            vals = sub[sub["config"] == cfg_name]
            if len(vals) == 0:
                continue
            print(f"  {cfg_name:>8s}  {len(vals):>3d}  ", end="")
            for m in dyn_metrics:
                v = vals[m].values
                print(f"{v.mean():>7.4f}+/-{v.std():>5.4f}  ", end="")
            print()

    return df


# ──────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Unified paper experiment runner for AEML",
    )
    subparsers = parser.add_subparsers(dest="study", required=True)

    # Common args helper
    def add_common_args(sp):
        sp.add_argument("--n-seeds", type=int, default=10)
        sp.add_argument("--base-seed", type=int, default=42)
        sp.add_argument("--epochs", type=int, default=EPOCHS_AE)
        sp.add_argument("--output", type=str, default=None)

    # Ablation
    p_abl = subparsers.add_parser("ablation", help="AE ablation (8 configs)")
    add_common_args(p_abl)
    p_abl.add_argument("--surface", type=str, default="paraboloid",
                        choices=list(SURFACE_MAP.keys()))

    # Extrapolation
    p_ext = subparsers.add_parser("extrapolation", help="Reconstruction extrapolation")
    add_common_args(p_ext)
    p_ext.add_argument("--surfaces", type=str, nargs="+",
                        default=["paraboloid", "hyperbolic_paraboloid", "monkey_saddle", "sinusoidal"],
                        choices=list(SURFACE_MAP.keys()))

    # Dynamics
    p_dyn = subparsers.add_parser("dynamics", help="Dynamics extrapolation")
    add_common_args(p_dyn)
    p_dyn.add_argument("--surface", type=str, default="paraboloid",
                        choices=list(SURFACE_MAP.keys()))

    args = parser.parse_args()
    seeds = [args.base_seed + i * 1000 for i in range(args.n_seeds)]

    print(f"Device: {DEVICE}")
    print(f"N_TRAIN: {N_TRAIN}")
    print(f"Seeds ({len(seeds)}): {seeds}")
    print(f"Epochs: {args.epochs}")

    t0 = time.time()

    if args.study == "ablation":
        output = args.output or "paper_ablation.csv"
        run_ablation(args.surface, seeds, args.epochs, output)
    elif args.study == "extrapolation":
        output = args.output or "paper_extrapolation.csv"
        run_extrapolation(args.surfaces, seeds, args.epochs, output)
    elif args.study == "dynamics":
        output = args.output or "paper_dynamics.csv"
        run_dynamics(args.surface, seeds, args.epochs, output)

    print(f"\nTotal time: {time.time() - t0:.1f}s")
    print("Done.")


if __name__ == "__main__":
    main()
