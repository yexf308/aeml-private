"""
Penalty Extrapolation Evaluation: N (normal_decoder_jacobian) and K (curvature)

Evaluates how the two new penalties affect extrapolation quality across:
1. Manifold extrapolation — reconstruction + tangent alignment
2. Diffusion extrapolation — covariance tangency
3. Drift extrapolation — tangent drift, projection-only drift, curvature-corrected drift

Key insight: The normal drift identity (I-P)b = ½II:Λ allows reconstructing drift
from the model's Hessian. The K penalty trains this reconstruction to be accurate.

Usage:
    # Quick smoke test
    python -m experiments.penalty_extrapolation_eval --surfaces paraboloid --epochs 50

    # Full evaluation
    python -m experiments.penalty_extrapolation_eval --epochs 500
"""

import argparse
import time
from typing import Dict

import numpy as np
import pandas as pd
import sympy as sp
import torch

from experiments.common import SURFACE_MAP, create_test_datasets
from src.numeric.autoencoders import AutoEncoder
from src.numeric.datagen import sample_from_manifold
from src.numeric.datasets import DatasetBatch
from src.numeric.geometry import (
    ambient_quadratic_variation_drift,
    regularized_metric_inverse,
    transform_covariance,
)
from src.numeric.losses import LossWeights
from src.numeric.training import ModelConfig, MultiModelTrainer, TrainingConfig
from src.symbolic.manifold_sdes import ManifoldSDE
from src.symbolic.riemannian import RiemannianManifold
from src.symbolic.surfaces import surface

# ---------------------------------------------------------------------------
# Penalty configurations to evaluate
# ---------------------------------------------------------------------------
EVAL_PENALTY_CONFIGS = {
    "baseline": LossWeights(),
    "T": LossWeights(tangent_bundle=1.0),
    "N": LossWeights(normal_decoder_jacobian=0.01),
    "K": LossWeights(curvature=1.0),
    "T+N": LossWeights(tangent_bundle=1.0, normal_decoder_jacobian=0.01),
    "T+K": LossWeights(tangent_bundle=1.0, curvature=1.0),
    "N+K": LossWeights(normal_decoder_jacobian=0.01, curvature=1.0),
    "T+N+K": LossWeights(tangent_bundle=1.0, normal_decoder_jacobian=0.01, curvature=1.0),
    "T+F": LossWeights(tangent_bundle=1.0, diffeo=1.0),
    "T+F+K": LossWeights(tangent_bundle=1.0, diffeo=1.0, curvature=1.0),
    "T+F+N": LossWeights(tangent_bundle=1.0, diffeo=1.0, normal_decoder_jacobian=0.01),
    "T+F+N+K": LossWeights(tangent_bundle=1.0, diffeo=1.0, normal_decoder_jacobian=0.01, curvature=1.0),
}


def create_manifold_sde(surface_name: str) -> ManifoldSDE:
    """Create manifold SDE with non-trivial dynamics."""
    u, v = sp.symbols("u v", real=True)
    local_coord, chart = surface(SURFACE_MAP[surface_name], u, v)
    manifold = RiemannianManifold(local_coord, chart)
    local_drift = sp.Matrix([-v, u])
    local_diffusion = sp.Matrix([[1 + u**2 / 4, u + v], [0, 1 + v**2 / 4]])
    return ManifoldSDE(manifold, local_drift=local_drift, local_diffusion=local_diffusion)


def compute_eval_metrics(model: AutoEncoder, dataset: DatasetBatch) -> Dict[str, float]:
    """
    Compute manifold, diffusion, and drift extrapolation metrics.

    Enhanced over dynamics_extrapolation_study: adds Hessian-based drift
    reconstruction using the curvature identity (I-P)b = ½(I-P)q.

    Returns dict with keys:
        Manifold:   reconstruction, tangent
        Diffusion:  cov_tangent
        Drift:      drift_tangent, normal_drift_true,
                    drift_proj_only, drift_curvature_corrected, curvature_drift_match
    """
    model.eval()
    device = dataset.samples.device

    # --- Reconstruction (no grad needed) ---
    with torch.no_grad():
        x = dataset.samples
        z = model.encoder(x)
        x_hat = model.decoder(z)
        recon_error = ((x_hat - x) ** 2).sum(dim=-1).mean().item()

    # --- Jacobian-based metrics (need grad for torch.func) ---
    x = dataset.samples.clone().requires_grad_(True)
    z = model.encoder(x)
    B, D = dataset.samples.shape
    d = z.shape[1]

    dphi = model.decoder_jacobian(z)  # (B, D, d)
    dphi_T = dphi.transpose(-1, -2)   # (B, d, D)

    # Learned tangent projector: P̂ = ∇φ (∇φᵀ∇φ)⁻¹ ∇φᵀ
    gram = torch.bmm(dphi_T, dphi)
    gram_inv = regularized_metric_inverse(gram)
    P_learned = torch.bmm(torch.bmm(dphi, gram_inv), dphi_T)

    P_true = dataset.p
    I_mat = torch.eye(D, device=device).unsqueeze(0).expand(B, -1, -1)
    N_learned = I_mat - P_learned
    N_true = I_mat - P_true

    # --- Manifold metrics ---
    tangent_error = ((P_learned - P_true) ** 2).sum(dim=(-1, -2)).mean().item() / 2

    # --- Diffusion metric ---
    Lambda_ambient = dataset.cov
    P_Lambda_P = torch.bmm(torch.bmm(P_learned, Lambda_ambient), P_learned)
    cov_tangent_error = ((P_Lambda_P - Lambda_ambient) ** 2).sum(dim=(-1, -2)).mean().item()

    # --- Drift metrics ---
    mu_ambient = dataset.mu  # (B, D)

    # Tangent drift error: ||P̂·b - P·b||²
    tangent_drift_learned = torch.bmm(P_learned, mu_ambient.unsqueeze(-1)).squeeze(-1)
    tangent_drift_true = torch.bmm(P_true, mu_ambient.unsqueeze(-1)).squeeze(-1)
    drift_tangent_error = ((tangent_drift_learned - tangent_drift_true) ** 2).sum(dim=-1).mean().item()

    # True normal drift magnitude: ||(I-P)·b||²
    normal_drift_true_vec = torch.bmm(N_true, mu_ambient.unsqueeze(-1)).squeeze(-1)
    normal_drift_true_val = (normal_drift_true_vec ** 2).sum(dim=-1).mean().item()

    # Projection-only drift error: ||P̂·b - b||² = ||(I-P̂)·b||²
    drift_proj_only_error = ((tangent_drift_learned - mu_ambient) ** 2).sum(dim=-1).mean().item()

    # --- Curvature-corrected drift (uses Hessian) ---
    d2phi = model.hessian_decoder(z)  # (B, D, d, d)
    dphi_pinv = torch.linalg.pinv(dphi)  # (B, d, D)
    local_cov = transform_covariance(Lambda_ambient, dphi_pinv)  # (B, d, d)
    qhat = ambient_quadratic_variation_drift(local_cov, d2phi)  # (B, D)

    # Model's curvature correction: ½(I-P̂)·q̂
    curvature_correction = torch.bmm(N_learned, (0.5 * qhat).unsqueeze(-1)).squeeze(-1)

    # Curvature-corrected drift: P̂·b + ½(I-P̂)·q̂
    drift_corrected = tangent_drift_learned + curvature_correction
    drift_corrected_error = ((drift_corrected - mu_ambient) ** 2).sum(dim=-1).mean().item()

    # Curvature drift match: ||½(I-P̂)·q̂ - (I-P)·b||²
    curvature_match_error = ((curvature_correction - normal_drift_true_vec) ** 2).sum(dim=-1).mean().item()

    return {
        # Manifold
        "reconstruction": recon_error,
        "tangent": tangent_error,
        # Diffusion
        "cov_tangent": cov_tangent_error,
        # Drift
        "drift_tangent": drift_tangent_error,
        "normal_drift_true": normal_drift_true_val,
        "drift_proj_only": drift_proj_only_error,
        "drift_curvature_corrected": drift_corrected_error,
        "curvature_drift_match": curvature_match_error,
    }


def run_evaluation(
    surfaces: list,
    penalty_configs: Dict[str, LossWeights],
    epochs: int = 500,
    n_train: int = 2000,
    n_test: int = 500,
    train_bound: float = 1.0,
    max_dist: float = 0.5,
    dist_step: float = 0.1,
    device: str = "cpu",
    seed: int = 42,
) -> pd.DataFrame:
    """Run full penalty extrapolation evaluation."""

    all_results = []

    for surface_name in surfaces:
        print(f"\n{'='*70}")
        print(f"Surface: {surface_name}")
        print(f"{'='*70}")

        t0 = time.time()
        manifold_sde = create_manifold_sde(surface_name)
        print(f"  Manifold SDE created in {time.time()-t0:.1f}s")

        # Sample training data
        train_data = sample_from_manifold(
            manifold_sde,
            [(-train_bound, train_bound), (-train_bound, train_bound)],
            n_samples=n_train,
            seed=seed,
            device=device,
        )

        # Create test datasets at different distances
        distances = [0.0] + list(np.arange(dist_step, max_dist + dist_step / 2, dist_step))
        test_datasets = create_test_datasets(
            manifold_sde, train_bound, distances, dist_step, n_test, device, seed
        )
        for dist in distances:
            print(f"  Test dist={dist:.1f}: {len(test_datasets[dist].samples)} samples")

        # Train and evaluate each penalty config
        for penalty_name, loss_weights in penalty_configs.items():
            print(f"\n  --- {penalty_name} ---")
            t1 = time.time()

            trainer = MultiModelTrainer(TrainingConfig(
                epochs=epochs,
                n_samples=n_train,
                input_dim=3,
                hidden_dim=64,
                latent_dim=2,
                learning_rate=0.005,
                batch_size=32,
                test_size=0.03,
                print_interval=max(epochs // 5, 1),
                device=device,
            ))
            trainer.add_model(ModelConfig(name=penalty_name, loss_weights=loss_weights))

            data_loader = trainer.create_data_loader(train_data)
            for epoch in range(epochs):
                losses = trainer.train_epoch(data_loader)
                if (epoch + 1) % max(epochs // 5, 1) == 0:
                    print(f"    Epoch {epoch+1}: {losses[penalty_name]:.6f}")

            model = trainer.models[penalty_name]
            train_time = time.time() - t1

            for dist in distances:
                metrics = compute_eval_metrics(model, test_datasets[dist])
                all_results.append({
                    "surface": surface_name,
                    "penalty": penalty_name,
                    "distance": dist,
                    "train_time": train_time,
                    **metrics,
                })

            # Print summary for this penalty
            d0 = [r for r in all_results if r["surface"] == surface_name
                  and r["penalty"] == penalty_name and r["distance"] == 0.0]
            dmax = [r for r in all_results if r["surface"] == surface_name
                    and r["penalty"] == penalty_name and r["distance"] == distances[-1]]
            if d0 and dmax:
                print(f"    Trained in {train_time:.1f}s | "
                      f"recon 0→{distances[-1]}: {d0[0]['reconstruction']:.4f}→{dmax[0]['reconstruction']:.4f} | "
                      f"tangent: {d0[0]['tangent']:.4f}→{dmax[0]['tangent']:.4f} | "
                      f"drift_corr: {d0[0]['drift_curvature_corrected']:.4f}→{dmax[0]['drift_curvature_corrected']:.4f}")

    return pd.DataFrame(all_results)


def print_summary(df: pd.DataFrame):
    """Print structured summary tables."""
    surfaces = df["surface"].unique()
    penalties = df["penalty"].unique()
    max_dist = df["distance"].max()

    # ================================================================
    # MANIFOLD EXTRAPOLATION
    # ================================================================
    print(f"\n{'='*70}")
    print("MANIFOLD EXTRAPOLATION")
    print(f"{'='*70}")

    for metric, label in [("reconstruction", "Reconstruction Error"),
                          ("tangent", "Tangent Alignment Error")]:
        print(f"\n{label} at dist=0.0 and dist={max_dist}:")
        print("-" * 70)
        header = f"{'penalty':<14}"
        for s in surfaces:
            header += f"  {s[:13]:>13}(0) {s[:13]:>13}({max_dist})"
        print(header)
        print("-" * 70)
        for p in penalties:
            row = f"{p:<14}"
            for s in surfaces:
                v0 = df[(df.penalty == p) & (df.surface == s) & (df.distance == 0.0)][metric]
                vm = df[(df.penalty == p) & (df.surface == s) & (df.distance == max_dist)][metric]
                v0_val = v0.values[0] if len(v0) > 0 else float("nan")
                vm_val = vm.values[0] if len(vm) > 0 else float("nan")
                row += f"  {v0_val:>13.6f} {vm_val:>13.6f}"
            print(row)

    # ================================================================
    # DIFFUSION EXTRAPOLATION
    # ================================================================
    print(f"\n{'='*70}")
    print("DIFFUSION EXTRAPOLATION")
    print(f"{'='*70}")

    print(f"\nCovariance Tangency Error at dist=0.0 and dist={max_dist}:")
    print("-" * 70)
    header = f"{'penalty':<14}"
    for s in surfaces:
        header += f"  {s[:13]:>13}(0) {s[:13]:>13}({max_dist})"
    print(header)
    print("-" * 70)
    for p in penalties:
        row = f"{p:<14}"
        for s in surfaces:
            v0 = df[(df.penalty == p) & (df.surface == s) & (df.distance == 0.0)]["cov_tangent"]
            vm = df[(df.penalty == p) & (df.surface == s) & (df.distance == max_dist)]["cov_tangent"]
            v0_val = v0.values[0] if len(v0) > 0 else float("nan")
            vm_val = vm.values[0] if len(vm) > 0 else float("nan")
            row += f"  {v0_val:>13.6f} {vm_val:>13.6f}"
        print(row)

    # ================================================================
    # DRIFT EXTRAPOLATION
    # ================================================================
    print(f"\n{'='*70}")
    print("DRIFT EXTRAPOLATION")
    print(f"{'='*70}")

    for metric, label in [("drift_tangent", "Tangent Drift Error ||P_hat*b - P*b||^2"),
                          ("drift_proj_only", "Projection-Only Drift Error ||P_hat*b - b||^2"),
                          ("drift_curvature_corrected", "Curvature-Corrected Drift Error ||P_hat*b + 1/2(I-P_hat)q_hat - b||^2"),
                          ("curvature_drift_match", "Curvature Drift Match ||1/2(I-P_hat)q_hat - (I-P)b||^2")]:
        print(f"\n{label}")
        print(f"  at dist=0.0 and dist={max_dist}:")
        print("-" * 70)
        header = f"{'penalty':<14}"
        for s in surfaces:
            header += f"  {s[:13]:>13}(0) {s[:13]:>13}({max_dist})"
        print(header)
        print("-" * 70)
        for p in penalties:
            row = f"{p:<14}"
            for s in surfaces:
                v0 = df[(df.penalty == p) & (df.surface == s) & (df.distance == 0.0)][metric]
                vm = df[(df.penalty == p) & (df.surface == s) & (df.distance == max_dist)][metric]
                v0_val = v0.values[0] if len(v0) > 0 else float("nan")
                vm_val = vm.values[0] if len(vm) > 0 else float("nan")
                row += f"  {v0_val:>13.6f} {vm_val:>13.6f}"
            print(row)

    # ================================================================
    # CURVATURE CORRECTION VALUE
    # ================================================================
    print(f"\n{'='*70}")
    print("VALUE OF CURVATURE CORRECTION (drift_proj_only vs drift_curvature_corrected)")
    print(f"{'='*70}")

    for s in surfaces:
        print(f"\n{s} at dist={max_dist}:")
        print(f"  {'penalty':<14} {'proj_only':>12} {'corrected':>12} {'improvement':>12}")
        print("  " + "-" * 54)
        for p in penalties:
            proj = df[(df.penalty == p) & (df.surface == s) & (df.distance == max_dist)]["drift_proj_only"]
            corr = df[(df.penalty == p) & (df.surface == s) & (df.distance == max_dist)]["drift_curvature_corrected"]
            if len(proj) > 0 and len(corr) > 0:
                pv = proj.values[0]
                cv = corr.values[0]
                imp = (1 - cv / pv) * 100 if pv > 1e-10 else 0.0
                print(f"  {p:<14} {pv:>12.6f} {cv:>12.6f} {imp:>11.1f}%")


def plot_results(df: pd.DataFrame, output_dir: str):
    """Plot degradation curves per metric category."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not available, skipping plots")
        return

    surfaces = df["surface"].unique()
    metric_groups = {
        "manifold": [("reconstruction", "Reconstruction"), ("tangent", "Tangent Alignment")],
        "diffusion": [("cov_tangent", "Covariance Tangency")],
        "drift": [
            ("drift_proj_only", "Proj-Only Drift"),
            ("drift_curvature_corrected", "Curvature-Corrected Drift"),
            ("curvature_drift_match", "Curvature Drift Match"),
        ],
    }

    for group_name, metrics in metric_groups.items():
        for surface_name in surfaces:
            fig, axes = plt.subplots(1, len(metrics), figsize=(6 * len(metrics), 5))
            if len(metrics) == 1:
                axes = [axes]

            sdf = df[df.surface == surface_name]
            for ax, (metric, label) in zip(axes, metrics):
                for penalty in sdf["penalty"].unique():
                    pdf = sdf[sdf.penalty == penalty].sort_values("distance")
                    ax.plot(pdf["distance"], pdf[metric], marker="o", label=penalty, linewidth=1.5, markersize=4)
                ax.set_xlabel("Extrapolation Distance")
                ax.set_ylabel(label)
                ax.set_title(f"{surface_name}: {label}")
                ax.legend(fontsize=7, ncol=2)
                ax.grid(True, alpha=0.3)
                ax.set_yscale("log")

            plt.tight_layout()
            path = f"{output_dir}/{group_name}_{surface_name}.png"
            plt.savefig(path, dpi=150, bbox_inches="tight")
            plt.close(fig)
            print(f"  Saved {path}")


def main():
    parser = argparse.ArgumentParser(description="AEML Penalty Extrapolation Evaluation")
    parser.add_argument("--surfaces", nargs="+",
                        default=["paraboloid", "monkey_saddle", "sinusoidal"],
                        choices=list(SURFACE_MAP.keys()))
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--n_train", type=int, default=2000)
    parser.add_argument("--n_test", type=int, default=500)
    parser.add_argument("--train_bound", type=float, default=1.0)
    parser.add_argument("--max_dist", type=float, default=0.5)
    parser.add_argument("--dist_step", type=float, default=0.1)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=str, default="penalty_eval_results.csv")
    parser.add_argument("--plot", action="store_true")
    parser.add_argument("--plot_dir", type=str, default="penalty_eval_plots")

    args = parser.parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Surfaces: {args.surfaces}")
    print(f"Penalties: {list(EVAL_PENALTY_CONFIGS.keys())}")
    print(f"Epochs: {args.epochs}")
    print(f"Estimated: {len(EVAL_PENALTY_CONFIGS) * len(args.surfaces)} model trainings")

    df = run_evaluation(
        surfaces=args.surfaces,
        penalty_configs=EVAL_PENALTY_CONFIGS,
        epochs=args.epochs,
        n_train=args.n_train,
        n_test=args.n_test,
        train_bound=args.train_bound,
        max_dist=args.max_dist,
        dist_step=args.dist_step,
        device=device,
        seed=args.seed,
    )

    df.to_csv(args.output, index=False)
    print(f"\nResults saved to {args.output}")

    print_summary(df)

    if args.plot:
        import os
        os.makedirs(args.plot_dir, exist_ok=True)
        plot_results(df, args.plot_dir)


if __name__ == "__main__":
    main()
