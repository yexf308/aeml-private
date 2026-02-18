"""
Diagnostic: Encoder Coordinate Alignment

Investigates whether the learned encoder output z = encoder(phi(u,v)) aligns
with the true local coordinates (u, v) for Monge-patch surfaces.

Background:
    In the trajectory fidelity study, simulate_learned_ambient evaluates
    TRUE SDE coefficients (defined in true local coords u,v) at the LEARNED
    latent coords z = encoder(x). This is only correct if z ~ (u, v).

    All study surfaces are Monge patches: phi(u,v) = (u, v, h(u,v)), so the
    true chart inverse is projection: (x1, x2, x3) -> (x1, x2). If the
    encoder does not approximately recover this projection, the SDE
    coefficients will be evaluated at wrong coordinates.

Hypothesis:
    Frobenius/geometric regularization encourages the encoder to learn a
    rotated or nonlinearly-warped version of the true coords (since isometric
    reparameterizations preserve the metric). The baseline encoder, with only
    reconstruction loss, may learn a simpler mapping closer to the true
    projection, explaining why baseline outperforms regularized methods in
    learned_ambient mode.

Usage:
    python -m experiments.diag_coord_alignment --surface paraboloid --epochs 500
    python -m experiments.diag_coord_alignment --surface all --epochs 500
"""

import argparse
import math
import torch
import numpy as np
import sympy as sp
from typing import Dict, Tuple

from src.numeric.autoencoders import AutoEncoder
from src.numeric.datagen import sample_from_manifold
from src.numeric.training import ModelConfig, MultiModelTrainer, TrainingConfig
from src.symbolic.manifold_sdes import ManifoldSDE
from src.symbolic.riemannian import RiemannianManifold
from src.symbolic.surfaces import surface

from experiments.common import SURFACE_MAP, PENALTY_CONFIGS
from experiments.trajectory_fidelity_study import lambdify_sde, sample_ring_initial

# Surfaces and penalty configs for this diagnostic
STUDY_SURFACES = ["paraboloid", "hyperbolic_paraboloid", "sinusoidal"]
DIAG_PENALTY_CONFIGS = {
    k: v for k, v in PENALTY_CONFIGS.items() if k in ("baseline", "T+F", "T+F+K")
}


# ============================================================================
# SDE creation (same as trajectory_fidelity_study)
# ============================================================================

def create_manifold_sde(surface_name: str) -> ManifoldSDE:
    """Create manifold SDE with non-trivial dynamics (same as trajectory fidelity study)."""
    u, v = sp.symbols("u v", real=True)
    local_coord, chart = surface(SURFACE_MAP[surface_name], u, v)
    manifold = RiemannianManifold(local_coord, chart)

    local_drift = sp.Matrix([-v, u])
    local_diffusion = sp.Matrix([[1 + u**2 / 4, u + v], [0, 1 + v**2 / 4]])

    return ManifoldSDE(manifold, local_drift=local_drift, local_diffusion=local_diffusion)


# ============================================================================
# Coordinate alignment analysis
# ============================================================================

def fit_affine_transform(
    z: np.ndarray, uv: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, float]:
    """Fit z ~ A @ uv + b using least squares.

    Args:
        z: (N, 2) encoder outputs.
        uv: (N, 2) true local coordinates.

    Returns:
        A: (2, 2) best-fit linear transform.
        b: (2,) best-fit offset.
        residual: mean squared residual after fit.
    """
    N = uv.shape[0]
    # Build design matrix [uv, 1] -> z
    # z_i = A @ uv_i + b  =>  z = [uv | 1] @ [A^T; b^T]
    design = np.hstack([uv, np.ones((N, 1))])  # (N, 3)
    # Solve for each output dim
    # z = design @ params, params shape (3, 2)
    params, _, _, _ = np.linalg.lstsq(design, z, rcond=None)
    A = params[:2, :].T  # (2, 2)
    b = params[2, :]     # (2,)
    z_pred = uv @ A.T + b
    residual = np.mean(np.sum((z - z_pred) ** 2, axis=1))
    return A, b, residual


def extract_rotation_angle(A: np.ndarray) -> float:
    """Extract the rotation angle (degrees) from a 2x2 linear transform via SVD.

    Decomposes A = U @ S @ V^T and extracts the rotation component.
    The total rotation is the composition of U and V, measuring how much
    the coordinate axes are rotated relative to identity.

    Returns angle in degrees in [0, 180].
    """
    U, S, Vt = np.linalg.svd(A)
    # The rotation part: R = U @ Vt
    R = U @ Vt
    # Ensure proper rotation (det = +1); if det = -1, flip a column
    if np.linalg.det(R) < 0:
        U[:, 1] *= -1
        R = U @ Vt
    angle_rad = np.arctan2(R[1, 0], R[0, 0])
    angle_deg = np.degrees(angle_rad) % 360
    # Normalize to [0, 180] (rotation by 190 deg is same as 170 deg for alignment)
    if angle_deg > 180:
        angle_deg = 360 - angle_deg
    return angle_deg


def compute_r_squared(z: np.ndarray, uv: np.ndarray) -> float:
    """Compute R^2 between z and uv (multivariate).

    R^2 = 1 - SS_res / SS_tot, where both are summed over all output dims.
    """
    ss_tot = np.sum((uv - uv.mean(axis=0)) ** 2)
    ss_res = np.sum((z - uv) ** 2)
    if ss_tot < 1e-12:
        return float("nan")
    return 1.0 - ss_res / ss_tot


def compute_r_squared_affine(z: np.ndarray, uv: np.ndarray) -> float:
    """Compute R^2 of the best affine fit z ~ A @ uv + b.

    This measures how well z can be explained by a linear transform of uv.
    """
    A, b, _ = fit_affine_transform(z, uv)
    z_pred = uv @ A.T + b
    ss_tot = np.sum((z - z.mean(axis=0)) ** 2)
    ss_res = np.sum((z - z_pred) ** 2)
    if ss_tot < 1e-12:
        return float("nan")
    return 1.0 - ss_res / ss_tot


def compute_sde_coeff_errors(
    z: torch.Tensor,
    uv: torch.Tensor,
    sde,
) -> Tuple[float, float]:
    """Compute relative errors of SDE coefficients at z vs at uv.

    Args:
        z: (N, 2) encoder outputs.
        uv: (N, 2) true local coordinates.
        sde: LambdifiedSDE instance.

    Returns:
        drift_rel_err: relative L2 error of drift(z) vs drift(uv).
        diff_rel_err: relative Frobenius error of diffusion(z) vs diffusion(uv).
    """
    drift_true = sde.local_drift(uv)       # (N, 2)
    drift_z = sde.local_drift(z)            # (N, 2)
    diff_true = sde.local_diffusion(uv)     # (N, 2, 2)
    diff_z = sde.local_diffusion(z)         # (N, 2, 2)

    # Drift relative error
    drift_diff = torch.norm(drift_z - drift_true, dim=1)  # (N,)
    drift_norm = torch.norm(drift_true, dim=1)             # (N,)
    # Avoid division by zero
    valid_drift = drift_norm > 1e-8
    if valid_drift.sum() > 0:
        drift_rel = (drift_diff[valid_drift] / drift_norm[valid_drift]).mean().item()
    else:
        drift_rel = float("nan")

    # Diffusion relative error (Frobenius)
    diff_diff = torch.norm(diff_z - diff_true, dim=(1, 2))  # (N,)
    diff_norm = torch.norm(diff_true, dim=(1, 2))            # (N,)
    valid_diff = diff_norm > 1e-8
    if valid_diff.sum() > 0:
        diff_rel = (diff_diff[valid_diff] / diff_norm[valid_diff]).mean().item()
    else:
        diff_rel = float("nan")

    return drift_rel, diff_rel


# ============================================================================
# Main diagnostic
# ============================================================================

def run_diagnostic(
    surface_name: str,
    train_bound: float = 1.0,
    n_train: int = 2000,
    n_test: int = 500,
    epochs: int = 500,
    device: str = "cpu",
    seed: int = 42,
):
    """Run coordinate alignment diagnostic for one surface."""
    print(f"\n{'='*72}")
    print(f"  Coordinate Alignment Diagnostic: {surface_name}")
    print(f"  Train: [-{train_bound}, {train_bound}]^2, n_train={n_train}, epochs={epochs}")
    print(f"{'='*72}")

    # Create manifold SDE and lambdify
    manifold_sde = create_manifold_sde(surface_name)
    sde = lambdify_sde(manifold_sde)

    # Sample training data
    print("\n  Sampling training data...")
    train_data = sample_from_manifold(
        manifold_sde,
        [(-train_bound, train_bound), (-train_bound, train_bound)],
        n_samples=n_train,
        seed=seed,
        device=device,
    )

    # Sample test points: training domain
    print("  Sampling test points (training domain)...")
    torch.manual_seed(seed + 5000)
    uv_train_test = (torch.rand(n_test, 2, device=device) * 2 - 1) * train_bound

    # Sample test points: extrapolation ring [1.0, 1.5]
    print("  Sampling test points (extrapolation ring)...")
    uv_extrap_test = sample_ring_initial(
        n_test, inner=1.0, outer=1.5, device=device, seed=seed + 6000,
    )
    n_extrap = len(uv_extrap_test)
    print(f"    Got {n_extrap} points in [1.0, 1.5] ring")

    # Map test points to ambient space via true chart
    x_train_test = sde.chart(uv_train_test).to(device)
    x_extrap_test = sde.chart(uv_extrap_test).to(device)

    # Train models
    results = {}
    for penalty_name, loss_weights in DIAG_PENALTY_CONFIGS.items():
        print(f"\n  --- Training: {penalty_name} ---")

        trainer = MultiModelTrainer(TrainingConfig(
            epochs=epochs,
            n_samples=n_train,
            input_dim=3,
            hidden_dim=64,
            latent_dim=2,
            learning_rate=0.005,
            batch_size=32,
            test_size=0.03,
            print_interval=max(1, epochs // 5),
            device=device,
        ))
        trainer.add_model(ModelConfig(name=penalty_name, loss_weights=loss_weights))

        data_loader = trainer.create_data_loader(train_data)
        for epoch in range(epochs):
            losses = trainer.train_epoch(data_loader)
            if (epoch + 1) % max(1, epochs // 5) == 0:
                print(f"    Epoch {epoch+1}: {losses[penalty_name]:.6f}")

        model = trainer.models[penalty_name]
        model.eval()

        # Analyze on both domains
        for domain_name, uv_test, x_test in [
            ("train", uv_train_test, x_train_test),
            ("extrap", uv_extrap_test, x_extrap_test),
        ]:
            with torch.no_grad():
                z = model.encoder(x_test)  # (N, 2)

            z_np = z.detach().cpu().numpy()
            uv_np = uv_test.detach().cpu().numpy()

            # 1. Direct comparison: z vs (u, v)
            mae = np.mean(np.abs(z_np - uv_np))
            r2_direct = compute_r_squared(z_np, uv_np)

            # 2. Affine fit: z ~ A @ (u,v) + b
            A, b_vec, affine_residual = fit_affine_transform(z_np, uv_np)
            r2_affine = compute_r_squared_affine(z_np, uv_np)
            rotation_deg = extract_rotation_angle(A)

            # 3. Singular values of A (scaling factors)
            _, svd_s, _ = np.linalg.svd(A)

            # 4. SDE coefficient errors
            drift_rel, diff_rel = compute_sde_coeff_errors(z.detach(), uv_test, sde)

            results[(penalty_name, domain_name)] = {
                "mae": mae,
                "r2_direct": r2_direct,
                "r2_affine": r2_affine,
                "rotation_deg": rotation_deg,
                "svd_s1": svd_s[0],
                "svd_s2": svd_s[1],
                "affine_residual": affine_residual,
                "drift_rel_err": drift_rel,
                "diff_rel_err": diff_rel,
                "A": A,
                "b": b_vec,
            }

    # Print summary tables
    _print_summary(surface_name, results)
    return results


def _print_summary(surface_name: str, results: Dict):
    """Print diagnostic summary tables."""
    penalties = list(DIAG_PENALTY_CONFIGS.keys())

    for domain_label, domain_key, domain_desc in [
        ("train [-1,1]^2", "train", "Training domain"),
        ("extrap [1.0, 1.5] ring", "extrap", "Extrapolation ring"),
    ]:
        print(f"\n  Surface: {surface_name}")
        print(f"  Domain: {domain_label}")
        print(f"  {'':->90}")
        header = (
            f"  {'Penalty':<10s} | {'z-uv MAE':>10s} | {'z-uv R2':>9s} | "
            f"{'Affine R2':>10s} | {'Rot (deg)':>10s} | "
            f"{'Drift Err':>10s} | {'Diff Err':>10s}"
        )
        print(header)
        print(f"  {'':->90}")

        for pen in penalties:
            key = (pen, domain_key)
            if key not in results:
                continue
            r = results[key]
            print(
                f"  {pen:<10s} | {r['mae']:>10.5f} | {r['r2_direct']:>9.4f} | "
                f"{r['r2_affine']:>10.4f} | {r['rotation_deg']:>10.2f} | "
                f"{r['drift_rel_err']:>10.4f} | {r['diff_rel_err']:>10.4f}"
            )

        print(f"  {'':->90}")

        # Print affine transform details
        print(f"\n  Affine transform details (z ~ A @ [u,v]^T + b):")
        for pen in penalties:
            key = (pen, domain_key)
            if key not in results:
                continue
            r = results[key]
            A = r["A"]
            b_vec = r["b"]
            print(f"    {pen:<10s}: A = [[{A[0,0]:+.4f}, {A[0,1]:+.4f}], "
                  f"[{A[1,0]:+.4f}, {A[1,1]:+.4f}]]  "
                  f"b = [{b_vec[0]:+.4f}, {b_vec[1]:+.4f}]  "
                  f"sv = [{r['svd_s1']:.4f}, {r['svd_s2']:.4f}]  "
                  f"resid = {r['affine_residual']:.6f}")

        print()


# ============================================================================
# CLI
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Diagnostic: Encoder Coordinate Alignment for Trajectory Fidelity"
    )
    parser.add_argument(
        "--surface", type=str, default="paraboloid",
        choices=STUDY_SURFACES + ["all"],
        help="Which surface(s) to diagnose (default: paraboloid)",
    )
    parser.add_argument(
        "--epochs", type=int, default=500,
        help="Training epochs (default: 500)",
    )
    parser.add_argument(
        "--device", type=str, default=None,
        help="Torch device (default: auto-detect)",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed (default: 42)",
    )

    args = parser.parse_args()
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    surfaces = STUDY_SURFACES if args.surface == "all" else [args.surface]
    all_results = {}

    for surf in surfaces:
        results = run_diagnostic(
            surface_name=surf,
            epochs=args.epochs,
            device=device,
            seed=args.seed,
        )
        all_results[surf] = results

    # Final cross-surface summary
    if len(surfaces) > 1:
        print("\n" + "=" * 72)
        print("  CROSS-SURFACE SUMMARY")
        print("=" * 72)

        penalties = list(DIAG_PENALTY_CONFIGS.keys())

        for domain_label, domain_key in [
            ("train [-1,1]^2", "train"),
            ("extrap [1.0, 1.5] ring", "extrap"),
        ]:
            print(f"\n  Domain: {domain_label}")
            print(f"  {'':->100}")
            header = (
                f"  {'Surface':<25s} | {'Penalty':<10s} | {'z-uv MAE':>10s} | "
                f"{'Rot (deg)':>10s} | {'Drift Err':>10s} | {'Diff Err':>10s} | "
                f"{'Affine R2':>10s}"
            )
            print(header)
            print(f"  {'':->100}")

            for surf in surfaces:
                for pen in penalties:
                    key = (pen, domain_key)
                    r = all_results[surf].get(key)
                    if r is None:
                        continue
                    print(
                        f"  {surf:<25s} | {pen:<10s} | {r['mae']:>10.5f} | "
                        f"{r['rotation_deg']:>10.2f} | {r['drift_rel_err']:>10.4f} | "
                        f"{r['diff_rel_err']:>10.4f} | {r['r2_affine']:>10.4f}"
                    )
                # Visual separator between surfaces
                if surf != surfaces[-1]:
                    print(f"  {'':->100}")

            print(f"  {'':->100}")

    print("\nDone.")


if __name__ == "__main__":
    main()
