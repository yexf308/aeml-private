"""
Generate trajectory fidelity data with drift smoothness regularization.

For each surface × seed:
  1. Train T+F AE (500 epochs, matching paper settings)
  2. Fork into {no-smooth, smooth} for Stage 2 (300 epochs each)
  3. Evaluate trajectory metrics via evaluate_pipeline()

This produces T+F and T+F+S columns for the paper's trajectory table.
The "no-smooth" fork should reproduce existing T+F numbers (same AE, same eval).

Usage:
    # Smoke test
    python -u -m experiments.trajectory_smoothing_study --n-seeds 1 --epochs 50 --sde-epochs 50 --surfaces paraboloid

    # Full run (all 4 surfaces, 10 seeds)
    python -u -m experiments.trajectory_smoothing_study --n-seeds 10
"""
import argparse
import time

import numpy as np
import pandas as pd
import torch

from src.numeric.datagen import sample_from_manifold
from src.numeric.geometry import curvature_drift_explicit_full, regularized_metric_inverse
from src.numeric.losses import LossWeights
from src.numeric.sde_nets import DriftNet, DiffusionNet
from src.numeric.sde_training import SDEPipelineTrainer

from experiments.data_driven_sde import (
    DEVICE, TRAIN_BOUND, BOUNDARY, N_TRAJ, DT, N_STEPS, LR_AE, LR_SDE,
    BATCH_SIZE, create_manifold_sde, evaluate_pipeline, train_autoencoder,
)
from experiments.trajectory_fidelity_study import lambdify_sde

SURFACES = ["paraboloid", "hyperbolic_paraboloid", "sinusoidal", "quartic_dome"]
N_TRAIN = 20
LAMBDA_SMOOTH = 0.5
AUG_SIGMA = 0.1

# T+F only (matching paper's AE for T+F column)
AE_LOSS_WEIGHTS = LossWeights(tangent_bundle=1.0, diffeo=1.0)


def compute_drift_error_at_grid(autoencoder, pipeline, sde, device):
    """E_mu of the learned drift_net at a held-out uniform grid."""
    s = torch.linspace(-0.9, 0.9, 10, device=device)
    grid = torch.stack(torch.meshgrid(s, s, indexing='ij'), dim=-1).reshape(-1, 2)

    b_true = sde.ambient_drift(grid.detach()).detach()

    with torch.no_grad():
        x_grid = sde.chart(grid)
        z = autoencoder.encoder(x_grid)

    dphi = torch.func.vmap(torch.func.jacrev(autoencoder.decoder))(z)

    with torch.no_grad():
        d2phi = autoencoder.decoder.hessian_network(z)
        g = dphi.mT @ dphi
        ginv = regularized_metric_inverse(g)
        pinv = ginv @ dphi.mT
        P_hat = dphi @ pinv
        P_hat = 0.5 * (P_hat + P_hat.mT)

        Lambda_true = sde.ambient_covariance(grid.detach()).detach()
        Lambda_tan = P_hat @ Lambda_true @ P_hat
        Lambda_tan = 0.5 * (Lambda_tan + Lambda_tan.mT)
        Sigma_z = pinv @ Lambda_tan @ pinv.mT
        Sigma_z = 0.5 * (Sigma_z + Sigma_z.mT)
        q = curvature_drift_explicit_full(d2phi, Sigma_z)

        b_z = pipeline.drift_net(z)
        dphi_bz = (dphi @ b_z.unsqueeze(-1)).squeeze(-1)
        b_pred = dphi_bz + q

    D = b_true.shape[-1]
    err = ((b_pred - b_true) ** 2).sum(-1) / D
    return err.median().item()


def run_fork(ae, train_data, sde, seed, lambda_smooth, epochs_sde):
    """Run Stage 2+3 with shared frozen AE."""
    x = train_data.samples.to(DEVICE)
    v = train_data.mu.to(DEVICE)
    Lambda = train_data.cov.to(DEVICE)
    batch_size = min(N_TRAIN, BATCH_SIZE)

    d = 2
    torch.manual_seed(seed + 100)
    drift_net = DriftNet(d).to(DEVICE)
    diffusion_net = DiffusionNet(d).to(DEVICE)
    pipeline = SDEPipelineTrainer(ae, drift_net, diffusion_net, device=DEVICE)

    drift_losses = pipeline.train_stage2(
        x, v, Lambda, epochs=epochs_sde, lr=LR_SDE,
        batch_size=batch_size, print_interval=0,
        lambda_smooth=lambda_smooth, aug_sigma=AUG_SIGMA,
        use_metric=False,  # Euclidean — shown equivalent to metric
    )

    diff_losses = pipeline.train_stage3(
        x, Lambda, epochs=epochs_sde, lr=LR_SDE,
        batch_size=batch_size, print_interval=0,
    )

    results = evaluate_pipeline(pipeline, ae, sde, seed)
    results["E_mu_held"] = compute_drift_error_at_grid(ae, pipeline, sde, DEVICE)
    results["drift_loss_final"] = drift_losses[-1]
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-seeds", type=int, default=10)
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--sde-epochs", type=int, default=300)
    parser.add_argument("--lambda-smooth", type=float, default=LAMBDA_SMOOTH)
    parser.add_argument("--surfaces", type=str, nargs="+", default=SURFACES)
    args = parser.parse_args()

    seeds = [42 + i * 1000 for i in range(args.n_seeds)]
    lam = args.lambda_smooth

    # Conditions: (name, lambda_smooth)
    conditions = [
        ("T+F",   0.0),
        ("T+F+S", lam),
    ]

    rows = []
    t0 = time.time()

    for surface_name in args.surfaces:
        print(f"\n{'#'*70}")
        print(f"  Surface: {surface_name}")
        print(f"{'#'*70}")

        for seed in seeds:
            manifold_sde = create_manifold_sde(surface_name)
            train_data = sample_from_manifold(
                manifold_sde,
                [(-TRAIN_BOUND, TRAIN_BOUND), (-TRAIN_BOUND, TRAIN_BOUND)],
                n_samples=N_TRAIN, seed=seed, device=DEVICE,
            )
            sde = lambdify_sde(create_manifold_sde(surface_name))

            # Train AE once per (surface, seed) — T+F only
            ae, ae_loss = train_autoencoder(
                train_data, args.epochs, AE_LOSS_WEIGHTS, "T+F", seed,
            )

            for cond_name, lam_s in conditions:
                print(f"\n  {surface_name} seed={seed} {cond_name} (lambda={lam_s})")
                t_fork = time.time()

                results = run_fork(
                    ae, train_data, sde, seed, lam_s, args.sde_epochs,
                )
                elapsed = time.time() - t_fork

                row = {
                    "surface": surface_name,
                    "config": cond_name,
                    "seed": seed,
                    "ae_loss": ae_loss,
                    **results,
                }
                rows.append(row)
                print(f"    E_mu_held={results['E_mu_held']:.4f}  "
                      f"W2={results['W2@1.0']:.4f}  "
                      f"sW2={results['sW2@1.0']:.4f}  "
                      f"MTE={results['MTE@1.0']:.4f}  ({elapsed:.0f}s)")

    df = pd.DataFrame(rows)
    csv_path = "paper_trajectory_smoothing.csv"
    df.to_csv(csv_path, index=False)
    print(f"\nSaved {len(df)} rows to {csv_path}")

    # Summary per surface
    from scipy import stats

    metrics = ["E_mu_held", "W2@1.0", "sW2@1.0", "MTE@1.0"]

    for surface_name in args.surfaces:
        print(f"\n{'='*70}")
        print(f"  {surface_name}")
        print(f"{'='*70}")

        df_s = df[df["surface"] == surface_name]
        tf = df_s[df_s["config"] == "T+F"].sort_values("seed")
        tfs = df_s[df_s["config"] == "T+F+S"].sort_values("seed")

        print(f"\n  T+F+S vs T+F:")
        for m in metrics:
            if m not in tf.columns:
                continue
            av, bv = tf[m].values, tfs[m].values
            if len(av) != len(bv) or len(av) < 2:
                continue
            mask = np.isfinite(av) & np.isfinite(bv)
            if mask.sum() < 2:
                continue
            ac, bc = av[mask], bv[mask]
            delta = (bc.mean() - ac.mean()) / ac.mean() * 100
            n_help = int((bc < ac).sum())
            _, p = stats.ttest_rel(bc, ac)
            sig = "**" if p < 0.01 else ("*" if p < 0.05 else ("+" if p < 0.1 else ""))
            print(f"    {m:<12} mean: {ac.mean():.3f} → {bc.mean():.3f}  "
                  f"{delta:>+7.1f}%  {n_help}/{len(ac)} wins  p={p:.4f} {sig}")

    print(f"\nTotal time: {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
