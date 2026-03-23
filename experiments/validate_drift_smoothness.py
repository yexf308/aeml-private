"""
Validate drift smoothness regularization on D=3 paraboloid, N=20, 10 seeds.

4 conditions:
  1. baseline     (no K, no smooth)
  2. K only       (K in AE, no smooth)
  3. smooth only  (no K, smooth in Stage 2)
  4. K + smooth   (K in AE, smooth in Stage 2)

For each seed, AE is trained once per config (T+F and T+F+K), then Stage 2+3
are forked with/without smoothing. This halves runtime and eliminates AE
variability between paired conditions.

Usage:
    python -u -m experiments.validate_drift_smoothness --n-seeds 10
    python -u -m experiments.validate_drift_smoothness --n-seeds 1  # smoke test
"""
import argparse
import copy
import time

import numpy as np
import pandas as pd
import torch
from scipy import stats

from src.numeric.datagen import sample_from_manifold
from src.numeric.geometry import curvature_drift_explicit_full, regularized_metric_inverse
from src.numeric.losses import LossWeights
from src.numeric.sde_nets import DriftNet, DiffusionNet
from src.numeric.sde_training import SDEPipelineTrainer

from experiments.data_driven_sde import (
    DEVICE, TRAIN_BOUND, BOUNDARY, N_TRAJ, DT, N_STEPS, LR_SDE,
    BATCH_SIZE, create_manifold_sde, evaluate_pipeline, train_autoencoder,
)
from experiments.trajectory_fidelity_study import lambdify_sde

SURFACE = "paraboloid"
N_TRAIN = 20
EPOCHS_AE = 200
EPOCHS_SDE = 300
LAMBDA_SMOOTH = 0.5
AUG_SIGMA = 0.1

# AE loss configs
NO_K = LossWeights(tangent_bundle=1.0, diffeo=1.0, curvature=0.0)
WITH_K = LossWeights(tangent_bundle=1.0, diffeo=1.0, curvature=0.1)


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


def run_fork(autoencoder, train_data, sde, seed, lambda_smooth, use_metric=True):
    """Run Stage 2+3 with a shared frozen AE, return metrics."""
    x = train_data.samples.to(DEVICE)
    v = train_data.mu.to(DEVICE)
    Lambda = train_data.cov.to(DEVICE)

    d = 2
    torch.manual_seed(seed + 100)
    drift_net = DriftNet(d).to(DEVICE)
    diffusion_net = DiffusionNet(d).to(DEVICE)
    pipeline = SDEPipelineTrainer(autoencoder, drift_net, diffusion_net, device=DEVICE)

    drift_losses = pipeline.train_stage2(
        x, v, Lambda, epochs=EPOCHS_SDE, lr=LR_SDE,
        batch_size=BATCH_SIZE, print_interval=0,
        lambda_smooth=lambda_smooth, aug_sigma=AUG_SIGMA,
        use_metric=use_metric,
    )

    diff_losses = pipeline.train_stage3(
        x, Lambda, epochs=EPOCHS_SDE, lr=LR_SDE,
        batch_size=BATCH_SIZE, print_interval=0,
    )

    results = evaluate_pipeline(pipeline, autoencoder, sde, seed)
    results["E_mu_held"] = compute_drift_error_at_grid(autoencoder, pipeline, sde, DEVICE)
    results["drift_loss_final"] = drift_losses[-1]
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-seeds", type=int, default=10)
    parser.add_argument("--lambda-smooth", type=float, default=LAMBDA_SMOOTH)
    args = parser.parse_args()

    seeds = [42 + i * 1000 for i in range(args.n_seeds)]
    lam = args.lambda_smooth

    # AE configs: (label_prefix, loss_weights, reg_name)
    ae_configs = [
        ("",  NO_K,   "T+F"),
        ("K", WITH_K, "T+F+K"),
    ]
    # Stage 2 configs: (label_suffix, lambda_smooth, use_metric)
    s2_configs = [
        ("",            0.0, True),
        ("+smooth",     lam, True),
        ("+smooth-euc", lam, False),
    ]

    rows = []
    t0 = time.time()

    for seed in seeds:
        manifold_sde = create_manifold_sde(SURFACE)
        train_data = sample_from_manifold(
            manifold_sde,
            [(-TRAIN_BOUND, TRAIN_BOUND), (-TRAIN_BOUND, TRAIN_BOUND)],
            n_samples=N_TRAIN, seed=seed, device=DEVICE,
        )
        sde = lambdify_sde(create_manifold_sde(SURFACE))

        for ae_prefix, ae_lw, ae_reg_name in ae_configs:
            # Train AE once per (seed, AE config)
            autoencoder, ae_loss = train_autoencoder(
                train_data, EPOCHS_AE, ae_lw, ae_reg_name, seed,
            )

            for s2_suffix, lam_s, use_met in s2_configs:
                cond = ae_prefix + ("baseline" if not ae_prefix and not s2_suffix else "")
                if not cond:
                    cond = s2_suffix.lstrip("+")
                if ae_prefix and s2_suffix:
                    cond = ae_prefix + s2_suffix

                print(f"\n{'='*60}")
                print(f"  seed={seed}  condition={cond}  "
                      f"lambda_smooth={lam_s}  metric={use_met}")
                print(f"{'='*60}")
                t_fork = time.time()
                results = run_fork(
                    autoencoder, train_data, sde, seed, lam_s,
                    use_metric=use_met,
                )
                elapsed = time.time() - t_fork

                row = {
                    "seed": seed,
                    "condition": cond,
                    "ae_loss": ae_loss,
                    **results,
                }
                rows.append(row)
                print(f"\n  >> E_mu_held={results['E_mu_held']:.4f}  "
                      f"W2={results['W2@1.0']:.4f}  "
                      f"drift_loss={results['drift_loss_final']:.6f}  "
                      f"({elapsed:.0f}s)")

    df = pd.DataFrame(rows)
    csv_path = "drift_smoothness_validation.csv"
    df.to_csv(csv_path, index=False)
    print(f"\nSaved {len(df)} rows to {csv_path}")

    # Summary helper
    metrics = ["E_mu_held", "W2@1.0", "sW2@1.0", "MTE@1.0"]

    def print_comparison(name, df_a, df_b):
        """Print paired t-test: df_b vs df_a (lower = better)."""
        print(f"\n  {name}:")
        a = df_a.sort_values("seed")
        b = df_b.sort_values("seed")
        for m in metrics:
            if m not in a.columns:
                continue
            av, bv = a[m].values, b[m].values
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
            print(f"    {m:<12} {delta:>+7.1f}%  {n_help}/{len(ac)} wins  p={p:.4f} {sig}")

    print(f"\n{'='*70}")
    print("SUMMARY (paired t-tests, lower = better)")
    print(f"{'='*70}")

    baseline = df[df["condition"] == "baseline"]
    k_only = df[df["condition"] == "K"]
    smooth = df[df["condition"] == "smooth"]
    smooth_euc = df[df["condition"] == "smooth-euc"]
    k_smooth = df[df["condition"] == "K+smooth"]
    k_smooth_euc = df[df["condition"] == "K+smooth-euc"]

    # vs baseline
    for name, cond in [("K", k_only), ("smooth (metric)", smooth),
                        ("smooth-euc", smooth_euc), ("K+smooth", k_smooth)]:
        print_comparison(f"{name} vs baseline", baseline, cond)

    # Key comparisons
    print_comparison("K+smooth vs smooth (does K help?)", smooth, k_smooth)
    print_comparison("smooth vs smooth-euc (does metric help?)", smooth_euc, smooth)
    print_comparison("K+smooth vs K+smooth-euc (does metric help w/ K?)", k_smooth_euc, k_smooth)
    print_comparison("K+smooth vs K (does smooth help K?)", k_only, k_smooth)

    print(f"\nTotal time: {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
