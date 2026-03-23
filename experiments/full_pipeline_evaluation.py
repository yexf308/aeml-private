"""
Full pipeline evaluation: chart + learned drift/diffusion nets.

Trains baseline, T+F, T+F+K autoencoders, then Stage 2+3 for each,
and evaluates trajectory metrics using the LEARNED SDE nets (not oracle).

This produces the correct end-to-end trajectory statistics for the paper.
Matches the paper's AE training (single-phase, 500 epochs).

Usage:
    # Smoke test
    python -u -m experiments.full_pipeline_evaluation --n-seeds 1 --epochs 50 --sde-epochs 50 --surfaces paraboloid

    # Full run
    python -u -m experiments.full_pipeline_evaluation --n-seeds 10
"""
import argparse
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
    DEVICE, TRAIN_BOUND, LR_SDE, BATCH_SIZE, N_TRAIN,
    create_manifold_sde, evaluate_pipeline, train_autoencoder,
)
from experiments.trajectory_fidelity_study import lambdify_sde

SURFACES = ["paraboloid", "hyperbolic_paraboloid", "sinusoidal", "quartic_dome"]

LAMBDA_SMOOTH = 0.5
AUG_SIGMA = 0.1

# AE configs (Stage 1 weights). K+S reuses the T+F+K AE.
CONFIGS = {
    "baseline": LossWeights(),
    "T+F":      LossWeights(tangent_bundle=1.0, diffeo=1.0),
    "T+F+K":    LossWeights(tangent_bundle=1.0, diffeo=1.0, curvature=0.1),
}


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


def run_sde_pipeline(ae, train_data, sde, seed, epochs_sde,
                     lambda_smooth=0.0, aug_sigma=AUG_SIGMA):
    """Run Stage 2+3 with frozen AE, evaluate full pipeline."""
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
        lambda_smooth=lambda_smooth, aug_sigma=aug_sigma,
    )

    diff_losses = pipeline.train_stage3(
        x, Lambda, epochs=epochs_sde, lr=LR_SDE,
        batch_size=batch_size, print_interval=0,
    )

    results = evaluate_pipeline(pipeline, ae, sde, seed)
    results["E_mu_held"] = compute_drift_error_at_grid(ae, pipeline, sde, DEVICE)
    results["drift_loss_final"] = drift_losses[-1]
    results["diff_loss_final"] = diff_losses[-1]
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-seeds", type=int, default=10)
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--sde-epochs", type=int, default=300)
    parser.add_argument("--surfaces", type=str, nargs="+", default=SURFACES)
    parser.add_argument("--lambda-smooth", type=float, default=LAMBDA_SMOOTH)
    parser.add_argument("--output", type=str, default="paper_full_pipeline.csv")
    args = parser.parse_args()

    seeds = [42 + i * 1000 for i in range(args.n_seeds)]
    # Conditions: baseline, T+F, T+F+K, T+F+K+S
    # K+S reuses the T+F+K AE, only Stage 2 differs (lambda_smooth > 0)
    conditions = ["baseline", "T+F", "T+F+K", "T+F+K+S"]

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

            # Train each AE once; K+S reuses the T+F+K autoencoder
            ae_cache = {}
            for cond in conditions:
                ae_key = "T+F+K" if cond == "T+F+K+S" else cond
                print(f"\n  {surface_name} seed={seed} {cond}")
                t_start = time.time()

                if ae_key not in ae_cache:
                    lw = CONFIGS[ae_key]
                    ae, ae_loss = train_autoencoder(
                        train_data, args.epochs, lw, ae_key, seed,
                    )
                    ae_cache[ae_key] = (ae, ae_loss)
                else:
                    print(f"    (reusing {ae_key} AE)")

                ae, ae_loss = ae_cache[ae_key]
                lam_s = args.lambda_smooth if cond == "T+F+K+S" else 0.0

                results = run_sde_pipeline(
                    ae, train_data, sde, seed, args.sde_epochs,
                    lambda_smooth=lam_s,
                )
                elapsed = time.time() - t_start

                row = {
                    "surface": surface_name,
                    "config": cond,
                    "seed": seed,
                    "ae_loss": ae_loss,
                    **results,
                }
                rows.append(row)
                print(f"    W2={results['W2@1.0']:.4f}  "
                      f"sW2={results['sW2@1.0']:.4f}  "
                      f"MTE={results['MTE@1.0']:.4f}  "
                      f"E_mu={results['E_mu_held']:.4f}  ({elapsed:.0f}s)")

    df = pd.DataFrame(rows)
    df.to_csv(args.output, index=False)
    print(f"\nSaved {len(df)} rows to {args.output}")

    # Summary
    metrics = ["W2@1.0", "sW2@1.0", "MTE@1.0", "E_mu_held"]

    def paired_test(name, df_a, df_b):
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
            print(f"    {m:<12} {ac.mean():.3f} → {bc.mean():.3f}  "
                  f"{delta:>+7.1f}%  {n_help}/{len(ac)} wins  p={p:.4f} {sig}")

    for surface_name in args.surfaces:
        print(f"\n{'='*70}")
        print(f"  {surface_name}")
        print(f"{'='*70}")

        df_s = df[df["surface"] == surface_name]
        bl = df_s[df_s["config"] == "baseline"]
        tf = df_s[df_s["config"] == "T+F"]
        tfk = df_s[df_s["config"] == "T+F+K"]
        tfks = df_s[df_s["config"] == "T+F+K+S"]

        if len(bl) > 0 and len(tf) > 0:
            paired_test("T+F vs baseline", bl, tf)
        if len(tf) > 0 and len(tfk) > 0:
            paired_test("T+F+K vs T+F", tf, tfk)
        if len(bl) > 0 and len(tfk) > 0:
            paired_test("T+F+K vs baseline", bl, tfk)
        if len(bl) > 0 and len(tfks) > 0:
            paired_test("T+F+K+S vs baseline", bl, tfks)
        if len(tfk) > 0 and len(tfks) > 0:
            paired_test("T+F+K+S vs T+F+K", tfk, tfks)

    print(f"\nTotal time: {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
