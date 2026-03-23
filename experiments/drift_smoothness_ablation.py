"""
Drift smoothness ablation: isolate the effect of drift smoothness regularization.

For each seed, train ONE T+F AE, then fork into:
  - T+F (no smooth): lambda_smooth = 0.0
  - T+F+S (smooth):  lambda_smooth = 0.5

This pairs the AE so any difference is purely from Stage 2 smoothness.

Usage:
    python -m experiments.drift_smoothness_ablation --epochs 50 --sde-epochs 50 --n-seeds 1  # smoke
    python -m experiments.drift_smoothness_ablation                                            # full
"""

import argparse
import time
import torch
import numpy as np
import pandas as pd
from scipy import stats

from src.numeric.losses import LossWeights
from src.numeric.sde_nets import DriftNet, DiffusionNet
from src.numeric.sde_training import SDEPipelineTrainer

from src.numeric.highd_manifolds import (
    FourierAugmentedSurface,
    sample_from_highd_manifold,
    create_highd_lambdified_sde,
)

from experiments.data_driven_sde import (
    simulate_ground_truth, compute_w2,
    TRAIN_BOUND, BATCH_SIZE, LR_AE, LR_SDE, DEVICE,
    N_TRAJ, DT, BOUNDARY,
)
from experiments.highd_N_D_sweep import (
    hidden_dims_for_D, local_drift_fn, local_diffusion_fn,
    N_EVAL, compute_coefficient_errors,
)
from experiments.highd_baseline_comparison import _train_ae
from experiments.mfpt_full_ablation import compute_mfpt_ambient

# ── Config ────────────────────────────────────────────────────────────────

AE_LW = LossWeights(tangent_bundle=1.0, diffeo=1.0)  # T+F for all AEs
LAMBDA_SMOOTH_VALUES = [0.0, 0.5]  # no smooth vs smooth
AUG_SIGMA = 0.1
SURFACES = ["paraboloid", "hyperbolic_paraboloid", "quartic_dome", "sinusoidal"]
RADII = [0.5, 1.0, 2.0, 3.0]
MFPT_N_STEPS = 200  # T=2.0


def train_sde_fork(ae, x, v, Lambda, seed, N, epochs_sde, lambda_smooth):
    """Train Stage 2 + 3 with given lambda_smooth, return pipeline."""
    d = 2
    bs = min(N, BATCH_SIZE)

    tmp = SDEPipelineTrainer(
        ae, DriftNet(d).to(DEVICE), DiffusionNet(d).to(DEVICE), device=DEVICE,
    )
    z_pre, dphi_pre, _ = tmp.precompute_decoder_derivatives(x)
    b_z_target, g = tmp.precompute_enc_pull_target(x, v, Lambda, dphi_pre)

    torch.manual_seed(seed + 100)
    drift_net = DriftNet(d).to(DEVICE)
    diff_net = DiffusionNet(d).to(DEVICE)
    pipe = SDEPipelineTrainer(ae, drift_net, diff_net, device=DEVICE)
    pipe.train_stage2_regression(
        z_pre, b_z_target, g,
        epochs=epochs_sde, lr=LR_SDE, batch_size=bs,
        print_interval=max(1, epochs_sde // 3),
        lambda_smooth=lambda_smooth, aug_sigma=AUG_SIGMA,
    )

    torch.manual_seed(seed + 200)
    diff_net2 = DiffusionNet(d).to(DEVICE)
    pipe2 = SDEPipelineTrainer(ae, drift_net, diff_net2, device=DEVICE)
    pipe2.train_stage3_precomputed(
        z_pre, dphi_pre, Lambda.to(DEVICE),
        epochs=epochs_sde, lr=LR_SDE, batch_size=bs,
        print_interval=max(1, epochs_sde // 3),
    )
    return pipe2


def evaluate_all(pipeline, ae, sde, seed, n_steps):
    """Compute MTE, W2, MFPT for one pipeline."""
    torch.manual_seed(seed + 999)
    init_local = (torch.rand(N_TRAJ, 2, device=DEVICE) * 2 - 1) * 0.5
    init_ambient = sde.chart(init_local).to(DEVICE)

    torch.manual_seed(seed + 1234)
    dW = torch.randn(N_TRAJ, n_steps, 2, device=DEVICE)

    gt_traj, gt_alive, _ = simulate_ground_truth(
        init_local, sde, n_steps, DT, dW, BOUNDARY * 2,
    )

    with torch.no_grad():
        z0 = ae.encoder(init_ambient)
    _, x_traj = pipeline.simulate(z0, n_steps, DT, dW=dW)

    # W2 and MTE at T=1.0
    step_1 = int(round(1.0 / DT))
    both_alive = gt_alive[:, step_1] if step_1 < gt_alive.shape[1] else gt_alive[:, -1]
    w2 = compute_w2(x_traj[:, step_1], gt_traj[:, step_1], both_alive, both_alive)

    # MTE: mean trajectory error (per-trajectory, then average)
    alive_mask = both_alive
    if alive_mask.sum() > 0:
        diff = (x_traj[:, step_1][alive_mask] - gt_traj[:, step_1][alive_mask])
        mte = diff.norm(dim=-1).mean().item()
    else:
        mte = float('nan')

    row = {"W2@1.0": w2, "MTE@1.0": mte}

    # MFPT
    gt_mfpt = compute_mfpt_ambient(gt_traj, RADII)
    lr_mfpt = compute_mfpt_ambient(x_traj, RADII)

    for r in RADII:
        gt_m = gt_mfpt[r]["mean"]
        lr_m = lr_mfpt[r]["mean"]
        row[f"MFPT_gt_r{r}"] = gt_m
        row[f"MFPT_lr_r{r}"] = lr_m
        if np.isfinite(gt_m) and gt_m > 0 and np.isfinite(lr_m):
            row[f"MFPT_err_r{r}"] = abs(lr_m - gt_m) / gt_m
        else:
            row[f"MFPT_err_r{r}"] = float('nan')
        row[f"exit_frac_r{r}"] = lr_mfpt[r]["exit_frac"]

    return row


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-seeds", type=int, default=10)
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--sde-epochs", type=int, default=300)
    parser.add_argument("--base-seed", type=int, default=42)
    parser.add_argument("--D", type=int, default=11)
    parser.add_argument("--N", type=int, default=50)
    parser.add_argument("--output", type=str, default="drift_smooth_ablation.csv")
    args = parser.parse_args()

    seeds = [args.base_seed + i * 1000 for i in range(args.n_seeds)]
    total = len(seeds) * len(SURFACES) * len(LAMBDA_SMOOTH_VALUES)

    print(f"Device: {DEVICE}")
    print(f"D={args.D}, N={args.N}")
    print(f"Surfaces: {SURFACES}")
    print(f"lambda_smooth values: {LAMBDA_SMOOTH_VALUES}")
    print(f"Seeds ({len(seeds)}): {seeds}")
    print(f"Total runs: {total}\n")

    t0 = time.time()
    all_rows = []

    for surface_name in SURFACES:
        for seed in seeds:
            print(f"\n{'='*60}")
            print(f"  {surface_name} | seed={seed}")
            print(f"{'='*60}")

            torch.manual_seed(seed)
            np.random.seed(seed)

            surface = FourierAugmentedSurface(surface_name, args.D)
            train_data = sample_from_highd_manifold(
                surface, local_drift_fn, local_diffusion_fn,
                [(-TRAIN_BOUND, TRAIN_BOUND), (-TRAIN_BOUND, TRAIN_BOUND)],
                n_samples=args.N, seed=seed, device=DEVICE,
            )
            x = train_data.samples.to(DEVICE)
            v = train_data.mu.to(DEVICE)
            Lambda = train_data.cov.to(DEVICE)
            sde = create_highd_lambdified_sde(surface, local_drift_fn, local_diffusion_fn)

            # Train ONE AE (T+F) per seed — shared across both smoothness conditions
            torch.manual_seed(seed)
            np.random.seed(seed)
            ae, recon = _train_ae(
                surface, args.D, seed, args.N, args.epochs, AE_LW, train_data,
            )
            print(f"  AE recon={recon:.6f}")

            for lam_s in LAMBDA_SMOOTH_VALUES:
                cond_label = f"T+F (S={lam_s})"
                print(f"\n  --- {cond_label} ---")

                pipeline = train_sde_fork(
                    ae, x, v, Lambda, seed, args.N, args.sde_epochs, lam_s,
                )

                row = evaluate_all(pipeline, ae, sde, seed, MFPT_N_STEPS)
                row.update({
                    "surface": surface_name,
                    "seed": seed,
                    "lambda_smooth": lam_s,
                    "condition": cond_label,
                    "recon_per_dim": recon,
                })

                print(f"    MTE@1.0={row['MTE@1.0']:.4f}  W2@1.0={row['W2@1.0']:.4f}")
                for r in RADII:
                    err = row[f"MFPT_err_r{r}"]
                    print(f"    r={r}: err={err:.1%} exit={row[f'exit_frac_r{r}']:.0%}")

                all_rows.append(row)

            # Save incrementally
            pd.DataFrame(all_rows).to_csv(args.output, index=False)

    elapsed = time.time() - t0
    print(f"\n\nTotal time: {elapsed / 60:.1f} min")

    df = pd.DataFrame(all_rows)
    df.to_csv(args.output, index=False)
    print(f"Saved {len(df)} rows to {args.output}")

    # ── Summary ───────────────────────────────────────────────────────
    print(f"\n\n{'='*80}")
    print("DRIFT SMOOTHNESS ABLATION SUMMARY")
    print(f"{'='*80}")

    metrics = ["MTE@1.0", "W2@1.0"] + [f"MFPT_err_r{r}" for r in RADII]

    for surface_name in SURFACES:
        sub = df[df["surface"] == surface_name]
        no_s = sub[sub["lambda_smooth"] == 0.0].sort_values("seed")
        with_s = sub[sub["lambda_smooth"] == 0.5].sort_values("seed")

        print(f"\n  {surface_name}:")
        print(f"    {'metric':>16s}  {'no-S mean':>10s}  {'with-S mean':>10s}  "
              f"{'Δ%':>8s}  {'wins':>5s}  {'p':>8s}")

        for m in metrics:
            a, b = no_s[m].values, with_s[m].values
            mask = np.isfinite(a) & np.isfinite(b)
            if mask.sum() < 2:
                print(f"    {m:>16s}  insufficient data")
                continue
            ac, bc = a[mask], b[mask]
            delta = (bc.mean() - ac.mean()) / ac.mean() * 100
            wins = int((bc < ac).sum())
            _, p = stats.ttest_rel(bc, ac)
            sig = "**" if p < 0.01 else "*" if p < 0.05 else "+" if p < 0.1 else ""
            print(f"    {m:>16s}  {ac.mean():>10.4f}  {bc.mean():>10.4f}  "
                  f"{delta:>+7.1f}%  {wins}/{len(ac)}  p={p:.4f} {sig}")


if __name__ == "__main__":
    main()
