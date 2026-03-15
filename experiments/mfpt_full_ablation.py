"""
MFPT ablation: all 4 AE conditions × 2 surfaces × 10 seeds.

Conditions: baseline (no penalties), T, F, T+F
All use encoder-pullback drift.
Radii: 0.5, 1.0, 2.0 (within training domain).

Usage:
    python -m experiments.mfpt_full_ablation --epochs 50 --sde-epochs 50 --n-seeds 1  # smoke
    python -m experiments.mfpt_full_ablation                                            # full
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

# ── Conditions ────────────────────────────────────────────────────────────

# Contractive weight (must match paper_experiments.LAMBDA_C)
LAMBDA_C = 0.1

CONDITIONS = {
    "baseline": LossWeights(),
    "T":        LossWeights(tangent_bundle=1.0),
    "F":        LossWeights(diffeo=1.0),
    "C":        LossWeights(contractive=LAMBDA_C),
    "T+F":      LossWeights(tangent_bundle=1.0, diffeo=1.0),
}

LAMBDA_SMOOTH = 0.0
SURFACES = ["paraboloid", "hyperbolic_paraboloid", "quartic_dome", "sinusoidal"]
RADII = [0.5, 1.0, 2.0, 3.0]
MFPT_N_STEPS = 200  # T=2.0


# ── MFPT ──────────────────────────────────────────────────────────────────

def compute_mfpt_ambient(traj, radii):
    """MFPT to exit ambient ball of radius r around x(0).

    Returns dict: r -> {"mean": float, "std": float, "exit_frac": float}
    """
    B, T1, D = traj.shape
    x0 = traj[:, 0:1]
    dists = ((traj - x0) ** 2).sum(-1).sqrt()  # (B, T+1)

    results = {}
    for r in radii:
        fpt = torch.full((B,), float('nan'), device=traj.device)
        for b in range(B):
            exits = (dists[b] > r).nonzero(as_tuple=True)[0]
            if len(exits) > 0:
                fpt[b] = exits[0].float() * DT
        exited = torch.isfinite(fpt)
        n_ex = exited.sum().item()
        if n_ex >= 5:
            results[r] = {
                "mean": fpt[exited].mean().item(),
                "std": fpt[exited].std().item(),
                "exit_frac": n_ex / B,
            }
        else:
            results[r] = {"mean": float('nan'), "std": float('nan'), "exit_frac": n_ex / B}
    return results


# ── Pipeline ──────────────────────────────────────────────────────────────

def train_pipeline(ae, x, v, Lambda, seed, N, epochs_sde):
    """Train Stage 2 + 3, return pipeline."""
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
        lambda_smooth=LAMBDA_SMOOTH, aug_sigma=0.1,
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


def evaluate_mfpt(pipeline, ae, sde, seed, n_steps):
    """Compute MFPT and W2 for one pipeline."""
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

    # MFPT
    gt_mfpt = compute_mfpt_ambient(gt_traj, RADII)
    lr_mfpt = compute_mfpt_ambient(x_traj, RADII)

    # Also W2 at T=1.0 for reference
    step_1 = int(round(1.0 / DT))
    both_alive = gt_alive[:, step_1] if step_1 < gt_alive.shape[1] else gt_alive[:, -1]
    w2 = compute_w2(x_traj[:, step_1], gt_traj[:, step_1], both_alive, both_alive)

    row = {"W2@1.0": w2}
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


# ── Main ──────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-seeds", type=int, default=10)
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--sde-epochs", type=int, default=300)
    parser.add_argument("--base-seed", type=int, default=42)
    parser.add_argument("--D", type=int, default=11)
    parser.add_argument("--N", type=int, default=50)
    parser.add_argument("--output", type=str, default="mfpt_ablation.csv")
    parser.add_argument("--surfaces", type=str, nargs="+", default=None,
                        help="Surfaces to run (default: all)")
    args = parser.parse_args()

    surfaces = args.surfaces if args.surfaces else SURFACES
    seeds = [args.base_seed + i * 1000 for i in range(args.n_seeds)]
    cond_names = list(CONDITIONS.keys())
    total = len(seeds) * len(surfaces) * len(cond_names)

    print(f"Device: {DEVICE}")
    print(f"D={args.D}, N={args.N}")
    print(f"Surfaces: {surfaces}")
    print(f"Conditions: {cond_names}")
    print(f"Seeds ({len(seeds)}): {seeds}")
    print(f"Radii: {RADII}")
    print(f"Total runs: {total}\n")

    t0 = time.time()
    all_rows = []

    for surface_name in surfaces:
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

            for cond_label, lw in CONDITIONS.items():
                print(f"\n  --- {cond_label} ---")
                torch.manual_seed(seed)
                np.random.seed(seed)

                ae, recon = _train_ae(
                    surface, args.D, seed, args.N, args.epochs, lw, train_data,
                )
                print(f"    recon={recon:.6f}")

                pipeline = train_pipeline(ae, x, v, Lambda, seed, args.N, args.sde_epochs)

                row = evaluate_mfpt(pipeline, ae, sde, seed, MFPT_N_STEPS)
                row.update({
                    "surface": surface_name,
                    "seed": seed,
                    "condition": cond_label,
                    "recon_per_dim": recon,
                })

                for r in RADII:
                    gt_m = row[f"MFPT_gt_r{r}"]
                    lr_m = row[f"MFPT_lr_r{r}"]
                    err = row[f"MFPT_err_r{r}"]
                    print(f"    r={r}: GT={gt_m:.3f} learned={lr_m:.3f} "
                          f"err={err:.1%} exit={row[f'exit_frac_r{r}']:.0%}")
                print(f"    W2@1.0={row['W2@1.0']:.4f}")

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
    print("MFPT ABLATION SUMMARY")
    print(f"{'='*80}")

    for surface_name in surfaces:
        print(f"\n  Surface: {surface_name}")
        sub = df[df["surface"] == surface_name]

        # MFPT error table
        for r in RADII:
            col = f"MFPT_err_r{r}"
            print(f"\n    r={r}:")
            print(f"    {'condition':>12s}  {'MFPT_err':>14s}  {'MFPT_learned':>14s}  {'MFPT_GT':>14s}  {'W2@1.0':>14s}")
            for cond in cond_names:
                cs = sub[sub["condition"] == cond]
                err = cs[col].values
                lr = cs[f"MFPT_lr_r{r}"].values
                gt = cs[f"MFPT_gt_r{r}"].values
                w2 = cs["W2@1.0"].values
                print(f"    {cond:>12s}  {np.nanmean(err):>6.1%}±{np.nanstd(err):>.1%}  "
                      f"{np.nanmean(lr):>6.3f}±{np.nanstd(lr):>.3f}  "
                      f"{np.nanmean(gt):>6.3f}±{np.nanstd(gt):>.3f}  "
                      f"{np.nanmean(w2):>6.3f}±{np.nanstd(w2):>.3f}")

        # Paired t-test: each condition vs baseline
        print(f"\n    Paired t-test vs baseline (MFPT_err, negative = better):")
        for r in RADII:
            col = f"MFPT_err_r{r}"
            print(f"      r={r}:")
            base_seeds = sub[sub["condition"] == "baseline"].sort_values("seed")
            for cond in ["T", "F", "C", "T+F"]:
                cond_seeds = sub[sub["condition"] == cond].sort_values("seed")
                if len(base_seeds) != len(cond_seeds):
                    continue
                base_v = base_seeds[col].values
                cond_v = cond_seeds[col].values
                mask = np.isfinite(base_v) & np.isfinite(cond_v)
                if mask.sum() < 2:
                    print(f"        {cond:>6s}: insufficient data")
                    continue
                diff = cond_v[mask] - base_v[mask]
                t_stat, p_val = stats.ttest_rel(cond_v[mask], base_v[mask])
                delta_pct = diff.mean() / base_v[mask].mean() * 100
                sig = "**" if p_val < 0.01 else "*" if p_val < 0.05 else "+" if p_val < 0.1 else ""
                wins = (cond_v[mask] < base_v[mask]).sum()
                print(f"        {cond:>6s}: Δ={delta_pct:+.1f}%  p={p_val:.3f}{sig}  "
                      f"wins={wins}/{mask.sum()}")


if __name__ == "__main__":
    main()
