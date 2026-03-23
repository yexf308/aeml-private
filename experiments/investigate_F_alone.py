"""
Investigate why F-alone shows '---' at D=201 in the MFPT table.

Train F-alone AE at D=201 on paraboloid, 5 seeds.
After training, measure: σ_min(Dφ), reconstruction error, tangent-space error,
coefficient errors ε_μ and ε_Σ, exit fractions at r∈{0.5,1,2,3}.

Compare against baseline and T+F using the same metrics.

Usage:
    python -m experiments.investigate_F_alone --n-seeds 1 --epochs 50 --sde-epochs 50  # smoke
    python -m experiments.investigate_F_alone                                            # full
"""

import argparse
import time
import torch
import numpy as np
import pandas as pd

from src.numeric.losses import LossWeights
from src.numeric.geometry import compute_sigma_min
from src.numeric.performance_stats import compute_losses_per_sample
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
from experiments.mfpt_full_ablation import (
    train_pipeline, compute_mfpt_ambient, RADII, MFPT_N_STEPS,
)

# ── Conditions ──────────────────────────────────────────────────────────

CONDITIONS = {
    "baseline": LossWeights(),
    "F":        LossWeights(diffeo=1.0),
    "T+F":      LossWeights(tangent_bundle=1.0, diffeo=1.0),
}

D = 201
N = 50
SURFACE = "paraboloid"


# ── Main ────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-seeds", type=int, default=5)
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--sde-epochs", type=int, default=300)
    parser.add_argument("--base-seed", type=int, default=42)
    parser.add_argument("--output", type=str, default="investigate_F_alone.csv")
    args = parser.parse_args()

    seeds = [args.base_seed + i * 1000 for i in range(args.n_seeds)]

    print(f"Device: {DEVICE}")
    print(f"D={D}, N={N}, Surface={SURFACE}")
    print(f"Conditions: {list(CONDITIONS.keys())}")
    print(f"Seeds ({len(seeds)}): {seeds}")
    print(f"Radii: {RADII}\n")

    t0 = time.time()
    all_rows = []

    surface = FourierAugmentedSurface(SURFACE, D)
    sde = create_highd_lambdified_sde(surface, local_drift_fn, local_diffusion_fn)

    for seed in seeds:
        print(f"\n{'='*60}")
        print(f"  seed={seed}")
        print(f"{'='*60}")

        torch.manual_seed(seed)
        np.random.seed(seed)

        train_data = sample_from_highd_manifold(
            surface, local_drift_fn, local_diffusion_fn,
            [(-TRAIN_BOUND, TRAIN_BOUND), (-TRAIN_BOUND, TRAIN_BOUND)],
            n_samples=N, seed=seed, device=DEVICE,
        )
        x = train_data.samples.to(DEVICE)
        v = train_data.mu.to(DEVICE)
        Lambda = train_data.cov.to(DEVICE)

        for cond_label, lw in CONDITIONS.items():
            print(f"\n  --- {cond_label} ---")
            torch.manual_seed(seed)
            np.random.seed(seed)

            ae, recon = _train_ae(surface, D, seed, N, args.epochs, lw, train_data)
            print(f"    recon={recon:.6f}")

            # σ_min diagnostic (in-distribution + extrapolation)
            torch.manual_seed(seed + 7000)
            eval_uv_in = (torch.rand(500, 2, device=DEVICE) * 2 - 1) * TRAIN_BOUND
            x_in = sde.chart(eval_uv_in).to(DEVICE)
            sigma_min_in = compute_sigma_min(ae, x_in)

            torch.manual_seed(seed + 8000)
            eval_uv_ext = (torch.rand(200, 2, device=DEVICE) * 2 - 1) * (TRAIN_BOUND + 0.1)
            x_ext = sde.chart(eval_uv_ext).to(DEVICE)
            sigma_min_ext = compute_sigma_min(ae, x_ext)

            print(f"    σ_min in-dist: min={sigma_min_in.min():.4f} "
                  f"p5={sigma_min_in.quantile(0.05):.4f} "
                  f"median={sigma_min_in.median():.4f}")
            print(f"    σ_min extrap:  min={sigma_min_ext.min():.4f} "
                  f"p5={sigma_min_ext.quantile(0.05):.4f} "
                  f"median={sigma_min_ext.median():.4f}")

            # Tangent-space error: ½||P̂ - P||²_F
            torch.manual_seed(seed + 9000)
            test_data = sample_from_highd_manifold(
                surface, local_drift_fn, local_diffusion_fn,
                [(-TRAIN_BOUND, TRAIN_BOUND), (-TRAIN_BOUND, TRAIN_BOUND)],
                n_samples=500, seed=seed + 9000, device=DEVICE,
            )
            losses_df = compute_losses_per_sample(ae, test_data)
            tangent_err = losses_df["tangent"].mean()
            print(f"    tangent_err={tangent_err:.6f}")

            # Coefficient errors
            torch.manual_seed(seed + 5000)
            eval_uv = (torch.rand(N_EVAL, 2, device=DEVICE) * 2 - 1) * TRAIN_BOUND
            e_mu, e_sigma = compute_coefficient_errors(ae, sde, surface, eval_uv, DEVICE)
            print(f"    E_mu={e_mu:.4f}  E_Sigma={e_sigma:.4f}")

            # Full pipeline: train SDE stages and compute MFPT
            pipeline = train_pipeline(ae, x, v, Lambda, seed, N, args.sde_epochs)

            torch.manual_seed(seed + 999)
            init_local = (torch.rand(N_TRAJ, 2, device=DEVICE) * 2 - 1) * 0.5
            init_ambient = sde.chart(init_local).to(DEVICE)

            torch.manual_seed(seed + 1234)
            dW = torch.randn(N_TRAJ, MFPT_N_STEPS, 2, device=DEVICE)

            gt_traj, gt_alive, _ = simulate_ground_truth(
                init_local, sde, MFPT_N_STEPS, DT, dW, BOUNDARY * 2,
            )

            with torch.no_grad():
                z0 = ae.encoder(init_ambient)
            _, x_traj = pipeline.simulate(z0, MFPT_N_STEPS, DT, dW=dW,
                                            boundary=BOUNDARY * 2)

            gt_mfpt = compute_mfpt_ambient(gt_traj, RADII)
            lr_mfpt = compute_mfpt_ambient(x_traj, RADII)

            row = {
                "condition": cond_label,
                "seed": seed,
                "recon_per_dim": recon,
                "tangent_err": tangent_err,
                "E_mu": e_mu,
                "E_Sigma": e_sigma,
                "sigma_min_in_min": sigma_min_in.min().item(),
                "sigma_min_in_p5": sigma_min_in.quantile(0.05).item(),
                "sigma_min_in_med": sigma_min_in.median().item(),
                "sigma_min_ext_min": sigma_min_ext.min().item(),
                "sigma_min_ext_p5": sigma_min_ext.quantile(0.05).item(),
                "sigma_min_ext_med": sigma_min_ext.median().item(),
            }

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

            for r in RADII:
                gt_m = row[f"MFPT_gt_r{r}"]
                lr_m = row[f"MFPT_lr_r{r}"]
                err = row[f"MFPT_err_r{r}"]
                print(f"    r={r}: GT={gt_m:.3f} learned={lr_m:.3f} "
                      f"err={err:.1%} exit={row[f'exit_frac_r{r}']:.0%}")

            all_rows.append(row)

        # Save incrementally
        pd.DataFrame(all_rows).to_csv(args.output, index=False)

    elapsed = time.time() - t0
    print(f"\n\nTotal time: {elapsed / 60:.1f} min")

    df = pd.DataFrame(all_rows)
    df.to_csv(args.output, index=False)
    print(f"Saved {len(df)} rows to {args.output}")

    # ── Summary ──────────────────────────────────────────────────────
    print(f"\n\n{'='*80}")
    print("F-ALONE INVESTIGATION SUMMARY")
    print(f"{'='*80}")

    for cond in CONDITIONS:
        sub = df[df["condition"] == cond]
        print(f"\n  {cond}:")
        print(f"    recon: {sub['recon_per_dim'].mean():.6f} ± {sub['recon_per_dim'].std():.6f}")
        print(f"    tangent: {sub['tangent_err'].mean():.6f} ± {sub['tangent_err'].std():.6f}")
        print(f"    E_mu: {sub['E_mu'].mean():.4f} ± {sub['E_mu'].std():.4f}")
        print(f"    E_Sigma: {sub['E_Sigma'].mean():.4f} ± {sub['E_Sigma'].std():.4f}")
        print(f"    σ_min (in-dist min): {sub['sigma_min_in_min'].mean():.4f} ± {sub['sigma_min_in_min'].std():.4f}")
        for r in RADII:
            err = sub[f"MFPT_err_r{r}"].values
            ef = sub[f"exit_frac_r{r}"].values
            print(f"    r={r}: MFPT_err={np.nanmean(err):.1%}±{np.nanstd(err):.1%} "
                  f"exit_frac={np.nanmean(ef):.0%}")


if __name__ == "__main__":
    main()
