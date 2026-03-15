"""
Ablation study with MB dynamics on paraboloid D=11.

9 configs × 10 seeds. Reports reconstruction, tangent, curvature, σ_min.
K-containing configs use two-phase training via train_ae_highd.

Output: mb_ablation.csv
"""

import argparse
import time
import torch
import numpy as np
import pandas as pd

from src.numeric.losses import LossWeights
from src.numeric.geometry import compute_sigma_min
from src.numeric.performance_stats import compute_losses_per_sample
from src.numeric.highd_manifolds import (
    FourierAugmentedSurface,
    sample_from_highd_manifold,
)

from experiments.mb_dynamics import (
    mb_local_drift_fn, mb_local_diffusion_fn,
    train_ae_highd, TRAIN_BOUND, DEVICE,
)

# Best from MB contractive sweep (mb_contractive_sweep.csv)
LAMBDA_C = 0.01

ABLATION_CONFIGS = {
    "baseline": LossWeights(),
    "T": LossWeights(tangent_bundle=1.0),
    "K": LossWeights(curvature=0.1),
    "F": LossWeights(diffeo=1.0),
    "C": LossWeights(contractive=LAMBDA_C),
    "T+K": LossWeights(tangent_bundle=1.0, curvature=0.1),
    "T+F": LossWeights(tangent_bundle=1.0, diffeo=1.0),
    "F+K": LossWeights(diffeo=1.0, curvature=0.1),
    "T+F+K": LossWeights(tangent_bundle=1.0, diffeo=1.0, curvature=0.1),
}

N_TRAIN = 50


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-seeds", type=int, default=10)
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--base-seed", type=int, default=42)
    parser.add_argument("--D", type=int, default=11)
    parser.add_argument("--output", type=str, default="mb_ablation.csv")
    args = parser.parse_args()

    seeds = [args.base_seed + i * 1000 for i in range(args.n_seeds)]

    print(f"Device: {DEVICE}")
    print(f"D={args.D}, N={N_TRAIN}")
    print(f"Configs: {list(ABLATION_CONFIGS.keys())}")
    print(f"Seeds ({len(seeds)}): {seeds}\n")

    surface = FourierAugmentedSurface("paraboloid", args.D)
    t0 = time.time()
    all_rows = []

    for seed in seeds:
        torch.manual_seed(seed)
        np.random.seed(seed)

        train_data = sample_from_highd_manifold(
            surface, mb_local_drift_fn, mb_local_diffusion_fn,
            [(-TRAIN_BOUND, TRAIN_BOUND), (-TRAIN_BOUND, TRAIN_BOUND)],
            n_samples=N_TRAIN, seed=seed, device=DEVICE,
        )

        for cfg_name, lw in ABLATION_CONFIGS.items():
            print(f"\n  {cfg_name} | seed={seed}")
            torch.manual_seed(seed)
            np.random.seed(seed)

            ae, recon = train_ae_highd(
                surface, args.D, seed, N_TRAIN, args.epochs, lw, train_data,
            )

            # Evaluate on 500-point test set
            test_data = sample_from_highd_manifold(
                surface, mb_local_drift_fn, mb_local_diffusion_fn,
                [(-TRAIN_BOUND, TRAIN_BOUND), (-TRAIN_BOUND, TRAIN_BOUND)],
                n_samples=500, seed=seed + 10000, device=DEVICE,
            )
            losses_df = compute_losses_per_sample(ae, test_data)

            # σ_min diagnostic
            sigma_min_vals = compute_sigma_min(ae, test_data.samples.to(DEVICE))

            row = {
                "config": cfg_name,
                "seed": seed,
                "reconstruction": losses_df["reconstruction"].mean(),
                "tangent": losses_df["tangent"].mean(),
                "curvature": losses_df["curvature"].mean(),
                "sigma_min_min": sigma_min_vals.min().item(),
                "sigma_min_p5": sigma_min_vals.quantile(0.05).item(),
                "sigma_min_median": sigma_min_vals.median().item(),
            }
            all_rows.append(row)
            print(f"    recon={recon:.6f}  tangent={row['tangent']:.6f}  "
                  f"σ_min_p5={row['sigma_min_p5']:.3f}")

        # Save incrementally
        pd.DataFrame(all_rows).to_csv(args.output, index=False)

    df = pd.DataFrame(all_rows)
    df.to_csv(args.output, index=False)
    print(f"\nSaved {len(df)} rows to {args.output}")

    # Summary
    print(f"\n{'='*70}")
    print("MB ABLATION SUMMARY (mean ± std)")
    print(f"{'='*70}")
    metrics = ["reconstruction", "tangent", "curvature"]
    print(f"{'config':>10s}  {'n':>3s}  ", end="")
    for m in metrics:
        print(f"{m:>14s} {'':>14s}  ", end="")
    print()
    for cfg_name in ABLATION_CONFIGS:
        sub = df[df["config"] == cfg_name]
        print(f"{cfg_name:>10s}  {len(sub):>3d}  ", end="")
        for m in metrics:
            vals = sub[m].values
            print(f"{vals.mean():>14.6f} +/-{vals.std():>10.6f}  ", end="")
        print()

    print(f"\nTotal time: {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
