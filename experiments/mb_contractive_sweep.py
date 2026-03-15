"""
Contractive weight sweep for MB dynamics.

Sweep λ_C ∈ {0.01, 0.05, 0.1, 0.5} on paraboloid D=11, 3 seeds.
Pick λ_C with lowest combined (recon + tangent) error.

Output: mb_contractive_sweep.csv
"""

import argparse
import time
import torch
import numpy as np
import pandas as pd

from src.numeric.losses import LossWeights
from src.numeric.performance_stats import compute_losses_per_sample
from src.numeric.highd_manifolds import (
    FourierAugmentedSurface,
    sample_from_highd_manifold,
)

from experiments.mb_dynamics import (
    mb_local_drift_fn, mb_local_diffusion_fn,
    train_ae_highd, TRAIN_BOUND, DEVICE,
)


LAMBDA_C_VALUES = [0.01, 0.05, 0.1, 0.5]
N_TRAIN = 50


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-seeds", type=int, default=3)
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--base-seed", type=int, default=42)
    parser.add_argument("--D", type=int, default=11)
    parser.add_argument("--output", type=str, default="mb_contractive_sweep.csv")
    args = parser.parse_args()

    seeds = [args.base_seed + i * 1000 for i in range(args.n_seeds)]
    surface = FourierAugmentedSurface("paraboloid", args.D)

    print(f"Device: {DEVICE}")
    print(f"D={args.D}, N={N_TRAIN}")
    print(f"λ_C values: {LAMBDA_C_VALUES}")
    print(f"Seeds: {seeds}\n")

    t0 = time.time()
    rows = []

    for seed in seeds:
        torch.manual_seed(seed)
        np.random.seed(seed)

        train_data = sample_from_highd_manifold(
            surface, mb_local_drift_fn, mb_local_diffusion_fn,
            [(-TRAIN_BOUND, TRAIN_BOUND), (-TRAIN_BOUND, TRAIN_BOUND)],
            n_samples=N_TRAIN, seed=seed, device=DEVICE,
        )

        for lam_c in LAMBDA_C_VALUES:
            print(f"\n  seed={seed}, λ_C={lam_c}")
            torch.manual_seed(seed)
            np.random.seed(seed)

            lw = LossWeights(contractive=lam_c)
            ae, recon = train_ae_highd(
                surface, args.D, seed, N_TRAIN, args.epochs, lw, train_data,
            )

            test_data = sample_from_highd_manifold(
                surface, mb_local_drift_fn, mb_local_diffusion_fn,
                [(-TRAIN_BOUND, TRAIN_BOUND), (-TRAIN_BOUND, TRAIN_BOUND)],
                n_samples=300, seed=seed + 9000, device=DEVICE,
            )
            losses_df = compute_losses_per_sample(ae, test_data)
            tangent = losses_df["tangent"].mean()
            combined = recon + tangent

            row = {
                "seed": seed,
                "lambda_c": lam_c,
                "recon_per_dim": recon,
                "tangent": tangent,
                "combined": combined,
            }
            rows.append(row)
            print(f"    recon={recon:.6f}  tangent={tangent:.6f}  combined={combined:.6f}")

    df = pd.DataFrame(rows)
    df.to_csv(args.output, index=False)
    print(f"\nSaved {len(df)} rows to {args.output}")

    # Summary
    print(f"\n{'='*60}")
    print("CONTRACTIVE SWEEP SUMMARY")
    print(f"{'='*60}")
    for lam_c in LAMBDA_C_VALUES:
        sub = df[df["lambda_c"] == lam_c]
        print(f"  λ_C={lam_c}: combined={sub['combined'].mean():.4f}±{sub['combined'].std():.4f}"
              f"  recon={sub['recon_per_dim'].mean():.4f}  tangent={sub['tangent'].mean():.4f}")

    best = df.groupby("lambda_c")["combined"].mean().idxmin()
    print(f"\n  Best: λ_C={best}")
    print(f"\nTotal time: {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
