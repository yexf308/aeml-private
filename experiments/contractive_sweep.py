"""
Contractive penalty weight sweep: λ_C ∈ {0.1, 0.5, 1.0} on paraboloid D=3, 3 seeds.

Pick the weight with lowest mean reconstruction + tangent error on a 500-point test set.

Usage:
    python -m experiments.contractive_sweep
"""

import numpy as np
import pandas as pd
import torch

from src.numeric.datagen import sample_from_manifold
from src.numeric.losses import LossWeights
from src.numeric.performance_stats import compute_losses_per_sample

from experiments.paper_experiments import (
    create_manifold_sde, train_ae_with_twophase,
    N_TRAIN, TRAIN_BOUND, EPOCHS_AE, DEVICE,
)

LAMBDA_VALUES = [0.1, 0.5, 1.0]
N_SEEDS = 3
BASE_SEED = 42


def main():
    manifold_sde = create_manifold_sde("paraboloid", with_dynamics=True)
    seeds = [BASE_SEED + i * 1000 for i in range(N_SEEDS)]

    rows = []
    for lam_c in LAMBDA_VALUES:
        lw = LossWeights(contractive=lam_c)
        for seed in seeds:
            torch.manual_seed(seed)
            np.random.seed(seed)

            train_data = sample_from_manifold(
                manifold_sde,
                [(-TRAIN_BOUND, TRAIN_BOUND), (-TRAIN_BOUND, TRAIN_BOUND)],
                n_samples=N_TRAIN, seed=seed, device=DEVICE,
            )

            model = train_ae_with_twophase(train_data, f"C_{lam_c}", lw, EPOCHS_AE)

            test_data = sample_from_manifold(
                manifold_sde,
                [(-TRAIN_BOUND, TRAIN_BOUND), (-TRAIN_BOUND, TRAIN_BOUND)],
                n_samples=500, seed=seed + 10000, device=DEVICE,
            )
            losses_df = compute_losses_per_sample(model, test_data)

            row = {
                "lambda_c": lam_c,
                "seed": seed,
                "reconstruction": losses_df["reconstruction"].mean(),
                "tangent": losses_df["tangent"].mean(),
            }
            rows.append(row)
            print(f"  λ_C={lam_c}  seed={seed}  "
                  f"recon={row['reconstruction']:.6f}  tangent={row['tangent']:.6f}")

    df = pd.DataFrame(rows)
    df.to_csv("contractive_sweep.csv", index=False)

    # Summary
    print(f"\n{'='*60}")
    print("CONTRACTIVE SWEEP SUMMARY")
    print(f"{'='*60}")
    for lam_c in LAMBDA_VALUES:
        sub = df[df["lambda_c"] == lam_c]
        recon = sub["reconstruction"].values
        tangent = sub["tangent"].values
        combined = recon + tangent
        print(f"  λ_C={lam_c:.1f}:  recon={recon.mean():.6f}±{recon.std():.6f}  "
              f"tangent={tangent.mean():.6f}±{tangent.std():.6f}  "
              f"combined={combined.mean():.6f}")

    # Pick best
    best = df.groupby("lambda_c").apply(
        lambda g: (g["reconstruction"] + g["tangent"]).mean()
    ).idxmin()
    print(f"\n  Best λ_C = {best}")
    print(f"  Update LAMBDA_C in paper_experiments.py and mfpt_full_ablation.py")


if __name__ == "__main__":
    main()
