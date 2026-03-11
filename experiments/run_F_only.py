"""Run F-only condition for extrapolation and dynamics studies.

Appends results to paper_extrapolation.csv and paper_dynamics.csv.

Usage:
    python -m experiments.run_F_only --n-seeds 10
"""

import argparse
import torch
import numpy as np
import pandas as pd
from pathlib import Path

from src.numeric.losses import LossWeights
from src.numeric.datagen import sample_from_manifold

from experiments.paper_experiments import (
    create_manifold_sde, train_ae_with_twophase,
    run_extrapolation_seed, run_dynamics_extrap_seed,
    DEVICE,
)

SURFACES = ["paraboloid", "hyperbolic_paraboloid", "quartic_dome", "sinusoidal"]
DYNAMICS_SURFACES = ["paraboloid"]  # Table 3 is paraboloid only

F_LW = LossWeights(diffeo=1.0)
DISTANCES = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]
DIST_STEP = 0.1


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-seeds", type=int, default=10)
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--base-seed", type=int, default=42)
    args = parser.parse_args()

    seeds = [args.base_seed + i * 1000 for i in range(args.n_seeds)]
    print(f"Device: {DEVICE}")
    print(f"Seeds: {seeds}")

    # ── Extrapolation study (F only) ──
    print(f"\n{'='*60}")
    print("EXTRAPOLATION: F only")
    print(f"{'='*60}")

    extrap_rows = []
    for surf in SURFACES:
        msde = create_manifold_sde(surf, with_dynamics=False)
        for seed in seeds:
            print(f"\n  {surf} | F | seed={seed}")
            rows = run_extrapolation_seed(
                msde, surf, "F", F_LW, seed, args.epochs,
                DISTANCES, DIST_STEP,
            )
            extrap_rows.extend(rows)
            r0 = [r for r in rows if r["distance"] == 0.0][0]
            print(f"    dist=0.0: recon={r0['reconstruction']:.6f}")

    # Append to existing CSV
    extrap_path = Path("paper_extrapolation.csv")
    df_new = pd.DataFrame(extrap_rows)
    if extrap_path.exists():
        df_old = pd.read_csv(extrap_path)
        # Remove any existing F rows to avoid duplicates
        df_old = df_old[df_old["config"] != "F"]
        df_all = pd.concat([df_old, df_new], ignore_index=True)
    else:
        df_all = df_new
    df_all.to_csv(extrap_path, index=False)
    print(f"\nAppended {len(df_new)} rows to {extrap_path}")

    # Summary
    for dist in [0.0, 0.3]:
        print(f"\n  F at delta={dist}:")
        sub = df_new[df_new["distance"] == dist]
        for surf in SURFACES:
            vals = sub[sub["surface"] == surf]["reconstruction"]
            if len(vals) > 0:
                print(f"    {surf:>25s}: {vals.mean():.3f} ± {vals.std():.3f}")

    # ── Dynamics study (F only, paraboloid) ──
    print(f"\n{'='*60}")
    print("DYNAMICS: F only (paraboloid)")
    print(f"{'='*60}")

    dyn_rows = []
    for surf in DYNAMICS_SURFACES:
        msde = create_manifold_sde(surf, with_dynamics=True)
        for seed in seeds:
            print(f"\n  {surf} | F | seed={seed}")
            rows = run_dynamics_extrap_seed(
                msde, surf, "F", F_LW, seed, args.epochs,
                DISTANCES, DIST_STEP,
            )
            dyn_rows.extend(rows)

    # Append to existing CSV
    dyn_path = Path("paper_dynamics.csv")
    df_new_dyn = pd.DataFrame(dyn_rows)
    if dyn_path.exists():
        df_old_dyn = pd.read_csv(dyn_path)
        df_old_dyn = df_old_dyn[df_old_dyn["config"] != "F"]
        df_all_dyn = pd.concat([df_old_dyn, df_new_dyn], ignore_index=True)
    else:
        df_all_dyn = df_new_dyn
    df_all_dyn.to_csv(dyn_path, index=False)
    print(f"\nAppended {len(df_new_dyn)} rows to {dyn_path}")

    # Summary
    for dist in [0.0, 0.3]:
        print(f"\n  F dynamics at delta={dist}:")
        sub = df_new_dyn[df_new_dyn["distance"] == dist]
        for surf in DYNAMICS_SURFACES:
            s = sub[sub["surface"] == surf]
            if len(s) > 0:
                print(f"    cov_tangent: {s['cov_tangent'].mean():.1f} ± {s['cov_tangent'].std():.1f}")
                print(f"    drift_tangent: {s['drift_tangent'].mean():.2f} ± {s['drift_tangent'].std():.2f}")


if __name__ == "__main__":
    main()
