"""
Diagnostic experiment: ODE vs SDE trajectory comparison.

Tests the hypothesis that curvature regularization (K) improves drift
coefficients but the improvement is masked by diffusion error in full SDE.

Protocol:
  1. Train AE with T+F and T+F+K (same setup as trajectory study)
  2. Run end_to_end simulation in two modes:
     - SDE: with shared Brownian motion (dW)
     - ODE: with dW=0 (deterministic, drift-only)
  3. Compare MTE between T+F and T+F+K in both modes

If diffusion error is the bottleneck:
  - SDE mode: K helps on ~2/4 surfaces (current result)
  - ODE mode: K helps on ALL surfaces (drift improvement is visible)

Usage:
    python -m experiments.ode_vs_sde_diagnostic --n-seeds 10
    python -m experiments.ode_vs_sde_diagnostic --n-seeds 5 --surfaces paraboloid sinusoidal
"""

import argparse
import time

import numpy as np
import pandas as pd
import torch
from scipy import stats

from src.numeric.datagen import sample_from_manifold
from src.numeric.training import MultiModelTrainer, TrainingConfig, TrainingPhase

from experiments.common import SURFACE_MAP, PENALTY_CONFIGS, make_model_config
from experiments.paper_experiments import (
    create_manifold_sde, train_ae_with_twophase,
    DEVICE, N_TRAIN, TRAIN_BOUND, EPOCHS_AE, BATCH_SIZE, LR, BOUNDARY, DT,
)
from experiments.trajectory_fidelity_study import (
    lambdify_sde, simulate_ground_truth,
    simulate_end_to_end_with_coefficients,
    compute_mte_at_step,
    PATH_TIMES,
)

N_TRAJ = 200
T_MAX = 1.0  # Only need pathwise times for this diagnostic


def run_diagnostic(surfaces, seeds, epochs, output):
    """Run ODE vs SDE diagnostic for T+F and T+F+K."""
    configs = {
        "T+F": PENALTY_CONFIGS["T+F"],
        "T+F+K": PENALTY_CONFIGS["T+F+K"],
    }
    n_steps = int(T_MAX / DT)
    snapshot_steps = [int(round(t / DT)) for t in PATH_TIMES]

    print(f"\n{'='*70}")
    print("ODE vs SDE DIAGNOSTIC")
    print(f"Surfaces: {surfaces}")
    print(f"Seeds: {seeds}")
    print(f"Configs: {list(configs.keys())}")
    print(f"N_TRAIN={N_TRAIN}, N_TRAJ={N_TRAJ}, T_MAX={T_MAX}, DT={DT}")
    print(f"{'='*70}")

    all_rows = []
    for surf in surfaces:
        print(f"\n--- Surface: {surf} ---")
        manifold_sde = create_manifold_sde(surf, with_dynamics=True)
        sde = lambdify_sde(manifold_sde)

        for seed in seeds:
            print(f"\n  Seed: {seed}")

            # Shared initial conditions
            torch.manual_seed(seed + 999)
            initial_local = (torch.rand(N_TRAJ, 2, device=DEVICE) * 2 - 1) * TRAIN_BOUND
            initial_ambient = sde.chart(initial_local).to(DEVICE)

            # Shared Brownian motion for SDE mode
            torch.manual_seed(seed + 1234)
            dW = torch.randn(N_TRAJ, n_steps, 2, device=DEVICE)

            # Zero noise for ODE mode
            dW_zero = torch.zeros_like(dW)

            # Ground truth trajectories (SDE and ODE)
            gt_sde_traj, gt_sde_alive, _ = simulate_ground_truth(
                initial_local, sde, n_steps, DT, dW, BOUNDARY,
            )
            gt_ode_traj, gt_ode_alive, _ = simulate_ground_truth(
                initial_local, sde, n_steps, DT, dW_zero, BOUNDARY,
            )

            gt_sde_surv = gt_sde_alive[:, -1].float().mean().item()
            gt_ode_surv = gt_ode_alive[:, -1].float().mean().item()
            print(f"    GT survival: SDE={gt_sde_surv:.0%}, ODE={gt_ode_surv:.0%}")

            for cfg_name, lw in configs.items():
                print(f"    Config: {cfg_name}")

                # Train AE
                torch.manual_seed(seed)
                np.random.seed(seed)
                train_data = sample_from_manifold(
                    manifold_sde,
                    [(-TRAIN_BOUND, TRAIN_BOUND), (-TRAIN_BOUND, TRAIN_BOUND)],
                    n_samples=N_TRAIN, seed=seed, device=DEVICE,
                )
                model = train_ae_with_twophase(train_data, cfg_name, lw, epochs)

                # SDE simulation
                sde_traj, sde_alive, sde_coeff = simulate_end_to_end_with_coefficients(
                    model, initial_ambient, sde, n_steps, DT, dW, BOUNDARY,
                    snapshot_steps=snapshot_steps,
                )

                # ODE simulation (same model, dW=0)
                ode_traj, ode_alive, ode_coeff = simulate_end_to_end_with_coefficients(
                    model, initial_ambient, sde, n_steps, DT, dW_zero, BOUNDARY,
                    snapshot_steps=snapshot_steps,
                )

                # Compute MTE at each path time
                for t in PATH_TIMES:
                    step = int(round(t / DT))

                    # SDE MTE
                    both_sde = sde_alive[:, step] & gt_sde_alive[:, step]
                    mte_sde = compute_mte_at_step(sde_traj, gt_sde_traj, both_sde, step)

                    # ODE MTE
                    both_ode = ode_alive[:, step] & gt_ode_alive[:, step]
                    mte_ode = compute_mte_at_step(ode_traj, gt_ode_traj, both_ode, step)

                    # Coefficient errors (from SDE run, at snapshot)
                    drift_err = float('nan')
                    cov_err = float('nan')
                    if step in sde_coeff:
                        ce = sde_coeff[step]
                        alive_ce = ce["alive"] & gt_sde_alive[:, step]
                        if alive_ce.any():
                            drift_err = ce["drift_error"][alive_ce].median().item()
                            cov_err = ce["cov_error"][alive_ce].median().item()

                    all_rows.append({
                        "surface": surf,
                        "config": cfg_name,
                        "seed": seed,
                        "time": t,
                        "MTE_sde": mte_sde,
                        "MTE_ode": mte_ode,
                        "E_mu": drift_err,
                        "E_Sigma": cov_err,
                    })

                # Print summary at t=1.0
                step1 = int(round(1.0 / DT))
                both_sde1 = sde_alive[:, step1] & gt_sde_alive[:, step1]
                both_ode1 = ode_alive[:, step1] & gt_ode_alive[:, step1]
                mte_sde1 = compute_mte_at_step(sde_traj, gt_sde_traj, both_sde1, step1)
                mte_ode1 = compute_mte_at_step(ode_traj, gt_ode_traj, both_ode1, step1)
                print(f"      MTE@1.0: SDE={mte_sde1:.4f}, ODE={mte_ode1:.4f}")

    df = pd.DataFrame(all_rows)
    if output:
        df.to_csv(output, index=False)
        print(f"\nSaved to {output}")

    # Analysis: paired t-test for K effect in SDE vs ODE
    print(f"\n{'='*70}")
    print("ANALYSIS: Does K help drift (ODE) even when it doesn't help SDE?")
    print(f"{'='*70}")

    for t in [0.5, 1.0]:
        print(f"\n  Time = {t}")
        sub = df[df["time"] == t]

        for surf in surfaces:
            s = sub[sub["surface"] == surf]
            tf = s[s["config"] == "T+F"].sort_values("seed")
            tfk = s[s["config"] == "T+F+K"].sort_values("seed")

            if len(tf) == 0 or len(tfk) == 0:
                continue

            # SDE: paired t-test
            sde_tf = tf["MTE_sde"].values
            sde_tfk = tfk["MTE_sde"].values
            sde_diff = sde_tf - sde_tfk  # positive = K helps
            sde_t, sde_p = stats.ttest_rel(sde_tf, sde_tfk)

            # ODE: paired t-test
            ode_tf = tf["MTE_ode"].values
            ode_tfk = tfk["MTE_ode"].values
            ode_diff = ode_tf - ode_tfk  # positive = K helps
            ode_t, ode_p = stats.ttest_rel(ode_tf, ode_tfk)

            k_helps_sde = sde_diff.mean() > 0
            k_helps_ode = ode_diff.mean() > 0

            sig = lambda p: "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "ns"

            print(f"\n    {surf}:")
            print(f"      SDE: T+F={sde_tf.mean():.4f}±{sde_tf.std():.4f}  "
                  f"T+F+K={sde_tfk.mean():.4f}±{sde_tfk.std():.4f}  "
                  f"Δ={sde_diff.mean():+.4f} ({sde_diff.mean()/sde_tf.mean()*100:+.1f}%)  "
                  f"p={sde_p:.3f} {sig(sde_p)}  {'K helps' if k_helps_sde else 'K hurts'}")
            print(f"      ODE: T+F={ode_tf.mean():.4f}±{ode_tf.std():.4f}  "
                  f"T+F+K={ode_tfk.mean():.4f}±{ode_tfk.std():.4f}  "
                  f"Δ={ode_diff.mean():+.4f} ({ode_diff.mean()/ode_tf.mean()*100:+.1f}%)  "
                  f"p={ode_p:.3f} {sig(ode_p)}  {'K helps' if k_helps_ode else 'K hurts'}")

            # E_mu and E_Sigma comparison
            tf_emu = tf["E_mu"].dropna().values
            tfk_emu = tfk["E_mu"].dropna().values
            tf_esig = tf["E_Sigma"].dropna().values
            tfk_esig = tfk["E_Sigma"].dropna().values
            if len(tf_emu) > 1 and len(tfk_emu) > 1:
                print(f"      E_μ:  T+F={np.mean(tf_emu):.4f}  T+F+K={np.mean(tfk_emu):.4f}  "
                      f"Δ={np.mean(tf_emu)-np.mean(tfk_emu):+.4f} ({(np.mean(tf_emu)-np.mean(tfk_emu))/np.mean(tf_emu)*100:+.1f}%)")
            if len(tf_esig) > 1 and len(tfk_esig) > 1:
                print(f"      E_Σ:  T+F={np.mean(tf_esig):.4f}  T+F+K={np.mean(tfk_esig):.4f}  "
                      f"Δ={np.mean(tf_esig)-np.mean(tfk_esig):+.4f} ({(np.mean(tf_esig)-np.mean(tfk_esig))/np.mean(tf_esig)*100:+.1f}%)")

    # Summary verdict
    print(f"\n{'='*70}")
    print("VERDICT")
    print(f"{'='*70}")
    sub1 = df[df["time"] == 1.0]
    sde_helps = 0
    ode_helps = 0
    for surf in surfaces:
        s = sub1[sub1["surface"] == surf]
        tf = s[s["config"] == "T+F"]["MTE_sde"].values
        tfk = s[s["config"] == "T+F+K"]["MTE_sde"].values
        if len(tf) > 0 and len(tfk) > 0 and np.mean(tf) > np.mean(tfk):
            sde_helps += 1
        tf_o = s[s["config"] == "T+F"]["MTE_ode"].values
        tfk_o = s[s["config"] == "T+F+K"]["MTE_ode"].values
        if len(tf_o) > 0 and len(tfk_o) > 0 and np.mean(tf_o) > np.mean(tfk_o):
            ode_helps += 1

    print(f"  K improves MTE in SDE mode: {sde_helps}/{len(surfaces)} surfaces")
    print(f"  K improves MTE in ODE mode: {ode_helps}/{len(surfaces)} surfaces")
    if ode_helps > sde_helps:
        print("  → CONFIRMED: K's drift improvement is masked by diffusion error in SDE")
    elif ode_helps == sde_helps:
        print("  → INCONCLUSIVE: K helps same surfaces in both modes")
    else:
        print("  → REJECTED: K doesn't improve ODE either")

    return df


def main():
    parser = argparse.ArgumentParser(description="ODE vs SDE diagnostic for K effect")
    parser.add_argument("--n-seeds", type=int, default=10)
    parser.add_argument("--base-seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=EPOCHS_AE)
    parser.add_argument("--surfaces", type=str, nargs="+",
                        default=["paraboloid", "hyperbolic_paraboloid", "quartic_dome", "sinusoidal"],
                        choices=list(SURFACE_MAP.keys()))
    parser.add_argument("--output", type=str, default="diagnostic_ode_vs_sde.csv")
    args = parser.parse_args()

    seeds = [args.base_seed + i * 1000 for i in range(args.n_seeds)]

    print(f"Device: {DEVICE}")
    print(f"Seeds ({len(seeds)}): {seeds}")

    t0 = time.time()
    run_diagnostic(args.surfaces, seeds, args.epochs, args.output)
    print(f"\nTotal time: {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
