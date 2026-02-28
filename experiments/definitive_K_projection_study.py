"""
Definitive K projection ablation: shared-checkpoint fork + N sensitivity.

Design:
  For each (surface, seed, N), train Phase 1 (T+F warmup) ONCE, then fork
  the checkpoint into 3 conditions:
    1. baseline  — no Phase 2, use Phase 1 AE as-is
    2. K(Phat)   — Phase 2 finetune with learned projection (use_true_projection=False)
    3. K(Pstar)  — Phase 2 finetune with true projection (use_true_projection=True)

  This isolates the K computation as the ONLY variable between conditions.

  N sensitivity: N=20, 50, 100 to see where curvature regularization helps.

  Parameters: 20 seeds x 3 surfaces x 3 conditions x 3 N values = 540 rows
  (but only 180 Phase 1 trainings since Phase 1 is shared across conditions).

Usage:
    # Smoke test
    python -m experiments.definitive_K_projection_study --n-seeds 1 --epochs 50 --sde-epochs 50

    # Full run on GPU
    srun --partition=dgx --gres=gpu:1 --time=04:00:00 \
      python3 -u -m experiments.definitive_K_projection_study --n-seeds 20
"""

import argparse
import copy
import time
import torch
import numpy as np
import pandas as pd
from scipy import stats

from src.numeric.datagen import sample_from_manifold
from src.numeric.losses import LossWeights
from src.numeric.sde_nets import DriftNet, DiffusionNet
from src.numeric.sde_training import SDEPipelineTrainer
from src.numeric.training import MultiModelTrainer, TrainingConfig

from experiments.common import make_model_config
from experiments.data_driven_sde import (
    create_manifold_sde, evaluate_pipeline, lambdify_sde,
    TRAIN_BOUND, BATCH_SIZE, LR_AE, LR_SDE, DEVICE,
)

SURFACES = ["sinusoidal", "paraboloid", "hyperbolic_paraboloid"]
N_VALUES = [20, 50, 100]

# Phase 1: T+F warmup (no curvature)
WARMUP_LW = LossWeights(tangent_bundle=1.0, diffeo=1.0)
# Phase 2: T+F+K finetune
FULL_LW = LossWeights(tangent_bundle=1.0, diffeo=1.0, curvature=0.1)

# (label, phase2_loss_weights_or_None, use_true_projection)
CONDITIONS = [
    ("baseline", None,    False),
    ("K(Phat)",  FULL_LW, False),
    ("K(Pstar)", FULL_LW, True),
]

METRICS = ["MTE@0.1", "MTE@0.5", "MTE@1.0", "W2@1.0", "MMD@1.0"]


def run_single_fork(surface_name, seed, n_train, epochs_ae=500, epochs_sde=300):
    """Run shared-checkpoint fork for one (surface, seed, N).

    Returns dict mapping condition_label -> metrics dict.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)

    # --- Sample data ---
    manifold_sde = create_manifold_sde(surface_name)
    train_data = sample_from_manifold(
        manifold_sde,
        [(-TRAIN_BOUND, TRAIN_BOUND), (-TRAIN_BOUND, TRAIN_BOUND)],
        n_samples=n_train, seed=seed, device=DEVICE,
    )
    x = train_data.samples.to(DEVICE)
    v = train_data.mu.to(DEVICE)
    Lambda = train_data.cov.to(DEVICE)

    # --- Phase 1: T+F warmup (ONCE) ---
    phase1_epochs = epochs_ae // 2

    trainer = MultiModelTrainer(TrainingConfig(
        epochs=epochs_ae, n_samples=n_train, input_dim=3, hidden_dim=64,
        latent_dim=2, learning_rate=LR_AE, batch_size=BATCH_SIZE,
        test_size=0.03, print_interval=max(1, phase1_epochs // 5), device=DEVICE,
    ))
    mc = make_model_config("ae", WARMUP_LW, hidden_dims=[64, 64])
    trainer.add_model(mc)
    loader = trainer.create_data_loader(train_data)

    for epoch in range(phase1_epochs):
        losses = trainer.train_epoch(loader)
        if (epoch + 1) % max(1, phase1_epochs // 5) == 0:
            print(f"    Phase1 Epoch {epoch+1}/{phase1_epochs}: loss={losses['ae']:.6f}")

    # Snapshot Phase 1 state
    phase1_state = copy.deepcopy(trainer.models["ae"].state_dict())
    phase1_optim_state = copy.deepcopy(trainer.optimizers["ae"].state_dict())
    phase1_sched_state = copy.deepcopy(trainer.schedulers["ae"].state_dict())

    # Lambdify SDE once for evaluation
    sde = lambdify_sde(create_manifold_sde(surface_name))

    # --- Fork into 3 conditions ---
    phase2_epochs = epochs_ae - phase1_epochs
    results = {}

    for cond_label, phase2_lw, use_true_proj in CONDITIONS:
        print(f"      Condition: {cond_label}")

        # Create fresh trainer, load Phase 1 checkpoint
        t2 = MultiModelTrainer(TrainingConfig(
            epochs=epochs_ae, n_samples=n_train, input_dim=3, hidden_dim=64,
            latent_dim=2, learning_rate=LR_AE, batch_size=BATCH_SIZE,
            test_size=0.03, print_interval=max(1, phase2_epochs // 5), device=DEVICE,
        ))
        mc2 = make_model_config("ae", WARMUP_LW, hidden_dims=[64, 64])
        mc2.use_true_projection = use_true_proj
        t2.add_model(mc2)
        t2.models["ae"].load_state_dict(phase1_state)
        t2.optimizers["ae"].load_state_dict(phase1_optim_state)
        t2.schedulers["ae"].load_state_dict(phase1_sched_state)

        if phase2_lw is not None:
            # Phase 2 finetune with K
            for epoch in range(phase2_epochs):
                ep_losses = t2.train_epoch(loader, {mc2.name: phase2_lw})
                if (epoch + 1) % max(1, phase2_epochs // 5) == 0:
                    print(f"        Phase2 Epoch {epoch+1}/{phase2_epochs}: "
                          f"loss={ep_losses['ae']:.6f}")

        ae = t2.models["ae"]
        ae.eval()

        # Precompute decoder derivatives for SDE stages
        d = 2
        dummy = SDEPipelineTrainer(
            ae, DriftNet(d).to(DEVICE), DiffusionNet(d).to(DEVICE), device=DEVICE,
        )
        z_pre, dphi_pre, d2phi_pre = dummy.precompute_decoder_derivatives(x)

        # Stage 2: Drift (deterministic init per seed)
        torch.manual_seed(seed + 100)
        drift_net = DriftNet(d).to(DEVICE)
        dp = SDEPipelineTrainer(
            ae, drift_net, DiffusionNet(d).to(DEVICE), device=DEVICE,
        )
        drift_losses = dp.train_stage2_precomputed(
            z_pre, dphi_pre, d2phi_pre, v, Lambda,
            epochs=epochs_sde, lr=LR_SDE, batch_size=BATCH_SIZE,
            print_interval=max(1, epochs_sde // 5),
        )
        drift_net.eval()

        # Stage 3: Diffusion (deterministic init per seed)
        torch.manual_seed(seed + 200)
        diff_net = DiffusionNet(d).to(DEVICE)
        pipeline = SDEPipelineTrainer(ae, drift_net, diff_net, device=DEVICE)
        diff_losses = pipeline.train_stage3_precomputed(
            z_pre, dphi_pre, Lambda,
            epochs=epochs_sde, lr=LR_SDE, batch_size=BATCH_SIZE,
            print_interval=max(1, epochs_sde // 5),
        )

        # Evaluate
        eval_results = evaluate_pipeline(pipeline, ae, sde, seed)
        results[cond_label] = {
            "ae_loss": ep_losses["ae"] if phase2_lw is not None else losses["ae"],
            "drift_loss": drift_losses[-1],
            "diff_loss": diff_losses[-1],
            **eval_results,
        }

    return results


def paired_ttest(vals_a, vals_b):
    """Paired t-test: is mean(a - b) significantly different from 0?"""
    diffs = np.array(vals_a) - np.array(vals_b)
    n = len(diffs)
    if n < 2:
        return float("nan"), float("nan")
    mean_d = diffs.mean()
    std_d = diffs.std(ddof=1)
    if std_d < 1e-15:
        return mean_d, 0.0
    t_stat = mean_d / (std_d / np.sqrt(n))
    p_val = stats.t.sf(abs(t_stat), df=n - 1) * 2
    return mean_d, p_val


def print_summary(df):
    """Print grouped summary tables with paired t-tests."""
    cond_labels = [c[0] for c in CONDITIONS]

    print(f"\n\n{'='*120}")
    print("DEFINITIVE K PROJECTION ABLATION RESULTS")
    print(f"{'='*120}")

    for n_train in sorted(df["N"].unique()):
        df_n = df[df["N"] == n_train]
        print(f"\n{'─'*120}")
        print(f"  N = {n_train}")
        print(f"{'─'*120}")

        # Mean ± std table
        print(f"\n  {'surface':>25s}  {'condition':>10s}  {'n':>3s}  ", end="")
        for m in METRICS:
            print(f"{'mean':>8s} {'+-std':>8s}  ", end="")
        print()
        print("  " + "-" * 110)

        for surface_name in SURFACES:
            for cond_label in cond_labels:
                subset = df_n[
                    (df_n["surface"] == surface_name)
                    & (df_n["condition"] == cond_label)
                ]
                n = len(subset)
                print(f"  {surface_name:>25s}  {cond_label:>10s}  {n:>3d}  ", end="")
                for m in METRICS:
                    vals = subset[m].values
                    print(f"{vals.mean():>8.4f} {vals.std():>8.4f}  ", end="")
                print()

        # Paired comparisons
        comparisons = [
            ("K(Phat)",  "baseline", "K(Phat) vs baseline"),
            ("K(Pstar)", "baseline", "K(Pstar) vs baseline"),
            ("K(Pstar)", "K(Phat)",  "K(Pstar) vs K(Phat)"),
        ]

        print(f"\n  Paired comparisons (negative delta = first condition is better):")
        for cond_a, cond_b, desc in comparisons:
            print(f"\n    {desc}:")
            print(f"    {'surface':>25s}  ", end="")
            for m in METRICS:
                print(f"{'delta':>8s} {'p-val':>8s}  ", end="")
            print("  win-rate  verdict")
            print("    " + "-" * 108)

            for surface_name in SURFACES:
                a = df_n[
                    (df_n["surface"] == surface_name)
                    & (df_n["condition"] == cond_a)
                ].sort_values("seed")
                b = df_n[
                    (df_n["surface"] == surface_name)
                    & (df_n["condition"] == cond_b)
                ].sort_values("seed")

                print(f"    {surface_name:>25s}  ", end="")
                verdicts = []
                for m in METRICS:
                    mean_d, p_val = paired_ttest(a[m].values, b[m].values)
                    sig = "**" if p_val < 0.01 else "*" if p_val < 0.05 else "+" if p_val < 0.1 else ""
                    print(f"{mean_d:>+8.4f} {p_val:>7.4f}{sig:<1s} ", end="")
                    if p_val < 0.05:
                        verdicts.append(f"{m}: {'worse' if mean_d > 0 else 'better'}")

                # Win rate on MTE@1.0 (lower is better)
                a_mte = a["MTE@1.0"].values
                b_mte = b["MTE@1.0"].values
                wins = np.sum(a_mte < b_mte)
                total = len(a_mte)
                win_str = f"{wins}/{total}"

                verdict_str = "; ".join(verdicts) if verdicts else "n.s."
                print(f"  {win_str:>8s}  {verdict_str}")


def main():
    parser = argparse.ArgumentParser(
        description="Definitive K projection ablation (shared-checkpoint fork + N sensitivity)",
    )
    parser.add_argument("--n-seeds", type=int, default=20)
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--sde-epochs", type=int, default=300)
    parser.add_argument("--base-seed", type=int, default=42)
    parser.add_argument("--n-values", type=int, nargs="+", default=N_VALUES,
                        help="Training set sizes to sweep (default: 20 50 100)")
    args = parser.parse_args()

    seeds = [args.base_seed + i * 1000 for i in range(args.n_seeds)]
    cond_labels = [c[0] for c in CONDITIONS]

    print(f"Device: {DEVICE}")
    print(f"Seeds ({len(seeds)}): {seeds}")
    print(f"Surfaces: {SURFACES}")
    print(f"Conditions: {cond_labels}")
    print(f"N values: {args.n_values}")
    print(f"AE epochs: {args.epochs} (Phase 1: {args.epochs // 2}, Phase 2: {args.epochs - args.epochs // 2})")
    print(f"SDE epochs: {args.sde_epochs}")
    expected_rows = len(args.n_values) * len(SURFACES) * len(cond_labels) * len(seeds)
    print(f"Expected rows: {expected_rows}\n")

    t0 = time.time()
    all_rows = []

    for n_train in args.n_values:
        for surface_name in SURFACES:
            for seed in seeds:
                print(f"\n{'='*60}")
                print(f"  N={n_train} | {surface_name} | seed={seed}")
                print(f"{'='*60}")

                fork_results = run_single_fork(
                    surface_name, seed, n_train,
                    epochs_ae=args.epochs, epochs_sde=args.sde_epochs,
                )
                for cond_label, metrics in fork_results.items():
                    all_rows.append({
                        "N": n_train,
                        "surface": surface_name,
                        "condition": cond_label,
                        "seed": seed,
                        **metrics,
                    })

                    print(f"    {cond_label:>10s}: MTE@1.0={metrics['MTE@1.0']:.4f}  "
                          f"W2@1.0={metrics['W2@1.0']:.4f}  MMD@1.0={metrics['MMD@1.0']:.4f}")

    df = pd.DataFrame(all_rows)
    csv_path = "definitive_K_projection_study.csv"
    df.to_csv(csv_path, index=False)
    print(f"\nSaved {len(df)} rows to {csv_path}")

    print_summary(df)

    print(f"\nTotal time: {time.time() - t0:.1f}s")
    print("Done.")


if __name__ == "__main__":
    main()
