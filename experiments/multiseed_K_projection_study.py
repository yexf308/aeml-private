"""
Multi-seed ablation: (I - P̂) vs (I - P★) in the curvature K loss.

Four conditions per surface × seed:
  1. T+F           — baseline, no K
  2. T+F+K(P̂) 2ph — current: K with learned projection, two-phase
  3. T+F+K(P★) 2ph — K with true projection, two-phase
  4. T+F+K(P★) 1ph — K with true projection, single-phase

Reports mean ± std across seeds and paired t-tests.

Usage:
    PYTHONUNBUFFERED=1 python -m experiments.multiseed_K_projection_study
    PYTHONUNBUFFERED=1 python -m experiments.multiseed_K_projection_study --n-seeds 10
"""

import argparse
import time
import torch
import numpy as np
import pandas as pd

from src.numeric.datagen import sample_from_manifold
from src.numeric.losses import LossWeights
from src.numeric.sde_nets import DriftNet, DiffusionNet
from src.numeric.sde_training import SDEPipelineTrainer
from src.numeric.training import MultiModelTrainer, TrainingConfig, TrainingPhase

from experiments.common import make_model_config
from experiments.data_driven_sde import (
    create_manifold_sde, evaluate_pipeline, lambdify_sde,
    TRAIN_BOUND, N_TRAIN, BATCH_SIZE, LR_AE, LR_SDE, DEVICE,
)

SURFACES = ["sinusoidal", "paraboloid", "hyperbolic_paraboloid"]

# (label, loss_weights, use_true_projection, two_phase)
CONDITIONS = [
    ("T+F",           LossWeights(tangent_bundle=1.0, diffeo=1.0),                  False, False),
    ("T+F+K(Phat)2ph",  LossWeights(tangent_bundle=1.0, diffeo=1.0, curvature=0.1),   False, True),
    ("T+F+K(Pstar)2ph",  LossWeights(tangent_bundle=1.0, diffeo=1.0, curvature=0.1),   True,  True),
    ("T+F+K(Pstar)1ph",  LossWeights(tangent_bundle=1.0, diffeo=1.0, curvature=0.1),   True,  False),
]


def run_single(surface_name, cond_label, lw, use_true_proj, two_phase,
               seed, epochs_ae=500, epochs_sde=300):
    """Run full pipeline for one condition x seed. Returns metrics dict."""
    torch.manual_seed(seed)
    np.random.seed(seed)

    manifold_sde = create_manifold_sde(surface_name)
    train_data = sample_from_manifold(
        manifold_sde,
        [(-TRAIN_BOUND, TRAIN_BOUND), (-TRAIN_BOUND, TRAIN_BOUND)],
        n_samples=N_TRAIN, seed=seed, device=DEVICE,
    )
    x = train_data.samples.to(DEVICE)
    v = train_data.mu.to(DEVICE)
    Lambda = train_data.cov.to(DEVICE)

    # Stage 1: AE
    has_K = lw.curvature > 0
    warmup_lw = LossWeights(tangent_bundle=lw.tangent_bundle, diffeo=lw.diffeo)

    trainer = MultiModelTrainer(TrainingConfig(
        epochs=epochs_ae, n_samples=N_TRAIN, input_dim=3, hidden_dim=64,
        latent_dim=2, learning_rate=LR_AE, batch_size=BATCH_SIZE,
        test_size=0.03, print_interval=max(1, epochs_ae // 5), device=DEVICE,
    ))
    # For initial model creation, use warmup weights if two-phase, else full weights
    init_lw = warmup_lw if (has_K and two_phase) else lw
    mc = make_model_config("ae", init_lw, hidden_dims=[64, 64])
    mc.use_true_projection = use_true_proj
    trainer.add_model(mc)
    loader = trainer.create_data_loader(train_data)

    if has_K and two_phase:
        phase1_epochs = epochs_ae // 2
        phase2_epochs = epochs_ae - phase1_epochs
        schedule = [
            TrainingPhase(epochs=phase1_epochs, loss_weights=warmup_lw, name="T+F-warmup"),
            TrainingPhase(epochs=phase2_epochs, loss_weights=lw, name="T+F+K-finetune"),
        ]
        trainer.train_with_schedule(loader, "ae", schedule,
                                    print_interval=max(1, epochs_ae // 5))
    else:
        for epoch in range(epochs_ae):
            losses = trainer.train_epoch(loader)
            if (epoch + 1) % max(1, epochs_ae // 5) == 0:
                print(f"      AE Epoch {epoch+1}: loss={losses['ae']:.6f}")

    ae = trainer.models["ae"]
    ae.eval()
    final_losses = trainer.train_epoch(loader)
    ae_loss = final_losses["ae"]

    # Precompute
    d = 2
    dummy = SDEPipelineTrainer(
        ae, DriftNet(d).to(DEVICE), DiffusionNet(d).to(DEVICE), device=DEVICE,
    )
    z_pre, dphi_pre, d2phi_pre = dummy.precompute_decoder_derivatives(x)

    # Stage 2: Drift
    torch.manual_seed(seed + 100)
    drift_net = DriftNet(d).to(DEVICE)
    dp = SDEPipelineTrainer(ae, drift_net, DiffusionNet(d).to(DEVICE), device=DEVICE)
    drift_losses = dp.train_stage2_precomputed(
        z_pre, dphi_pre, d2phi_pre, v, Lambda,
        epochs=epochs_sde, lr=LR_SDE, batch_size=BATCH_SIZE,
        print_interval=max(1, epochs_sde // 5),
    )
    drift_net.eval()

    # Stage 3: Diffusion
    torch.manual_seed(seed + 200)
    diff_net = DiffusionNet(d).to(DEVICE)
    pipeline = SDEPipelineTrainer(ae, drift_net, diff_net, device=DEVICE)
    diff_losses = pipeline.train_stage3_precomputed(
        z_pre, dphi_pre, Lambda,
        epochs=epochs_sde, lr=LR_SDE, batch_size=BATCH_SIZE,
        print_interval=max(1, epochs_sde // 5),
    )

    # Evaluate
    sde = lambdify_sde(create_manifold_sde(surface_name))
    eval_results = evaluate_pipeline(pipeline, ae, sde, seed)

    return {
        "surface": surface_name,
        "condition": cond_label,
        "seed": seed,
        "ae_loss": ae_loss,
        "drift_loss": drift_losses[-1],
        "diff_loss": diff_losses[-1],
        **eval_results,
    }


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
    from scipy import stats
    p_val = stats.t.sf(abs(t_stat), df=n - 1) * 2
    return mean_d, p_val


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-seeds", type=int, default=5)
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--sde-epochs", type=int, default=300)
    parser.add_argument("--base-seed", type=int, default=42)
    args = parser.parse_args()

    seeds = [args.base_seed + i * 1000 for i in range(args.n_seeds)]
    cond_labels = [c[0] for c in CONDITIONS]
    print(f"Device: {DEVICE}")
    print(f"Seeds: {seeds}")
    print(f"Surfaces: {SURFACES}")
    print(f"Conditions: {cond_labels}")
    print(f"AE epochs: {args.epochs}, SDE epochs: {args.sde_epochs}\n")

    t0 = time.time()
    all_rows = []

    for surface_name in SURFACES:
        for seed in seeds:
            for cond_label, lw, use_true_proj, two_phase in CONDITIONS:
                print(f"\n{'='*60}")
                print(f"  {surface_name} | {cond_label} | seed={seed}")
                print(f"{'='*60}")

                row = run_single(
                    surface_name, cond_label, lw, use_true_proj, two_phase,
                    seed, epochs_ae=args.epochs, epochs_sde=args.sde_epochs,
                )
                all_rows.append(row)

                print(f"    MTE@1.0={row['MTE@1.0']:.4f}  W2@1.0={row['W2@1.0']:.4f}  "
                      f"MMD@1.0={row['MMD@1.0']:.4f}")

    df = pd.DataFrame(all_rows)
    csv_path = "multiseed_K_projection_study.csv"
    df.to_csv(csv_path, index=False)
    print(f"\nSaved raw results to {csv_path}")

    # Summary
    metrics = ["MTE@1.0", "W2@1.0", "MMD@1.0"]

    print(f"\n\n{'='*110}")
    print("K PROJECTION ABLATION RESULTS")
    print(f"{'='*110}")

    print(f"\n{'surface':>25s}  {'condition':>16s}  {'n':>3s}  ", end="")
    for m in metrics:
        print(f"{'mean':>8s} {'+-std':>8s}  ", end="")
    print()
    print("-" * 110)

    for surface_name in SURFACES:
        for cond_label in cond_labels:
            subset = df[(df["surface"] == surface_name) & (df["condition"] == cond_label)]
            n = len(subset)
            print(f"{surface_name:>25s}  {cond_label:>16s}  {n:>3d}  ", end="")
            for m in metrics:
                vals = subset[m].values
                print(f"{vals.mean():>8.4f} {vals.std():>8.4f}  ", end="")
            print()

    # Paired comparisons
    comparisons = [
        ("T+F+K(Phat)2ph", "T+F",           "K(Phat) vs baseline"),
        ("T+F+K(Pstar)2ph", "T+F",           "K(Pstar) vs baseline"),
        ("T+F+K(Pstar)2ph", "T+F+K(Phat)2ph",  "Pstar vs Phat (both 2ph)"),
        ("T+F+K(Pstar)1ph", "T+F+K(Pstar)2ph",  "1ph vs 2ph (both Pstar)"),
    ]

    print(f"\n\n{'='*110}")
    print("PAIRED COMPARISONS (negative delta = first condition is better)")
    print(f"{'='*110}")

    for cond_a, cond_b, desc in comparisons:
        print(f"\n  {desc}: {cond_a} vs {cond_b}")
        print(f"  {'surface':>25s}  ", end="")
        for m in metrics:
            print(f"{'delta':>8s} {'p-val':>8s}  ", end="")
        print("  verdict")
        print("  " + "-" * 100)

        for surface_name in SURFACES:
            a = df[(df["surface"] == surface_name) & (df["condition"] == cond_a)].sort_values("seed")
            b = df[(df["surface"] == surface_name) & (df["condition"] == cond_b)].sort_values("seed")

            print(f"  {surface_name:>25s}  ", end="")
            verdicts = []
            for m in metrics:
                mean_d, p_val = paired_ttest(a[m].values, b[m].values)
                sig = "**" if p_val < 0.01 else "*" if p_val < 0.05 else "+" if p_val < 0.1 else ""
                print(f"{mean_d:>+8.4f} {p_val:>7.4f}{sig:<1s} ", end="")
                if p_val < 0.05:
                    verdicts.append(f"{m}: {'worse' if mean_d > 0 else 'better'}")
            verdict_str = "; ".join(verdicts) if verdicts else "not significant"
            print(f"  {verdict_str}")

    print(f"\nTotal time: {time.time() - t0:.1f}s")
    print("Done.")


if __name__ == "__main__":
    main()
