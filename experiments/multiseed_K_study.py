"""
Multi-seed study: Does curvature (K) regularization reliably help or hurt?

For each surface × AE config (T+F, T+F+K) × seed:
1. Train AE
2. Train drift_net + diffusion_net
3. Evaluate MTE, W2, MMD

Reports mean ± std across seeds, and paired t-test p-values.

Usage:
    PYTHONUNBUFFERED=1 python -m experiments.multiseed_K_study
    PYTHONUNBUFFERED=1 python -m experiments.multiseed_K_study --n-seeds 10
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

REG_CONFIGS = {
    "T+F":   LossWeights(tangent_bundle=1.0, diffeo=1.0),
    "T+F+K": LossWeights(tangent_bundle=1.0, diffeo=1.0, curvature=0.1),
}


def run_single_seed(surface_name, reg_name, lw, seed, epochs_ae=500, epochs_sde=300):
    """Run full pipeline for one surface × reg × seed. Returns metrics dict."""
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

    # Stage 1: AE (two-phase if K is active)
    has_K = lw.curvature > 0
    warmup_lw = LossWeights(tangent_bundle=lw.tangent_bundle, diffeo=lw.diffeo)
    trainer = MultiModelTrainer(TrainingConfig(
        epochs=epochs_ae, n_samples=N_TRAIN, input_dim=3, hidden_dim=64,
        latent_dim=2, learning_rate=LR_AE, batch_size=BATCH_SIZE,
        test_size=0.03, print_interval=max(1, epochs_ae // 5), device=DEVICE,
    ))
    trainer.add_model(make_model_config("ae", warmup_lw if has_K else lw, hidden_dims=[64, 64]))
    loader = trainer.create_data_loader(train_data)

    if has_K:
        # Phase 1: T+F warmup, Phase 2: T+F+K fine-tune
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
    # Get final loss by running one more forward pass
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
        "reg": reg_name,
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
    p_val = stats.t.sf(abs(t_stat), df=n - 1) * 2  # two-sided
    return mean_d, p_val


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-seeds", type=int, default=5)
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--sde-epochs", type=int, default=300)
    parser.add_argument("--base-seed", type=int, default=42)
    args = parser.parse_args()

    seeds = [args.base_seed + i * 1000 for i in range(args.n_seeds)]
    print(f"Device: {DEVICE}")
    print(f"Seeds: {seeds}")
    print(f"Surfaces: {SURFACES}")
    print(f"Configs: {list(REG_CONFIGS.keys())}")
    print(f"AE epochs: {args.epochs}, SDE epochs: {args.sde_epochs}\n")

    t0 = time.time()
    all_rows = []

    for surface_name in SURFACES:
        for seed in seeds:
            for reg_name, lw in REG_CONFIGS.items():
                print(f"\n{'='*60}")
                print(f"  {surface_name} | {reg_name} | seed={seed}")
                print(f"{'='*60}")

                row = run_single_seed(
                    surface_name, reg_name, lw, seed,
                    epochs_ae=args.epochs, epochs_sde=args.sde_epochs,
                )
                all_rows.append(row)

                print(f"    MTE@1.0={row['MTE@1.0']:.4f}  W2@1.0={row['W2@1.0']:.4f}  "
                      f"MMD@1.0={row['MMD@1.0']:.4f}")

    df = pd.DataFrame(all_rows)

    # Save raw results
    csv_path = "multiseed_K_study.csv"
    df.to_csv(csv_path, index=False)
    print(f"\nSaved raw results to {csv_path}")

    # ====== Summary ======
    print(f"\n\n{'='*100}")
    print("MULTI-SEED K STUDY RESULTS")
    print(f"{'='*100}")

    metrics = ["MTE@1.0", "W2@1.0", "MMD@1.0"]

    # Per-surface summary: mean ± std
    print(f"\n{'surface':>25s}  {'reg':>6s}  {'n':>3s}  ", end="")
    for m in metrics:
        print(f"{'mean':>8s} {'±std':>8s}  ", end="")
    print()
    print("-" * 90)

    for surface_name in SURFACES:
        for reg_name in REG_CONFIGS:
            subset = df[(df["surface"] == surface_name) & (df["reg"] == reg_name)]
            n = len(subset)
            print(f"{surface_name:>25s}  {reg_name:>6s}  {n:>3d}  ", end="")
            for m in metrics:
                vals = subset[m].values
                print(f"{vals.mean():>8.4f} {vals.std():>8.4f}  ", end="")
            print()

    # Paired comparison: T+F+K vs T+F
    print(f"\n\n{'='*100}")
    print("PAIRED COMPARISON: T+F+K vs T+F (negative delta = K helps)")
    print(f"{'='*100}")

    print(f"\n{'surface':>25s}  ", end="")
    for m in metrics:
        print(f"{'delta':>8s} {'p-val':>8s}  ", end="")
    print("  verdict")
    print("-" * 100)

    for surface_name in SURFACES:
        tf = df[(df["surface"] == surface_name) & (df["reg"] == "T+F")]
        tfk = df[(df["surface"] == surface_name) & (df["reg"] == "T+F+K")]

        # Match by seed for paired test
        tf_sorted = tf.sort_values("seed")
        tfk_sorted = tfk.sort_values("seed")

        print(f"{surface_name:>25s}  ", end="")
        verdicts = []
        for m in metrics:
            mean_d, p_val = paired_ttest(tfk_sorted[m].values, tf_sorted[m].values)
            sig = "*" if p_val < 0.05 else ""
            print(f"{mean_d:>+8.4f} {p_val:>8.4f}{sig} ", end="")
            if p_val < 0.05:
                verdicts.append(f"{m}: {'K hurts' if mean_d > 0 else 'K helps'}")
        verdict_str = "; ".join(verdicts) if verdicts else "not significant"
        print(f"  {verdict_str}")

    # Per-seed detail
    print(f"\n\n{'='*100}")
    print("PER-SEED DETAIL")
    print(f"{'='*100}")

    for surface_name in SURFACES:
        print(f"\n  {surface_name}:")
        tf = df[(df["surface"] == surface_name) & (df["reg"] == "T+F")].sort_values("seed")
        tfk = df[(df["surface"] == surface_name) & (df["reg"] == "T+F+K")].sort_values("seed")

        print(f"    {'seed':>6s}  {'T+F MTE':>8s}  {'T+F+K MTE':>10s}  {'delta':>8s}  "
              f"{'T+F W2':>8s}  {'T+F+K W2':>10s}  {'delta':>8s}")
        for (_, r_tf), (_, r_tfk) in zip(tf.iterrows(), tfk.iterrows()):
            d_mte = r_tfk["MTE@1.0"] - r_tf["MTE@1.0"]
            d_w2 = r_tfk["W2@1.0"] - r_tf["W2@1.0"]
            print(f"    {int(r_tf['seed']):>6d}  {r_tf['MTE@1.0']:>8.4f}  {r_tfk['MTE@1.0']:>10.4f}  "
                  f"{d_mte:>+8.4f}  {r_tf['W2@1.0']:>8.4f}  {r_tfk['W2@1.0']:>10.4f}  {d_w2:>+8.4f}")

    print(f"\nTotal time: {time.time() - t0:.1f}s")
    print("Done.")


if __name__ == "__main__":
    main()
