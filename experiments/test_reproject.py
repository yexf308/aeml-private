"""
Quick test: does reprojection (encoder∘decoder after each EM step) close the
trajectory gap between T+F and unconstrained AE?

Runs D=11, N=50, paraboloid, 5 seeds.
Each seed trains both conditions and evaluates with/without reprojection.

Usage:
    python -m experiments.test_reproject --n-seeds 1 --epochs 50 --sde-epochs 50  # smoke
    python -m experiments.test_reproject                                           # full
"""

import argparse
import copy
import time
import torch
import numpy as np

from src.numeric.losses import LossWeights
from src.numeric.sde_nets import DriftNet, DiffusionNet
from src.numeric.sde_training import SDEPipelineTrainer
from src.numeric.geometry import regularized_metric_inverse

from src.numeric.highd_manifolds import (
    FourierAugmentedSurface,
    sample_from_highd_manifold,
    create_highd_lambdified_sde,
)

from experiments.common import make_model_config
from experiments.data_driven_sde import (
    simulate_ground_truth, compute_w2, compute_mte_at_step, compute_mmd,
    TRAIN_BOUND, BATCH_SIZE, LR_AE, LR_SDE, DEVICE,
    N_TRAJ, N_STEPS, DT, BOUNDARY,
)
from experiments.highd_N_D_sweep import (
    hidden_dims_for_D, local_drift_fn, local_diffusion_fn,
)
from experiments.highd_baseline_comparison import _train_ae

NO_PENALTY_LW = LossWeights()
TF_LW = LossWeights(tangent_bundle=1.0, diffeo=1.0)
LAMBDA_SMOOTH = 0.5


def evaluate_with_reproject(pipeline, ae, sde, seed, reproject):
    """Evaluate trajectory fidelity with or without reprojection."""
    torch.manual_seed(seed + 999)
    init_local = (torch.rand(N_TRAJ, 2, device=DEVICE) * 2 - 1) * TRAIN_BOUND
    init_ambient = sde.chart(init_local).to(DEVICE)

    torch.manual_seed(seed + 1234)
    dW = torch.randn(N_TRAJ, N_STEPS, 2, device=DEVICE)

    gt_traj, gt_alive, _ = simulate_ground_truth(
        init_local, sde, N_STEPS, DT, dW, BOUNDARY,
    )

    with torch.no_grad():
        z0 = ae.encoder(init_ambient)
    _, learned_traj = pipeline.simulate(z0, N_STEPS, DT, dW=dW, reproject=reproject)

    step_final = int(round(1.0 / DT))
    both_alive = gt_alive[:, step_final]

    w2 = compute_w2(learned_traj[:, step_final], gt_traj[:, step_final],
                     both_alive, both_alive)
    mte = compute_mte_at_step(learned_traj, gt_traj, both_alive, step_final)
    mmd = compute_mmd(learned_traj[:, step_final], gt_traj[:, step_final],
                       torch.ones(N_TRAJ, dtype=torch.bool, device=DEVICE),
                       gt_alive[:, step_final])

    # Also check MTE at t=0.5
    step_mid = int(round(0.5 / DT))
    both_mid = gt_alive[:, step_mid]
    mte_05 = compute_mte_at_step(learned_traj, gt_traj, both_mid, step_mid)

    return {"W2@1.0": w2, "MMD@1.0": mmd, "MTE@1.0": mte, "MTE@0.5": mte_05}


def train_pipeline(ae, x, v, Lambda, sde, surface, seed, N, epochs_sde):
    """Train Stage 2 + 3, return pipeline object."""
    d = 2
    batch_size = min(N, BATCH_SIZE)

    drift_net = DriftNet(d).to(DEVICE)
    diffusion_net = DiffusionNet(d).to(DEVICE)
    pipeline = SDEPipelineTrainer(ae, drift_net, diffusion_net, device=DEVICE)

    z_pre, dphi_pre, _ = pipeline.precompute_decoder_derivatives(x)
    b_z_target, g = pipeline.precompute_enc_pull_target(x, v, Lambda, dphi_pre)

    # Stage 2
    torch.manual_seed(seed + 100)
    drift_net = DriftNet(d).to(DEVICE)
    diffusion_net = DiffusionNet(d).to(DEVICE)
    pipeline = SDEPipelineTrainer(ae, drift_net, diffusion_net, device=DEVICE)
    pipeline.train_stage2_regression(
        z_pre, b_z_target, g,
        epochs=epochs_sde, lr=LR_SDE, batch_size=batch_size,
        print_interval=max(1, epochs_sde // 3),
        lambda_smooth=LAMBDA_SMOOTH, aug_sigma=0.1,
    )

    # Stage 3
    torch.manual_seed(seed + 200)
    diffusion_net2 = DiffusionNet(d).to(DEVICE)
    pipeline2 = SDEPipelineTrainer(ae, drift_net, diffusion_net2, device=DEVICE)
    pipeline2.train_stage3_precomputed(
        z_pre, dphi_pre, Lambda.to(DEVICE),
        epochs=epochs_sde, lr=LR_SDE, batch_size=batch_size,
        print_interval=max(1, epochs_sde // 3),
    )

    return pipeline2


def main():
    parser = argparse.ArgumentParser(description="Reprojection test")
    parser.add_argument("--n-seeds", type=int, default=5)
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--sde-epochs", type=int, default=300)
    parser.add_argument("--base-seed", type=int, default=42)
    parser.add_argument("--D", type=int, default=11)
    parser.add_argument("--N", type=int, default=50)
    parser.add_argument("--surface", type=str, default="paraboloid")
    args = parser.parse_args()

    seeds = [args.base_seed + i * 1000 for i in range(args.n_seeds)]
    print(f"Device: {DEVICE}")
    print(f"D={args.D}, N={args.N}, surface={args.surface}")
    print(f"Seeds: {seeds}\n")

    # Collect results: rows of (seed, condition, reproject, metrics)
    rows = []

    for seed in seeds:
        print(f"\n{'='*60}")
        print(f"  SEED {seed}")
        print(f"{'='*60}")

        torch.manual_seed(seed)
        np.random.seed(seed)

        surface = FourierAugmentedSurface(args.surface, args.D)
        train_data = sample_from_highd_manifold(
            surface, local_drift_fn, local_diffusion_fn,
            [(-TRAIN_BOUND, TRAIN_BOUND), (-TRAIN_BOUND, TRAIN_BOUND)],
            n_samples=args.N, seed=seed, device=DEVICE,
        )
        x = train_data.samples.to(DEVICE)
        v = train_data.mu.to(DEVICE)
        Lambda = train_data.cov.to(DEVICE)
        sde = create_highd_lambdified_sde(surface, local_drift_fn, local_diffusion_fn)

        for cond_label, lw in [("no_penalty", NO_PENALTY_LW), ("T+F", TF_LW)]:
            print(f"\n  --- {cond_label} ---")
            torch.manual_seed(seed)
            np.random.seed(seed)

            ae, recon = _train_ae(surface, args.D, seed, args.N, args.epochs, lw, train_data)
            print(f"    recon_per_dim={recon:.6f}")

            pipeline = train_pipeline(ae, x, v, Lambda, sde, surface, seed, args.N, args.sde_epochs)

            for reproj in [False, True]:
                tag = "reproject" if reproj else "standard"
                print(f"\n    Eval ({tag}):")
                metrics = evaluate_with_reproject(pipeline, ae, sde, seed, reproject=reproj)
                print(f"      W2={metrics['W2@1.0']:.4f}  MMD={metrics['MMD@1.0']:.4f}  "
                      f"MTE@0.5={metrics['MTE@0.5']:.4f}  MTE@1.0={metrics['MTE@1.0']:.4f}")

                rows.append({
                    "seed": seed, "condition": cond_label,
                    "reproject": reproj, **metrics,
                })

    # ── Summary ───────────────────────────────────────────────────────
    print(f"\n\n{'='*80}")
    print("SUMMARY (mean ± std across seeds)")
    print(f"{'='*80}\n")

    print(f"  {'condition':>12s} {'reproject':>10s}  {'W2@1.0':>14s}  {'MMD@1.0':>14s}  {'MTE@1.0':>14s}")
    print(f"  {'-'*12} {'-'*10}  {'-'*14}  {'-'*14}  {'-'*14}")

    for cond in ["no_penalty", "T+F"]:
        for reproj in [False, True]:
            subset = [r for r in rows if r["condition"] == cond and r["reproject"] == reproj]
            w2s = [r["W2@1.0"] for r in subset]
            mmds = [r["MMD@1.0"] for r in subset]
            mtes = [r["MTE@1.0"] for r in subset]
            tag = "yes" if reproj else "no"
            print(f"  {cond:>12s} {tag:>10s}  "
                  f"{np.mean(w2s):.4f}±{np.std(w2s):.4f}  "
                  f"{np.mean(mmds):.4f}±{np.std(mmds):.4f}  "
                  f"{np.mean(mtes):.4f}±{np.std(mtes):.4f}")

    # Does reprojection help T+F more than no_penalty?
    print(f"\n  Effect of reprojection (Δ% vs standard, negative = better):")
    for cond in ["no_penalty", "T+F"]:
        std = [r for r in rows if r["condition"] == cond and not r["reproject"]]
        rep = [r for r in rows if r["condition"] == cond and r["reproject"]]
        for metric in ["W2@1.0", "MMD@1.0", "MTE@1.0"]:
            std_vals = np.array([r[metric] for r in std])
            rep_vals = np.array([r[metric] for r in rep])
            delta_pct = ((rep_vals - std_vals) / std_vals * 100).mean()
            print(f"    {cond:>12s} {metric}: {delta_pct:+.1f}%")

    # Does T+F+reproject beat no_penalty standard?
    print(f"\n  T+F(reproject) vs no_penalty(standard):")
    tf_rep = [r for r in rows if r["condition"] == "T+F" and r["reproject"]]
    np_std = [r for r in rows if r["condition"] == "no_penalty" and not r["reproject"]]
    for metric in ["W2@1.0", "MMD@1.0", "MTE@1.0"]:
        tf_vals = np.array([r[metric] for r in tf_rep])
        np_vals = np.array([r[metric] for r in np_std])
        delta_pct = ((tf_vals.mean() - np_vals.mean()) / np_vals.mean() * 100)
        print(f"    {metric}: {delta_pct:+.1f}%")


if __name__ == "__main__":
    main()
