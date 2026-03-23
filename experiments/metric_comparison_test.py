"""
Test alternative comparison metrics: latent-space W2, MFPT, local-coordinate comparison.

Hypothesis: ambient-space W2 penalizes chart mismatch, not just dynamics.
If we compare in latent coords (encode GT) or use MFPT, the T+F gap might vanish.

Usage:
    python -m experiments.metric_comparison_test --epochs 50 --sde-epochs 50  # smoke
    python -m experiments.metric_comparison_test                               # full
"""

import argparse
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

from experiments.data_driven_sde import (
    simulate_ground_truth, compute_w2, compute_mte_at_step,
    TRAIN_BOUND, BATCH_SIZE, LR_AE, LR_SDE, DEVICE,
    N_TRAJ, N_STEPS, DT, BOUNDARY, STOP_BOUND,
)
from experiments.highd_N_D_sweep import (
    hidden_dims_for_D, local_drift_fn, local_diffusion_fn,
)
from experiments.highd_baseline_comparison import _train_ae

NO_PENALTY_LW = LossWeights()
TF_LW = LossWeights(tangent_bundle=1.0, diffeo=1.0)
LAMBDA_SMOOTH = 0.5


def compute_latent_w2(z_learned, z_gt_encoded, alive):
    """W2 in latent space between learned z(t) and encoder(gt_traj(t))."""
    a = z_learned[alive]
    b = z_gt_encoded[alive]
    if len(a) < 5:
        return float('nan')
    from scipy.stats import wasserstein_distance
    # Component-wise W2 then sum (fast approximation)
    w2 = 0.0
    for dim in range(a.shape[1]):
        w2 += wasserstein_distance(
            a[:, dim].cpu().numpy(), b[:, dim].cpu().numpy()
        ) ** 2
    return w2 ** 0.5


def compute_local_coord_w2(local_learned, local_gt, alive):
    """W2 in local (u,v) coordinates."""
    a = local_learned[alive]
    b = local_gt[alive]
    if len(a) < 5:
        return float('nan')
    from scipy.stats import wasserstein_distance
    w2 = 0.0
    for dim in range(a.shape[1]):
        w2 += wasserstein_distance(
            a[:, dim].cpu().numpy(), b[:, dim].cpu().numpy()
        ) ** 2
    return w2 ** 0.5


def compute_mfpt(z_traj, radius):
    """Mean first passage time: average step when ||z(t) - z(0)|| > radius.

    Returns mean and std of first passage time across trajectories.
    """
    B, T, d = z_traj.shape
    z0 = z_traj[:, 0:1]  # (B, 1, d)
    displacements = ((z_traj - z0) ** 2).sum(-1).sqrt()  # (B, T)

    fpt = torch.full((B,), T - 1, dtype=torch.float, device=z_traj.device)
    for b in range(B):
        exits = (displacements[b] > radius).nonzero(as_tuple=True)[0]
        if len(exits) > 0:
            fpt[b] = exits[0].float()

    return fpt.mean().item(), fpt.std().item()


def evaluate_all_metrics(pipeline, ae, sde, seed, device):
    """Evaluate with ambient, latent, local-coord W2, and MFPT."""
    torch.manual_seed(seed + 999)
    init_local = (torch.rand(N_TRAJ, 2, device=device) * 2 - 1) * TRAIN_BOUND
    init_ambient = sde.chart(init_local).to(device)

    torch.manual_seed(seed + 1234)
    dW = torch.randn(N_TRAJ, N_STEPS, 2, device=device)

    # Ground truth
    gt_traj, gt_alive, gt_local = simulate_ground_truth(
        init_local, sde, N_STEPS, DT, dW, BOUNDARY,
    )

    # Learned trajectory
    with torch.no_grad():
        z0 = ae.encoder(init_ambient)
    z_traj, x_traj = pipeline.simulate(z0, N_STEPS, DT, dW=dW)

    step_final = int(round(1.0 / DT))
    both_alive = gt_alive[:, step_final]

    results = {}

    # 1. Standard ambient W2
    w2_ambient = compute_w2(
        x_traj[:, step_final], gt_traj[:, step_final],
        both_alive, both_alive,
    )
    results["W2_ambient"] = w2_ambient

    # 2. Ambient MTE
    mte = compute_mte_at_step(x_traj, gt_traj, both_alive, step_final)
    results["MTE_ambient"] = mte

    # 3. Latent W2: encode GT trajectory, compare z-space
    with torch.no_grad():
        gt_flat = gt_traj[:, step_final]  # (B, D)
        z_gt_encoded = ae.encoder(gt_flat)  # (B, d)
    z_learned_final = z_traj[:, step_final]

    w2_latent = compute_latent_w2(z_learned_final, z_gt_encoded, both_alive)
    results["W2_latent"] = w2_latent

    # 4. Latent MTE
    latent_err = ((z_learned_final - z_gt_encoded) ** 2).sum(-1)
    latent_mte = latent_err[both_alive].mean().sqrt().item() if both_alive.sum() > 0 else float('nan')
    results["MTE_latent"] = latent_mte

    # 5. Local-coordinate comparison:
    #    decode learned z, then invert through TRUE chart to get local coords.
    #    This only works approximately — use encoder as a proxy for the true chart inverse.
    #    Better: for GT, we have gt_local directly. For learned, use encoder(x_learned) ≈ local coords
    #    IF the encoder approximates the true chart inverse.
    #    Actually: we compare z_learned to z_gt_encoded — same as latent W2.
    #    For a TRUE local comparison, we'd need the exact chart inverse.

    # 6. MFPT in latent space (time to exit a ball of various radii)
    for radius in [0.5, 1.0, 2.0]:
        mfpt_learned, mfpt_std = compute_mfpt(z_traj, radius)
        # Also compute GT MFPT
        # Encode GT trajectory to latent for fair comparison
        with torch.no_grad():
            gt_flat_all = gt_traj.reshape(-1, gt_traj.shape[-1])
            z_gt_all = ae.encoder(gt_flat_all)
            z_gt_traj = z_gt_all.reshape(gt_traj.shape[0], gt_traj.shape[1], -1)
        mfpt_gt, mfpt_gt_std = compute_mfpt(z_gt_traj, radius)

        results[f"MFPT_learned_r{radius}"] = mfpt_learned
        results[f"MFPT_gt_r{radius}"] = mfpt_gt
        results[f"MFPT_err_r{radius}"] = abs(mfpt_learned - mfpt_gt)

    # 7. Per-step ambient vs latent MTE to see where they diverge
    for t_val in [0.1, 0.5, 1.0]:
        step = int(round(t_val / DT))
        alive = gt_alive[:, step]
        if alive.sum() < 5:
            results[f"MTE_amb@{t_val}"] = float('nan')
            results[f"MTE_lat@{t_val}"] = float('nan')
            continue

        amb_err = ((x_traj[:, step] - gt_traj[:, step]) ** 2).sum(-1)
        results[f"MTE_amb@{t_val}"] = amb_err[alive].mean().sqrt().item()

        with torch.no_grad():
            z_gt_step = ae.encoder(gt_traj[:, step])
        lat_err = ((z_traj[:, step] - z_gt_step) ** 2).sum(-1)
        results[f"MTE_lat@{t_val}"] = lat_err[alive].mean().sqrt().item()

    return results


def train_pipeline(ae, x, v, Lambda, seed, N, epochs_sde):
    """Train Stage 2 + 3, return pipeline."""
    d = 2
    batch_size = min(N, BATCH_SIZE)

    drift_net = DriftNet(d).to(DEVICE)
    diffusion_net = DiffusionNet(d).to(DEVICE)
    pipeline = SDEPipelineTrainer(ae, drift_net, diffusion_net, device=DEVICE)

    z_pre, dphi_pre, _ = pipeline.precompute_decoder_derivatives(x)
    b_z_target, g = pipeline.precompute_enc_pull_target(x, v, Lambda, dphi_pre)

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

    torch.manual_seed(seed + 200)
    diffusion_net2 = DiffusionNet(d).to(DEVICE)
    pipeline2 = SDEPipelineTrainer(ae, pipeline.drift_net, diffusion_net2, device=DEVICE)
    pipeline2.train_stage3_precomputed(
        z_pre, dphi_pre, Lambda.to(DEVICE),
        epochs=epochs_sde, lr=LR_SDE, batch_size=batch_size,
        print_interval=max(1, epochs_sde // 3),
    )

    return pipeline2


def main():
    parser = argparse.ArgumentParser()
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

    all_rows = []

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

            pipeline = train_pipeline(ae, x, v, Lambda, seed, args.N, args.sde_epochs)

            print(f"\n    Evaluating all metrics...")
            metrics = evaluate_all_metrics(pipeline, ae, sde, seed, DEVICE)

            for k, v_val in sorted(metrics.items()):
                print(f"    {k}: {v_val:.4f}")

            all_rows.append({"seed": seed, "condition": cond_label, **metrics})

    # ── Summary ───────────────────────────────────────────────────────
    print(f"\n\n{'='*80}")
    print("SUMMARY")
    print(f"{'='*80}")

    metric_keys = [k for k in all_rows[0].keys() if k not in ("seed", "condition")]

    print(f"\n  {'metric':<25s}  {'no_penalty':>16s}  {'T+F':>16s}  {'Δ%':>8s}")
    print(f"  {'-'*25}  {'-'*16}  {'-'*16}  {'-'*8}")

    for mk in metric_keys:
        np_vals = [r[mk] for r in all_rows if r["condition"] == "no_penalty"]
        tf_vals = [r[mk] for r in all_rows if r["condition"] == "T+F"]
        np_mean = np.nanmean(np_vals)
        tf_mean = np.nanmean(tf_vals)
        if abs(np_mean) > 1e-8:
            delta = (tf_mean - np_mean) / abs(np_mean) * 100
        else:
            delta = 0
        print(f"  {mk:<25s}  {np_mean:>7.4f}±{np.nanstd(np_vals):>.4f}  "
              f"{tf_mean:>7.4f}±{np.nanstd(tf_vals):>.4f}  {delta:>+7.1f}%")


if __name__ == "__main__":
    main()
