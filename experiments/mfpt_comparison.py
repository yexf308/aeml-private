"""
MFPT (Mean First Passage Time) comparison: no_penalty vs T+F.

MFPT = expected time to first exit a region. Intrinsic to the dynamics,
less sensitive to chart distortion than W2.

Two boundary definitions:
  1. Ambient ball:  first t where ||x(t) - x(0)||_R^D > r
  2. Local box:     first t where |u(t)| > L or |v(t)| > L  (GT coords)
     For learned traj: decode x(t), approximate local coords via encoder.

Usage:
    python -m experiments.mfpt_comparison --epochs 50 --sde-epochs 50   # smoke
    python -m experiments.mfpt_comparison                                # full
"""

import argparse
import torch
import numpy as np
from scipy import stats

from src.numeric.losses import LossWeights
from src.numeric.sde_nets import DriftNet, DiffusionNet
from src.numeric.sde_training import SDEPipelineTrainer

from src.numeric.highd_manifolds import (
    FourierAugmentedSurface,
    sample_from_highd_manifold,
    create_highd_lambdified_sde,
)

from experiments.data_driven_sde import (
    simulate_ground_truth,
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

# Use longer horizon for MFPT — more steps, same dt
MFPT_N_STEPS = 200  # T=2.0


def compute_mfpt_ambient_ball(traj, radii):
    """MFPT for exiting a ball of radius r centered at x(0) in ambient R^D.

    Args:
        traj: (B, T+1, D) trajectory in ambient space
        radii: list of radii to test

    Returns:
        dict: radius -> (mean_fpt, std_fpt) in time units
    """
    B, T1, D = traj.shape
    x0 = traj[:, 0:1]  # (B, 1, D)
    dists = ((traj - x0) ** 2).sum(-1).sqrt()  # (B, T+1)

    results = {}
    for r in radii:
        fpt = torch.full((B,), float('nan'), device=traj.device)
        for b in range(B):
            exits = (dists[b] > r).nonzero(as_tuple=True)[0]
            if len(exits) > 0:
                fpt[b] = exits[0].float() * DT
        # Only count trajectories that actually exited
        exited = torch.isfinite(fpt)
        n_exited = exited.sum().item()
        if n_exited >= 5:
            results[r] = {
                "mean": fpt[exited].mean().item(),
                "std": fpt[exited].std().item(),
                "exit_frac": n_exited / B,
            }
        else:
            results[r] = {
                "mean": float('nan'),
                "std": float('nan'),
                "exit_frac": n_exited / B,
            }
    return results


def compute_mfpt_local_box(local_traj, bounds):
    """MFPT for exiting a box |u| > L or |v| > L in local coordinates.

    Args:
        local_traj: (B, T+1, 2) trajectory in local (u,v) coords
        bounds: list of L values

    Returns:
        dict: L -> (mean_fpt, std_fpt) in time units
    """
    B, T1, d = local_traj.shape

    results = {}
    for L in bounds:
        fpt = torch.full((B,), float('nan'), device=local_traj.device)
        for b in range(B):
            outside = (local_traj[b].abs() > L).any(dim=-1)  # (T+1,)
            exits = outside.nonzero(as_tuple=True)[0]
            if len(exits) > 0:
                fpt[b] = exits[0].float() * DT
        exited = torch.isfinite(fpt)
        n_exited = exited.sum().item()
        if n_exited >= 5:
            results[L] = {
                "mean": fpt[exited].mean().item(),
                "std": fpt[exited].std().item(),
                "exit_frac": n_exited / B,
            }
        else:
            results[L] = {
                "mean": float('nan'),
                "std": float('nan'),
                "exit_frac": n_exited / B,
            }
    return results


def train_pipeline(ae, x, v, Lambda, seed, N, epochs_sde):
    """Train Stage 2 + 3."""
    d = 2
    batch_size = min(N, BATCH_SIZE)

    tmp = SDEPipelineTrainer(
        ae, DriftNet(d).to(DEVICE), DiffusionNet(d).to(DEVICE), device=DEVICE,
    )
    z_pre, dphi_pre, _ = tmp.precompute_decoder_derivatives(x)
    b_z_target, g = tmp.precompute_enc_pull_target(x, v, Lambda, dphi_pre)

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
    pipeline2 = SDEPipelineTrainer(ae, drift_net, diffusion_net2, device=DEVICE)
    pipeline2.train_stage3_precomputed(
        z_pre, dphi_pre, Lambda.to(DEVICE),
        epochs=epochs_sde, lr=LR_SDE, batch_size=batch_size,
        print_interval=max(1, epochs_sde // 3),
    )
    return pipeline2


def run_mfpt(pipeline, ae, sde, seed, n_steps):
    """Compute MFPT metrics for learned and GT trajectories."""
    torch.manual_seed(seed + 999)
    # Start near origin for cleaner MFPT
    init_local = (torch.rand(N_TRAJ, 2, device=DEVICE) * 2 - 1) * 0.5  # [-0.5, 0.5]
    init_ambient = sde.chart(init_local).to(DEVICE)

    torch.manual_seed(seed + 1234)
    dW = torch.randn(N_TRAJ, n_steps, 2, device=DEVICE)

    # GT trajectory (in local coords and ambient)
    gt_traj, gt_alive, gt_local = simulate_ground_truth(
        init_local, sde, n_steps, DT, dW, BOUNDARY * 2,  # wider boundary for MFPT
    )

    # Learned trajectory
    with torch.no_grad():
        z0 = ae.encoder(init_ambient)
    z_traj, x_traj = pipeline.simulate(z0, n_steps, DT, dW=dW,
                                        boundary=BOUNDARY * 2)

    # Ambient ball MFPT
    ambient_radii = [0.5, 1.0, 2.0, 3.0]
    gt_amb_mfpt = compute_mfpt_ambient_ball(gt_traj, ambient_radii)
    learned_amb_mfpt = compute_mfpt_ambient_ball(x_traj, ambient_radii)

    # Local box MFPT (GT has true local coords; learned uses encoder as proxy)
    local_bounds = [0.5, 1.0, 1.5]
    gt_local_mfpt = compute_mfpt_local_box(gt_local, local_bounds)

    # For learned: encode decoded trajectory to get approximate local coords
    with torch.no_grad():
        # Use encoder as proxy for local coords
        x_flat = x_traj.reshape(-1, x_traj.shape[-1])
        z_flat = ae.encoder(x_flat)
        z_traj_reencoded = z_flat.reshape(x_traj.shape[0], x_traj.shape[1], -1)
    learned_local_mfpt = compute_mfpt_local_box(z_traj_reencoded, local_bounds)

    return {
        "gt_amb": gt_amb_mfpt,
        "learned_amb": learned_amb_mfpt,
        "gt_local": gt_local_mfpt,
        "learned_local": learned_local_mfpt,
    }


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
    print(f"Seeds: {seeds}")
    print(f"MFPT horizon: T={MFPT_N_STEPS * DT:.1f}\n")

    all_results = {cond: [] for cond in ["no_penalty", "T+F"]}

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

            print(f"\n    Computing MFPT...")
            mfpt = run_mfpt(pipeline, ae, sde, seed, MFPT_N_STEPS)

            # Print
            print(f"\n    Ambient ball MFPT:")
            for r in [0.5, 1.0, 2.0, 3.0]:
                gt = mfpt["gt_amb"].get(r, {})
                lr = mfpt["learned_amb"].get(r, {})
                gt_m = gt.get("mean", float("nan"))
                lr_m = lr.get("mean", float("nan"))
                err = abs(lr_m - gt_m) / gt_m * 100 if gt_m > 0 else float("nan")
                print(f"      r={r:.1f}: GT={gt_m:.3f}  learned={lr_m:.3f}  "
                      f"err={err:.1f}%  exit_frac={lr.get('exit_frac', 0):.2f}")

            print(f"\n    Local box MFPT:")
            for L in [0.5, 1.0, 1.5]:
                gt = mfpt["gt_local"].get(L, {})
                lr = mfpt["learned_local"].get(L, {})
                gt_m = gt.get("mean", float("nan"))
                lr_m = lr.get("mean", float("nan"))
                err = abs(lr_m - gt_m) / gt_m * 100 if gt_m > 0 else float("nan")
                print(f"      L={L:.1f}: GT={gt_m:.3f}  learned={lr_m:.3f}  "
                      f"err={err:.1f}%  exit_frac={lr.get('exit_frac', 0):.2f}")

            all_results[cond_label].append(mfpt)

    # ── Summary ───────────────────────────────────────────────────────
    print(f"\n\n{'='*80}")
    print("MFPT SUMMARY (mean across seeds)")
    print(f"{'='*80}")

    print(f"\n  Ambient ball MFPT relative error |learned - GT| / GT:")
    print(f"  {'radius':>8s}  {'no_penalty':>14s}  {'T+F':>14s}")
    for r in [0.5, 1.0, 2.0, 3.0]:
        np_errs = []
        tf_errs = []
        for cond, errs_list in [("no_penalty", np_errs), ("T+F", tf_errs)]:
            for mfpt in all_results[cond]:
                gt_m = mfpt["gt_amb"].get(r, {}).get("mean", float("nan"))
                lr_m = mfpt["learned_amb"].get(r, {}).get("mean", float("nan"))
                if gt_m > 0 and np.isfinite(gt_m) and np.isfinite(lr_m):
                    errs_list.append(abs(lr_m - gt_m) / gt_m)
        np_err = np.mean(np_errs) * 100 if np_errs else float("nan")
        tf_err = np.mean(tf_errs) * 100 if tf_errs else float("nan")
        print(f"  r={r:>5.1f}  {np_err:>12.1f}%  {tf_err:>12.1f}%")

    print(f"\n  Local box MFPT relative error |learned - GT| / GT:")
    print(f"  {'bound':>8s}  {'no_penalty':>14s}  {'T+F':>14s}")
    for L in [0.5, 1.0, 1.5]:
        np_errs = []
        tf_errs = []
        for cond, errs_list in [("no_penalty", np_errs), ("T+F", tf_errs)]:
            for mfpt in all_results[cond]:
                gt_m = mfpt["gt_local"].get(L, {}).get("mean", float("nan"))
                lr_m = mfpt["learned_local"].get(L, {}).get("mean", float("nan"))
                if gt_m > 0 and np.isfinite(gt_m) and np.isfinite(lr_m):
                    errs_list.append(abs(lr_m - gt_m) / gt_m)
        np_err = np.mean(np_errs) * 100 if np_errs else float("nan")
        tf_err = np.mean(tf_errs) * 100 if tf_errs else float("nan")
        print(f"  L={L:>5.1f}  {np_err:>12.1f}%  {tf_err:>12.1f}%")

    # Raw MFPT values
    print(f"\n  Raw ambient MFPT values (mean across seeds):")
    print(f"  {'radius':>8s}  {'GT':>10s}  {'no_penalty':>12s}  {'T+F':>12s}")
    for r in [0.5, 1.0, 2.0, 3.0]:
        gts = []
        nps = []
        tfs = []
        for cond, vals in [("no_penalty", nps), ("T+F", tfs)]:
            for mfpt in all_results[cond]:
                vals.append(mfpt["learned_amb"].get(r, {}).get("mean", float("nan")))
                if cond == "no_penalty":
                    gts.append(mfpt["gt_amb"].get(r, {}).get("mean", float("nan")))
        # GT differs between conditions due to different encoders for local box,
        # but ambient GT should be the same
        print(f"  r={r:>5.1f}  {np.nanmean(gts):>10.3f}  "
              f"{np.nanmean(nps):>12.3f}  {np.nanmean(tfs):>12.3f}")


if __name__ == "__main__":
    main()
