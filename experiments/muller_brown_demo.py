"""
Müller-Brown potential on a paraboloid manifold: physically motivated demo.

Overdamped Langevin dynamics in the (rescaled) Müller-Brown potential,
on a paraboloid surface embedded in R^D via Fourier augmentation.
The manifold geometry (curvature) comes from the surface; the dynamics
(metastable drift) come from the MB potential — independent, as in
constrained molecular dynamics.

Metrics:
  - Chart quality: reconstruction, tangent error, E_μ, σ_min
  - Inter-well MFPT: first passage time from well 1 to well 2 or 3
  - Transition rate: #transitions between wells / total time

Usage:
    python -m experiments.muller_brown_demo --epochs 50 --sde-epochs 50 --n-seeds 1  # smoke
    python -m experiments.muller_brown_demo                                            # full
"""

import argparse
import math
import time
import torch
import numpy as np
import pandas as pd

from src.numeric.losses import LossWeights
from src.numeric.geometry import compute_sigma_min
from src.numeric.performance_stats import compute_losses_per_sample
from src.numeric.sde_nets import DriftNet, DiffusionNet
from src.numeric.sde_training import SDEPipelineTrainer

from src.numeric.highd_manifolds import (
    FourierAugmentedSurface,
    sample_from_highd_manifold,
    create_highd_lambdified_sde,
)

from experiments.data_driven_sde import (
    simulate_ground_truth, compute_w2,
    BATCH_SIZE, LR_AE, LR_SDE, DEVICE,
    N_TRAJ, DT, BOUNDARY,
)
from experiments.highd_N_D_sweep import (
    hidden_dims_for_D, N_EVAL, compute_coefficient_errors,
)
from experiments.highd_baseline_comparison import _train_ae
from experiments.mfpt_full_ablation import train_pipeline


# ── Müller-Brown dynamics ────────────────────────────────────────────────

_A = torch.tensor([-200., -100., -170., 15.])
_a = torch.tensor([-1., -1., -6.5, 0.7])
_b = torch.tensor([0., 0., 11., 0.6])
_c = torch.tensor([-10., -10., -6.5, 0.7])
_x0 = torch.tensor([1., 0., -0.5, -1.])
_y0 = torch.tensor([0., 0.5, 1.5, 1.])

KT = 0.15
V_SCALE = 200.0

# Three MB wells in (u,v) ∈ [-1,1]² coordinates
WELLS_UV = torch.tensor([
    [-0.30,  0.55],   # well 1 (deepest, V≈-0.73)
    [ 0.57, -0.58],   # well 2 (V≈-0.54)
    [ 0.07, -0.23],   # well 3 (shallowest, V≈-0.40)
])

DEBOUNCE_STEPS = 5    # must stay in new well this many steps to count
WELL_NAMES = ["W1", "W2", "W3"]


def mb_potential(uv: torch.Tensor) -> torch.Tensor:
    """Rescaled MB potential V(u,v)/V_SCALE in [-1,1]² coordinates."""
    u, v = uv[..., 0], uv[..., 1]
    x = (2.7 * u - 0.3) / 2
    y = (2.5 * v + 1.5) / 2
    dev = uv.device
    A, a, b, c = _A.to(dev), _a.to(dev), _b.to(dev), _c.to(dev)
    x0, y0 = _x0.to(dev), _y0.to(dev)
    V = torch.zeros_like(x)
    for i in range(4):
        V = V + A[i] * torch.exp(
            a[i] * (x - x0[i])**2
            + b[i] * (x - x0[i]) * (y - y0[i])
            + c[i] * (y - y0[i])**2
        )
    return V / V_SCALE


def mb_local_drift_fn(uv: torch.Tensor) -> torch.Tensor:
    """Overdamped Langevin drift: -∇V(u,v). Input (B,2) -> (B,2)."""
    uv_g = uv.detach().requires_grad_(True)
    V = mb_potential(uv_g).sum()
    grad = torch.autograd.grad(V, uv_g)[0]
    return -grad.detach()


def mb_local_diffusion_fn(uv: torch.Tensor) -> torch.Tensor:
    """Isotropic diffusion: √(2kT) I₂. Input (B,2) -> (B,2,2)."""
    B = uv.shape[0]
    sigma = math.sqrt(2 * KT)
    return sigma * torch.eye(2, device=uv.device).unsqueeze(0).expand(B, -1, -1)


# ── Inter-well MFPT and transition rate ──────────────────────────────────

def assign_wells_ambient(traj: torch.Tensor, well_centers_ambient: torch.Tensor) -> torch.Tensor:
    """Voronoi assignment of trajectory points to nearest well center.

    Args:
        traj: (B, T+1, D) ambient trajectory
        well_centers_ambient: (3, D) ambient coordinates of well centers

    Returns:
        assignment: (B, T+1) integer well index (0, 1, or 2)
    """
    # (B, T+1, 1, D) - (1, 1, 3, D) -> (B, T+1, 3)
    dists = torch.cdist(traj, well_centers_ambient.unsqueeze(0).expand(traj.shape[0], -1, -1))
    return dists.argmin(dim=-1)


def compute_interwell_mfpt(assignments: torch.Tensor, start_well: int, dt: float) -> dict:
    """First passage time from start_well to any other well (with debounce).

    Args:
        assignments: (B, T+1) well assignments
        start_well: starting well index
        dt: time step

    Returns:
        dict with 'mean', 'std', 'transition_frac', 'target_counts'
    """
    B, T1 = assignments.shape
    fpt = torch.full((B,), float('nan'))
    targets = []

    for b in range(B):
        consecutive = 0
        candidate_well = -1
        for t in range(1, T1):
            w = assignments[b, t].item()
            if w != start_well:
                if w == candidate_well:
                    consecutive += 1
                else:
                    candidate_well = w
                    consecutive = 1
                if consecutive >= DEBOUNCE_STEPS:
                    fpt[b] = (t - DEBOUNCE_STEPS + 1) * dt
                    targets.append(candidate_well)
                    break
            else:
                candidate_well = -1
                consecutive = 0

    transitioned = torch.isfinite(fpt)
    n_trans = transitioned.sum().item()

    result = {
        "mean": fpt[transitioned].mean().item() if n_trans >= 2 else float('nan'),
        "std": fpt[transitioned].std().item() if n_trans >= 2 else float('nan'),
        "transition_frac": n_trans / B,
    }
    # Count transitions to each target well
    for i, name in enumerate(WELL_NAMES):
        if i != start_well:
            result[f"frac_to_{name}"] = targets.count(i) / max(n_trans, 1)

    return result


def compute_transition_rate(assignments: torch.Tensor, dt: float) -> dict:
    """Count transitions between wells per unit time (with debounce).

    Args:
        assignments: (B, T+1) well assignments
        dt: time step

    Returns:
        dict with 'rate_per_traj' (transitions per trajectory per unit time),
        'total_transitions', 'total_time'
    """
    B, T1 = assignments.shape
    total_time = B * (T1 - 1) * dt
    total_transitions = 0

    for b in range(B):
        current_well = assignments[b, 0].item()
        consecutive = 0
        candidate_well = -1
        for t in range(1, T1):
            w = assignments[b, t].item()
            if w != current_well:
                if w == candidate_well:
                    consecutive += 1
                else:
                    candidate_well = w
                    consecutive = 1
                if consecutive >= DEBOUNCE_STEPS:
                    total_transitions += 1
                    current_well = candidate_well
                    consecutive = 0
                    candidate_well = -1
            else:
                candidate_well = -1
                consecutive = 0

    return {
        "total_transitions": total_transitions,
        "total_time": total_time,
        "rate": total_transitions / total_time if total_time > 0 else 0,
        "transitions_per_traj": total_transitions / B,
    }


# ── Conditions ───────────────────────────────────────────────────────────

CONDITIONS = {
    "baseline": LossWeights(),
    "T+F":      LossWeights(tangent_bundle=1.0, diffeo=1.0),
}

TRAIN_BOUND = 0.8
N_TRAIN = 50
LONG_T = 20.0       # simulation time for inter-well metrics
LONG_N_STEPS = 2000  # = LONG_T / DT


# ── Main ─────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-seeds", type=int, default=5)
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--sde-epochs", type=int, default=300)
    parser.add_argument("--base-seed", type=int, default=42)
    parser.add_argument("--D", type=int, default=11)
    parser.add_argument("--output", type=str, default="muller_brown_demo.csv")
    args = parser.parse_args()

    seeds = [args.base_seed + i * 1000 for i in range(args.n_seeds)]

    print(f"Device: {DEVICE}")
    print(f"D={args.D}, N={N_TRAIN}, kT={KT}")
    print(f"Surface: paraboloid (Fourier-augmented)")
    print(f"Dynamics: Müller-Brown overdamped Langevin")
    print(f"Conditions: {list(CONDITIONS.keys())}")
    print(f"Seeds ({len(seeds)}): {seeds}")
    print(f"LONG_T={LONG_T}, LONG_N_STEPS={LONG_N_STEPS}")
    print(f"TRAIN_BOUND={TRAIN_BOUND}")
    print(f"Well centers (u,v): {WELLS_UV.tolist()}\n")

    t0 = time.time()
    all_rows = []

    surface = FourierAugmentedSurface("paraboloid", args.D)
    sde = create_highd_lambdified_sde(surface, mb_local_drift_fn, mb_local_diffusion_fn)

    # Well centers in ambient space
    wells_ambient = sde.chart(WELLS_UV.to(DEVICE)).to(DEVICE)  # (3, D)
    print(f"Well centers ambient norms: {wells_ambient.norm(dim=-1).tolist()}")

    for seed in seeds:
        print(f"\n{'='*60}")
        print(f"  seed={seed}")
        print(f"{'='*60}")

        torch.manual_seed(seed)
        np.random.seed(seed)

        train_data = sample_from_highd_manifold(
            surface, mb_local_drift_fn, mb_local_diffusion_fn,
            [(-TRAIN_BOUND, TRAIN_BOUND), (-TRAIN_BOUND, TRAIN_BOUND)],
            n_samples=N_TRAIN, seed=seed, device=DEVICE,
        )
        x = train_data.samples.to(DEVICE)
        v = train_data.mu.to(DEVICE)
        Lambda = train_data.cov.to(DEVICE)

        # Initial conditions at well 1 center (+ small noise)
        torch.manual_seed(seed + 999)
        init_local = torch.randn(N_TRAJ, 2, device=DEVICE) * 0.1
        init_local[:, 0] += WELLS_UV[0, 0]
        init_local[:, 1] += WELLS_UV[0, 1]
        init_local.clamp_(-TRAIN_BOUND, TRAIN_BOUND)
        init_ambient = sde.chart(init_local).to(DEVICE)

        # GT Langevin trajectories (independent noise)
        torch.manual_seed(seed + 1234)
        dW_gt = torch.randn(N_TRAJ, LONG_N_STEPS, 2, device=DEVICE)
        gt_traj, gt_alive, _ = simulate_ground_truth(
            init_local, sde, LONG_N_STEPS, DT, dW_gt, BOUNDARY * 3,
        )
        gt_assign = assign_wells_ambient(gt_traj, wells_ambient)
        gt_mfpt = compute_interwell_mfpt(gt_assign, start_well=0, dt=DT)
        gt_rate = compute_transition_rate(gt_assign, dt=DT)
        print(f"  GT inter-well MFPT: {gt_mfpt['mean']:.3f}±{gt_mfpt['std']:.3f} "
              f"(frac={gt_mfpt['transition_frac']:.0%})")
        print(f"  GT transition rate: {gt_rate['rate']:.4f}/t "
              f"({gt_rate['total_transitions']} transitions in {gt_rate['total_time']:.0f}t)")

        for cond_label, lw in CONDITIONS.items():
            print(f"\n  --- {cond_label} ---")
            torch.manual_seed(seed)
            np.random.seed(seed)

            ae, recon = _train_ae(surface, args.D, seed, N_TRAIN, args.epochs, lw, train_data)
            print(f"    recon={recon:.6f}")

            # σ_min
            torch.manual_seed(seed + 7000)
            eval_uv_in = (torch.rand(300, 2, device=DEVICE) * 2 - 1) * TRAIN_BOUND
            x_in = sde.chart(eval_uv_in).to(DEVICE)
            sigma_min_in = compute_sigma_min(ae, x_in)
            print(f"    σ_min: min={sigma_min_in.min():.3f} med={sigma_min_in.median():.3f}")

            # Tangent error
            torch.manual_seed(seed + 9000)
            test_data = sample_from_highd_manifold(
                surface, mb_local_drift_fn, mb_local_diffusion_fn,
                [(-TRAIN_BOUND, TRAIN_BOUND), (-TRAIN_BOUND, TRAIN_BOUND)],
                n_samples=300, seed=seed + 9000, device=DEVICE,
            )
            losses_df = compute_losses_per_sample(ae, test_data)
            tangent_err = losses_df["tangent"].mean()
            print(f"    tangent_err={tangent_err:.6f}")

            # Coefficient errors
            torch.manual_seed(seed + 5000)
            eval_uv = (torch.rand(N_EVAL, 2, device=DEVICE) * 2 - 1) * TRAIN_BOUND
            e_mu, e_sigma = compute_coefficient_errors(ae, sde, surface, eval_uv, DEVICE)
            print(f"    E_mu={e_mu:.4f}  E_Sigma={e_sigma:.4f}")

            # SDE pipeline
            pipeline = train_pipeline(ae, x, v, Lambda, seed, N_TRAIN, args.sde_epochs)

            # Learned Langevin trajectories (independent noise)
            torch.manual_seed(seed + 5678 + hash(cond_label) % 10000)
            dW_lr = torch.randn(N_TRAJ, LONG_N_STEPS, 2, device=DEVICE)
            with torch.no_grad():
                z0 = ae.encoder(init_ambient)
            _, x_traj = pipeline.simulate(z0, LONG_N_STEPS, DT, dW=dW_lr)

            # Inter-well MFPT
            lr_assign = assign_wells_ambient(x_traj, wells_ambient)
            lr_mfpt = compute_interwell_mfpt(lr_assign, start_well=0, dt=DT)
            lr_rate = compute_transition_rate(lr_assign, dt=DT)

            print(f"    Inter-well MFPT: {lr_mfpt['mean']:.3f}±{lr_mfpt['std']:.3f} "
                  f"(frac={lr_mfpt['transition_frac']:.0%})")
            print(f"    Transition rate: {lr_rate['rate']:.4f}/t "
                  f"({lr_rate['total_transitions']} transitions in {lr_rate['total_time']:.0f}t)")

            # MFPT error
            if np.isfinite(gt_mfpt['mean']) and gt_mfpt['mean'] > 0 and np.isfinite(lr_mfpt['mean']):
                mfpt_err = abs(lr_mfpt['mean'] - gt_mfpt['mean']) / gt_mfpt['mean']
            else:
                mfpt_err = float('nan')

            # Rate error
            if gt_rate['rate'] > 0 and lr_rate['rate'] > 0:
                rate_err = abs(lr_rate['rate'] - gt_rate['rate']) / gt_rate['rate']
            else:
                rate_err = float('nan')

            row = {
                "condition": cond_label,
                "seed": seed,
                "recon_per_dim": recon,
                "tangent_err": tangent_err,
                "E_mu": e_mu,
                "E_Sigma": e_sigma,
                "sigma_min": sigma_min_in.min().item(),
                "gt_mfpt": gt_mfpt["mean"],
                "lr_mfpt": lr_mfpt["mean"],
                "mfpt_err": mfpt_err,
                "transition_frac": lr_mfpt["transition_frac"],
                "gt_transition_frac": gt_mfpt["transition_frac"],
                "gt_rate": gt_rate["rate"],
                "lr_rate": lr_rate["rate"],
                "rate_err": rate_err,
                "gt_transitions": gt_rate["total_transitions"],
                "lr_transitions": lr_rate["total_transitions"],
            }
            all_rows.append(row)

            if np.isfinite(mfpt_err):
                print(f"    MFPT err: {mfpt_err:.1%}")
            if np.isfinite(rate_err):
                print(f"    Rate err: {rate_err:.1%}")

        pd.DataFrame(all_rows).to_csv(args.output, index=False)

    elapsed = time.time() - t0
    print(f"\n\nTotal time: {elapsed / 60:.1f} min")

    df = pd.DataFrame(all_rows)
    df.to_csv(args.output, index=False)
    print(f"Saved {len(df)} rows to {args.output}")

    # ── Summary ──────────────────────────────────────────────────────
    print(f"\n\n{'='*70}")
    print("MÜLLER-BROWN DEMO SUMMARY (paraboloid, D=%d)" % args.D)
    print(f"{'='*70}")

    for cond in CONDITIONS:
        sub = df[df["condition"] == cond]
        print(f"\n  {cond}:")
        print(f"    tangent:  {sub['tangent_err'].mean():.4f} ± {sub['tangent_err'].std():.4f}")
        print(f"    E_mu:     {sub['E_mu'].mean():.4f} ± {sub['E_mu'].std():.4f}")
        print(f"    σ_min:    {sub['sigma_min'].mean():.3f} ± {sub['sigma_min'].std():.3f}")
        mfpt_err = sub["mfpt_err"].values
        print(f"    MFPT err: {np.nanmean(mfpt_err):.1%} ± {np.nanstd(mfpt_err):.1%}")
        print(f"    Trans frac: {sub['transition_frac'].mean():.1%} "
              f"(GT: {sub['gt_transition_frac'].mean():.1%})")
        rate_err = sub["rate_err"].values
        print(f"    Rate err: {np.nanmean(rate_err):.1%} ± {np.nanstd(rate_err):.1%}")
        print(f"    Rate: {sub['lr_rate'].mean():.4f} (GT: {sub['gt_rate'].mean():.4f})")


if __name__ == "__main__":
    main()
