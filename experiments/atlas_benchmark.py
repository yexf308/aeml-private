"""
ATLAS benchmark: local-chart model reduction baseline.

Implements the ATLAS algorithm (Ye-Yang-Maggioni) for comparison
with the AE-based pipeline. ATLAS estimates drift and diffusion
from short trajectory bursts at landmark points, builds a
Gaussian-blended atlas of local charts, and simulates in ambient
space using the blended coefficients.

Reference: "Nonlinear model reduction for slow-fast stochastic
systems near manifolds" (Ye, Yang, Maggioni).

Two modes:
  - burst mode: estimate (b, Λ) from short trajectory bursts
    (the standard ATLAS procedure)
  - oracle mode: use exact (b, Λ) at landmarks (fair comparison
    with our pipeline, which also uses exact coefficients)
"""

import argparse
import time
import torch
import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import List

from experiments.data_driven_sde import DEVICE, N_TRAJ, simulate_ground_truth
from experiments.mb_dynamics import (
    mb_local_drift_fn, mb_local_diffusion_fn,
    WELLS_UV, TRAIN_BOUND, importance_sample_mb,
    assign_wells_coreset_2d, compute_pairwise_mfpt,
    compute_implied_timescales, compute_stationary_probabilities,
)
from experiments.mb_mfpt import MB_DT, LONG_T, LONG_N_STEPS, LAG_TIMES, CENSOR_BOUND
from experiments.highd_N_D_sweep import compute_coefficient_errors, N_EVAL
from src.numeric.highd_manifolds import (
    FourierAugmentedSurface, sample_from_highd_manifold,
    create_highd_lambdified_sde,
)


# ── ATLAS Chart ──────────────────────────────────────────────────────────

@dataclass
class ATLASChart:
    """Local chart storing drift, diffusion, tangent basis."""
    center: torch.Tensor    # (D,) landmark position in ambient space
    U: torch.Tensor         # (D, d) tangent basis (top d left-singular vectors of Λ)
    sigma: torch.Tensor     # (d,) singular values (sqrt of eigenvalues of Λ)
    b: torch.Tensor         # (D,) ambient drift at landmark
    Lambda: torch.Tensor    # (D, D) ambient diffusion covariance (rank-d projected)


def build_atlas_from_data(x, v, Lambda, d=2, device=DEVICE):
    """Build ATLAS charts from precomputed ambient data.

    This is the 'oracle mode': exact (b, Λ) are given at each landmark,
    matching the data our AE pipeline receives.

    Args:
        x: Ambient landmark positions, shape (N, D).
        v: Ambient drift at landmarks, shape (N, D).
        Lambda: Ambient covariance at landmarks, shape (N, D, D).
        d: Intrinsic dimension.

    Returns:
        charts: List of ATLASChart.
    """
    N, D = x.shape
    charts = []
    for i in range(N):
        # SVD of Lambda_i to get tangent basis
        U_full, S_full, _ = torch.linalg.svd(Lambda[i])
        U_d = U_full[:, :d]             # (D, d)
        sigma_d = S_full[:d].sqrt()     # (d,)
        # Project Lambda to rank d
        Lambda_d = U_d @ torch.diag(S_full[:d]) @ U_d.T

        charts.append(ATLASChart(
            center=x[i],
            U=U_d,
            sigma=sigma_d,
            b=v[i],
            Lambda=Lambda_d,
        ))
    return charts


def build_atlas_from_bursts(sde, landmarks_uv, D, d=2, n_bursts=500,
                            burst_dt=0.001, burst_steps=50, device=DEVICE):
    """Build ATLAS charts by estimating (b, Λ) from short trajectory bursts.

    This is the standard ATLAS procedure: at each landmark, run n_bursts
    short trajectories, then estimate drift and diffusion via linear
    regression of conditional mean and covariance against time.

    Args:
        sde: LambdifiedSDE with .chart(), .drift(), .diffusion().
        landmarks_uv: (N, 2) local coordinates of landmarks.
        D: Ambient dimension.
        d: Intrinsic dimension.
        n_bursts: Number of short trajectories per landmark.
        burst_dt: Time step for bursts.
        burst_steps: Number of steps per burst.

    Returns:
        charts: List of ATLASChart.
    """
    N = landmarks_uv.shape[0]
    charts = []

    for i in range(N):
        uv_i = landmarks_uv[i:i+1].expand(n_bursts, -1).to(device)
        x0 = sde.chart(uv_i).to(device)  # (n_bursts, D)

        # Run short bursts
        dW = torch.randn(n_bursts, burst_steps, 2, device=device)
        _, _, local_traj = simulate_ground_truth(
            uv_i, sde, burst_steps, burst_dt, dW, boundary=None,
        )
        # local_traj: (n_bursts, burst_steps+1, 2)
        # Lift to ambient
        amb_traj = torch.zeros(n_bursts, burst_steps + 1, D, device=device)
        for t in range(burst_steps + 1):
            amb_traj[:, t] = sde.chart(local_traj[:, t])

        # Estimate drift and diffusion via linear regression
        # m(t) = E[X(t) - X(0)], Cov(t) = E[(X(t)-X(0))(X(t)-X(0))^T]
        t_span = torch.arange(1, burst_steps + 1, device=device).float() * burst_dt
        t_bar = t_span - t_span.mean()

        dX = amb_traj[:, 1:] - amb_traj[:, 0:1]  # (n_bursts, burst_steps, D)
        mean_dX = dX.mean(dim=0)  # (burst_steps, D)

        # Drift: linear regression of mean_dX against t
        mean_bar = mean_dX - mean_dX.mean(dim=0, keepdim=True)
        b_est = (mean_bar * t_bar.unsqueeze(-1)).sum(dim=0) / (t_bar ** 2).sum()

        # Covariance at each time step
        cov_list = []
        for t in range(burst_steps):
            dx_t = dX[:, t]  # (n_bursts, D)
            cov_t = (dx_t.unsqueeze(-1) @ dx_t.unsqueeze(-2)).mean(dim=0)  # (D, D)
            cov_list.append(cov_t)
        cov_stack = torch.stack(cov_list)  # (burst_steps, D, D)
        cov_bar = cov_stack - cov_stack.mean(dim=0, keepdim=True)
        Lambda_est = (cov_bar * t_bar.view(-1, 1, 1)).sum(dim=0) / (t_bar ** 2).sum()

        # SVD to get tangent basis
        U_full, S_full, _ = torch.linalg.svd(Lambda_est)
        # Take only positive eigenvalues
        S_full = S_full.clamp(min=0)
        U_d = U_full[:, :d]
        sigma_d = S_full[:d].sqrt()
        Lambda_d = U_d @ torch.diag(S_full[:d]) @ U_d.T

        # Landmark position = mean of X(0)
        x0_mean = x0.mean(dim=0)

        charts.append(ATLASChart(
            center=x0_mean,
            U=U_d,
            sigma=sigma_d,
            b=b_est,
            Lambda=Lambda_d,
        ))

    return charts


# ── ATLAS Simulation ─────────────────────────────────────────────────────

def atlas_weighted_drift_diffusion(x, charts, d, t0=0.001, chi_p=1.0):
    """Compute Gaussian-blended drift and diffusion at point x.

    Following weighted_drift_diffusion2.m: for each neighboring chart,
    compute Mahalanobis-like distance T_k, weight = exp(-T_k), then
    blend drift and diffusion.

    Args:
        x: Current position, shape (B, D).
        charts: List of ATLASChart.
        d: Intrinsic dimension.
        t0: Scale parameter for Mahalanobis distance.
        chi_p: Normalization for Mahalanobis distance.

    Returns:
        x_proj: Projected position, shape (B, D).
        b_blend: Blended drift, shape (B, D).
        H_blend: Blended diffusion sqrt, shape (B, D, d).
    """
    B, D = x.shape
    device = x.device
    N_charts = len(charts)

    # Stack chart data for vectorized computation
    centers = torch.stack([c.center for c in charts]).to(device)  # (N, D)
    all_U = torch.stack([c.U for c in charts]).to(device)         # (N, D, d)
    all_sigma = torch.stack([c.sigma for c in charts]).to(device)  # (N, d)
    all_b = torch.stack([c.b for c in charts]).to(device)         # (N, D)

    # Compute Mahalanobis distance from x to each chart
    # x_coeff = (x - center) @ U, then t = sum(x_coeff^2 / sigma^2) / chi_p
    dx = x.unsqueeze(1) - centers.unsqueeze(0)  # (B, N, D)
    x_coeff = torch.bmm(
        dx.reshape(B * N_charts, 1, D),
        all_U.unsqueeze(0).expand(B, -1, -1, -1).reshape(B * N_charts, D, d),
    ).reshape(B, N_charts, d)  # (B, N, d)

    t_val = (x_coeff ** 2 / (all_sigma.unsqueeze(0) ** 2 + 1e-10)).sum(dim=-1) / chi_p
    T_k = torch.sqrt(t_val / t0).clamp(max=10.0)  # (B, N)
    weights = torch.exp(-T_k)  # (B, N)

    # Weighted projection: project x onto each chart's tangent plane
    # x_local_k = x_coeff_k @ U_k^T + center_k
    # x_coeff: (B, N, d), U: (N, D, d) → U^T: (N, d, D)
    x_local = torch.einsum('bni,nDi->bnD', x_coeff, all_U)  # (B, N, D)
    x_local = x_local + centers.unsqueeze(0)  # (B, N, D)
    x_local_weighted = x_local * weights.unsqueeze(-1)  # (B, N, D)

    weight_sum = weights.sum(dim=1, keepdim=True).clamp(min=1e-10)  # (B, 1)
    x_proj = x_local_weighted.sum(dim=1) / weight_sum  # (B, D)

    # Blended drift
    b_blend = (all_b.unsqueeze(0) * weights.unsqueeze(-1)).sum(dim=1) / weight_sum

    # Blended diffusion: blend Lambda (covariance), then re-SVD to get H
    # Following MATLAB: Lambda_blend = Σ w_k Λ_k / Σ w_k, then H = U_d sqrt(S_d)
    all_Lambda = torch.stack([c.Lambda for c in charts]).to(device)  # (N, D, D)
    Lambda_blend = (all_Lambda.unsqueeze(0) * weights.unsqueeze(-1).unsqueeze(-1)).sum(dim=1) / weight_sum.unsqueeze(-1)
    # SVD of blended covariance → H = U_d * sqrt(S_d)
    U_blend, S_blend, _ = torch.linalg.svd(Lambda_blend)  # (B, D, D), (B, D)
    S_blend = S_blend.clamp(min=0)
    H_blend = U_blend[:, :, :d] * S_blend[:, :d].unsqueeze(-2).sqrt()  # (B, D, d)

    return x_proj, b_blend, H_blend, Lambda_blend


@torch.no_grad()
def atlas_simulate(x0, charts, n_steps, dt, d=2, t0=0.001, chi_p=1.0, dW=None):
    """Euler-Maruyama simulation in ambient space using ATLAS blending.

    x_{n+1} = x_proj + b * dt + H @ dW

    Args:
        x0: Initial ambient positions, shape (B, D).
        charts: List of ATLASChart.
        n_steps: Number of time steps.
        dt: Time step.
        d: Intrinsic dimension.
        dW: Optional pre-generated noise, shape (B, n_steps, d).

    Returns:
        x_traj: Ambient trajectory, shape (B, n_steps+1, D).
    """
    B, D = x0.shape
    device = x0.device

    if dW is None:
        dW = torch.randn(B, n_steps, d, device=device)
    dW_scaled = dW * (dt ** 0.5)

    x_traj = torch.zeros(B, n_steps + 1, D, device=device)
    x_traj[:, 0] = x0
    x = x0.clone()

    for t in range(n_steps):
        x_proj, b, H, _ = atlas_weighted_drift_diffusion(x, charts, d, t0, chi_p)
        noise = (H @ dW_scaled[:, t].unsqueeze(-1)).squeeze(-1)  # (B, D)
        x = x_proj + b * dt + noise
        x_traj[:, t + 1] = x

    return x_traj


# ── Benchmark ────────────────────────────────────────────────────────────

def run_atlas_benchmark(surface_name, D, seed, n_train, sde, train_data,
                        mode="oracle", n_bursts=500, burst_dt=0.001,
                        burst_steps=50, gt_results=None):
    """Run ATLAS on one (surface, seed) and compute MFPT."""
    torch.manual_seed(seed)
    np.random.seed(seed)

    x = train_data.samples.to(DEVICE)
    v = train_data.mu.to(DEVICE)
    Lambda = train_data.cov.to(DEVICE)
    uv = train_data.local_samples.to(DEVICE)

    # Build atlas
    if mode == "oracle":
        charts = build_atlas_from_data(x, v, Lambda, d=2, device=DEVICE)
    else:
        charts = build_atlas_from_bursts(
            sde, uv, D, d=2, n_bursts=n_bursts,
            burst_dt=burst_dt, burst_steps=burst_steps, device=DEVICE,
        )

    # Coefficient errors (evaluate at held-out points)
    # For ATLAS, compute ambient drift/covariance at eval points via blending
    torch.manual_seed(seed + 5000)
    eval_uv = (torch.rand(N_EVAL, 2, device=DEVICE) * 2 - 1) * TRAIN_BOUND
    surface = FourierAugmentedSurface(surface_name, D)
    eval_data = sample_from_highd_manifold(
        surface, mb_local_drift_fn, mb_local_diffusion_fn,
        [(-TRAIN_BOUND, TRAIN_BOUND)] * 2,
        n_samples=N_EVAL, seed=seed + 5000, device=DEVICE,
        uv_override=eval_uv,
    )
    x_eval = eval_data.samples.to(DEVICE)
    v_eval = eval_data.mu.to(DEVICE)

    # ATLAS drift at eval points — use same metric as AE pipeline
    # E_mu = median of ||b_atlas - b_true||^2 (squared L2, matching compute_coefficient_errors)
    _, b_atlas, _, _ = atlas_weighted_drift_diffusion(x_eval, charts, d=2)
    e_mu = (b_atlas - v_eval).pow(2).sum(-1).median().item()
    print(f"    ATLAS E_mu={e_mu:.4f}")

    # Simulation
    init_ambient = gt_results['init_ambient']
    n_traj = init_ambient.shape[0]

    torch.manual_seed(seed + 5678)
    dW = torch.randn(n_traj, LONG_N_STEPS, 2, device=DEVICE)
    x_traj = atlas_simulate(
        init_ambient, charts, LONG_N_STEPS, MB_DT, d=2, dW=dW,
    )

    # Extract (u,v) from ambient trajectory (first 2 components)
    lr_uv = x_traj[:, :, :2]

    # Censor
    ever_out = (lr_uv.abs() > CENSOR_BOUND).any(dim=-1).any(dim=-1)
    exit_frac = ever_out.float().mean().item()

    # Well assignment and MFPT
    lr_assign = assign_wells_coreset_2d(lr_uv, WELLS_UV.to(DEVICE))
    lr_assign[ever_out] = -1
    lr_mfpt = compute_pairwise_mfpt(lr_assign, dt=MB_DT)

    gt_mfpt = gt_results['gt_pairwise_mfpt']
    errs = []
    for i in range(3):
        for j in range(3):
            if i == j:
                continue
            g, l = gt_mfpt[i, j].item(), lr_mfpt[i, j].item()
            if np.isfinite(g) and g > 0 and np.isfinite(l):
                errs.append(abs(l - g) / g)
    mfpt_err = np.mean(errs) if errs else float('nan')

    print(f"    ATLAS MFPT err={mfpt_err:.1%}  exit={exit_frac:.1%}")

    return {
        'surface': surface_name,
        'method': f'ATLAS_{mode}',
        'seed': seed,
        'E_mu': e_mu,
        'mfpt_err': mfpt_err,
        'exit_frac': exit_frac,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-seeds", type=int, default=5)
    parser.add_argument("--D", type=int, default=11)
    parser.add_argument("--N", type=int, default=200)
    parser.add_argument("--mode", type=str, default="oracle",
                        choices=["oracle", "burst"])
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--surfaces", type=str, nargs="+",
                        default=["paraboloid"])
    parser.add_argument("--n-traj", type=int, default=None)
    args = parser.parse_args()

    if args.n_traj is not None:
        import experiments.data_driven_sde as _dds
        _dds.N_TRAJ = args.n_traj
        global N_TRAJ
        N_TRAJ = args.n_traj

    if args.output is None:
        args.output = f"atlas_benchmark_d{args.D}.csv"

    seeds = [42 + i * 1000 for i in range(args.n_seeds)]
    print(f"ATLAS benchmark: D={args.D}, N={args.N}, mode={args.mode}")
    print(f"Surfaces: {args.surfaces}, Seeds: {seeds}")

    t0 = time.time()
    all_rows = []

    for surface_name in args.surfaces:
        for seed in seeds:
            print(f"\n{'='*60}")
            print(f"  {surface_name} | seed={seed}")
            print(f"{'='*60}")

            torch.manual_seed(seed)
            np.random.seed(seed)

            surface = FourierAugmentedSurface(surface_name, args.D)
            uv_is = importance_sample_mb(args.N, seed=seed, device=DEVICE)
            train_data = sample_from_highd_manifold(
                surface, mb_local_drift_fn, mb_local_diffusion_fn,
                [(-TRAIN_BOUND, TRAIN_BOUND)] * 2,
                n_samples=args.N, seed=seed, device=DEVICE,
                uv_override=uv_is,
            )
            sde = create_highd_lambdified_sde(
                surface, mb_local_drift_fn, mb_local_diffusion_fn,
            )

            # GT simulation
            torch.manual_seed(seed + 999)
            n_traj = N_TRAJ
            init_local = torch.randn(n_traj, 2, device=DEVICE) * 0.1
            init_local[:, 0] += WELLS_UV[0, 0]
            init_local[:, 1] += WELLS_UV[0, 1]
            init_local.clamp_(-TRAIN_BOUND, TRAIN_BOUND)
            init_ambient = sde.chart(init_local).to(DEVICE)

            torch.manual_seed(seed + 1234)
            dW_gt = torch.randn(n_traj, LONG_N_STEPS, 2, device=DEVICE)
            _, _, gt_local = simulate_ground_truth(
                init_local, sde, LONG_N_STEPS, MB_DT, dW_gt,
                boundary=None,
            )
            gt_ever_out = (gt_local.abs() > CENSOR_BOUND).any(dim=-1).any(dim=-1)
            gt_assign = assign_wells_coreset_2d(gt_local, WELLS_UV.to(DEVICE))
            gt_assign[gt_ever_out] = -1
            gt_mfpt = compute_pairwise_mfpt(gt_assign, dt=MB_DT)
            del dW_gt

            gt_results = {
                'gt_pairwise_mfpt': gt_mfpt,
                'init_ambient': init_ambient,
            }

            row = run_atlas_benchmark(
                surface_name, args.D, seed, args.N, sde, train_data,
                mode=args.mode, gt_results=gt_results,
            )
            all_rows.append(row)
            pd.DataFrame(all_rows).to_csv(args.output, index=False)

    elapsed = time.time() - t0
    df = pd.DataFrame(all_rows)
    df.to_csv(args.output, index=False)
    print(f"\nTotal time: {elapsed / 60:.1f} min")
    print(f"Saved {len(df)} rows to {args.output}")

    # Summary
    print(f"\n{'='*60}")
    for s in args.surfaces:
        sub = df[df['surface'] == s]
        print(f"\n  {s}: E_mu={sub['E_mu'].mean():.4f}  "
              f"MFPT_err={sub['mfpt_err'].mean():.1%}±{sub['mfpt_err'].std():.1%}  "
              f"exit={sub['exit_frac'].mean():.1%}")


if __name__ == "__main__":
    main()
