"""
Oracle drift experiment: replace learned drift_net with exact encoder-pullback
drift on a dense grid, to isolate the Stage 2 bottleneck.

Following GPT Pro's recommendation: "replace the drift MLP with an oracle-quality
interpolant of the exact pullback target on a dense grid. If T+F then clearly wins,
you have nailed the bottleneck."

Conditions per (surface, seed):
  - baseline + learned drift   (= current pipeline)
  - baseline + oracle drift
  - T+F + learned drift        (= current pipeline)
  - T+F + oracle drift

Also computes target roughness diagnostics: R1, R2, kappa_g.
"""

import argparse
import time
import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd

from src.numeric.losses import LossWeights
from src.numeric.sde_nets import DriftNet, DiffusionNet
from src.numeric.sde_training import SDEPipelineTrainer
from src.numeric.highd_manifolds import (
    FourierAugmentedSurface,
    sample_from_highd_manifold,
    create_highd_lambdified_sde,
)

from experiments.data_driven_sde import (
    simulate_ground_truth, BATCH_SIZE, LR_SDE, DEVICE, N_TRAJ,
)
from experiments.highd_N_D_sweep import compute_coefficient_errors, N_EVAL
from experiments.mfpt_full_ablation import train_pipeline, LAMBDA_SMOOTH

from experiments.mb_dynamics import (
    mb_local_drift_fn, mb_local_diffusion_fn,
    train_ae_highd,
    WELLS_UV, WELL_NAMES, TRAIN_BOUND,
    assign_wells_coreset_2d, compute_pairwise_mfpt,
    compute_implied_timescales, compute_stationary_probabilities,
    compute_exit_fraction, importance_sample_mb,
    CORE_RADIUS, CORE_DWELL,
)
from experiments.mb_mfpt import (
    MB_DRIFT_HIDDEN, MB_SDE_EPOCHS, MB_DT, LONG_T, LONG_N_STEPS,
    LAG_TIMES, CENSOR_BOUND,
)


# ── Oracle drift via GPU grid interpolation ──────────────────────────────

def build_oracle_fields(ae, surface, grid_res=50, pad=0.15, device=DEVICE,
                        train_data=None):
    """Compute exact encoder-pullback drift + diffusion on a dense latent grid.

    Grid bounds are determined from the actual latent range of training data
    (+ wells + saddle) rather than TRAIN_BOUND, since the AE can map the
    physical domain to a different latent range.

    Returns:
        bz_grid: (1, 2, grid_res, grid_res) tensor for grid_sample
        sigma_grid: (1, 4, grid_res, grid_res) tensor for grid_sample
        z_min, z_max: grid bounds for normalization
        diagnostics: dict with R1, R2, kappa_g
    """
    # Determine latent grid bounds from actual AE mapping
    with torch.no_grad():
        # Encode training data
        if train_data is not None:
            z_train = ae.encoder(train_data.samples.to(device))
        else:
            # Fallback: sample a grid in (u,v) and encode
            uv_probe = (torch.rand(500, 2, device=device) * 2 - 1) * TRAIN_BOUND
            x_probe = surface.chart(uv_probe).to(device)
            z_train = ae.encoder(x_probe)
        # Also encode wells and saddle
        from experiments.mb_dynamics import SADDLE_UV
        key_uv = torch.cat([WELLS_UV, SADDLE_UV.unsqueeze(0)], dim=0).to(device)
        x_key = surface.chart(key_uv).to(device)
        z_key = ae.encoder(x_key)
        z_all = torch.cat([z_train, z_key], dim=0)
        z_lo = z_all.min(dim=0).values
        z_hi = z_all.max(dim=0).values
    # Pad by fraction of range
    z_range = z_hi - z_lo
    z_lo = z_lo - pad * z_range
    z_hi = z_hi + pad * z_range

    lin0 = torch.linspace(z_lo[0].item(), z_hi[0].item(), grid_res, device=device)
    lin1 = torch.linspace(z_lo[1].item(), z_hi[1].item(), grid_res, device=device)
    z1, z2 = torch.meshgrid(lin0, lin1, indexing='ij')
    z_grid = torch.stack([z1.flatten(), z2.flatten()], dim=-1)  # (G, 2)
    G = z_grid.shape[0]

    # Decode grid → ambient → extract (u,v) → recompute exact SDE coefficients
    with torch.no_grad():
        x_decoded = ae.decoder(z_grid)
    uv = x_decoded[:, :2].detach()
    # Clamp to training domain (decoded points may extrapolate slightly)
    uv = uv.clamp(-TRAIN_BOUND * 1.2, TRAIN_BOUND * 1.2)

    # Get exact ambient drift and covariance at these (u,v) points
    D = x_decoded.shape[1]
    data = sample_from_highd_manifold(
        surface, mb_local_drift_fn, mb_local_diffusion_fn,
        [(-TRAIN_BOUND, TRAIN_BOUND), (-TRAIN_BOUND, TRAIN_BOUND)],
        n_samples=G, seed=0, device=device,
        uv_override=uv,
    )
    x_exact = data.samples.to(device)
    v_exact = data.mu.to(device)
    Lambda_exact = data.cov.to(device)

    # Compute encoder-pullback drift: b_z = Dπ·v + ½Λ:∇²π
    tmp = SDEPipelineTrainer(
        ae, DriftNet(2).to(device), DiffusionNet(2).to(device), device=device,
    )
    z_pre, dphi_pre, _ = tmp.precompute_decoder_derivatives(x_exact)
    b_z_oracle, g = tmp.precompute_enc_pull_target(
        x_exact, v_exact, Lambda_exact, dphi_pre, batch_size=32,
    )

    # Compute encoder Jacobian for oracle diffusion
    dpi = tmp.precompute_encoder_jacobian(x_exact, batch_size=128)  # (G, 2, D)
    # a_z = Dπ Λ Dπ^T
    a_z = dpi @ Lambda_exact @ dpi.transpose(-1, -2)  # (G, 2, 2)
    # Cholesky → σ_z
    a_z = a_z + 1e-8 * torch.eye(2, device=device)  # regularize
    sigma_z_oracle = torch.linalg.cholesky(a_z)  # (G, 2, 2)

    # Reshape for grid_sample: (1, C, H, W) with H=grid_res, W=grid_res
    bz_img = b_z_oracle.reshape(grid_res, grid_res, 2).permute(2, 0, 1).unsqueeze(0)
    sigma_flat = sigma_z_oracle.reshape(G, 4)  # flatten 2x2 matrix
    sigma_img = sigma_flat.reshape(grid_res, grid_res, 4).permute(2, 0, 1).unsqueeze(0)

    # Per-dimension bounds as tensors for query_grid
    z_lo_t = torch.tensor([z_lo[0].item(), z_lo[1].item()], device=device)
    z_hi_t = torch.tensor([z_hi[0].item(), z_hi[1].item()], device=device)

    # ── Target roughness diagnostics ──
    # R1 = E[||∇_z b_target||_F²] via finite differences
    bz_2d = b_z_oracle.reshape(grid_res, grid_res, 2)
    dz0 = lin0[1] - lin0[0]
    dz1 = lin1[1] - lin1[0]
    # Gradient along z1 (dim 0)
    dbz_dz1 = (bz_2d[2:, 1:-1] - bz_2d[:-2, 1:-1]) / (2 * dz0)
    # Gradient along z2 (dim 1)
    dbz_dz2 = (bz_2d[1:-1, 2:] - bz_2d[1:-1, :-2]) / (2 * dz1)
    R1 = (dbz_dz1.pow(2).sum(-1) + dbz_dz2.pow(2).sum(-1)).mean().item()

    # R2 = E[||∇²_z b_target||_F²] (second derivatives)
    d2bz_dz1 = (bz_2d[2:, 1:-1] - 2 * bz_2d[1:-1, 1:-1] + bz_2d[:-2, 1:-1]) / dz0**2
    d2bz_dz2 = (bz_2d[1:-1, 2:] - 2 * bz_2d[1:-1, 1:-1] + bz_2d[1:-1, :-2]) / dz1**2
    R2 = (d2bz_dz1.pow(2).sum(-1) + d2bz_dz2.pow(2).sum(-1)).mean().item()

    # kappa_g = E[cond(g)]
    g_vals = g.reshape(grid_res, grid_res, 2, 2)
    svs = torch.linalg.svdvals(g_vals)  # (grid_res, grid_res, 2)
    kappa = svs[..., 0] / svs[..., 1].clamp(min=1e-10)
    kappa_g = kappa.mean().item()

    diagnostics = {"R1": R1, "R2": R2, "kappa_g": kappa_g}

    return bz_img, sigma_img, z_lo_t, z_hi_t, diagnostics


def query_grid(grid_img, z, z_lo, z_hi):
    """Query a grid_sample field at latent points z.

    Args:
        grid_img: (1, C, H, W) precomputed field
        z: (B, 2) query points
        z_lo, z_hi: per-dimension grid bounds, each shape (2,)

    Returns:
        values: (B, C) interpolated values
    """
    # Normalize z to [-1, 1] for grid_sample (per dimension)
    z_norm = (z - z_lo) / (z_hi - z_lo) * 2 - 1  # (B, 2)
    z_norm = z_norm.clamp(-1, 1)
    # grid_sample expects (N, H_out, W_out, 2) with (x, y) = (W, H) convention
    # Our z[:, 0] is dim 0 (rows/H), z[:, 1] is dim 1 (cols/W)
    # grid_sample convention: grid[..., 0] = x (along W), grid[..., 1] = y (along H)
    grid_query = torch.stack([z_norm[:, 1], z_norm[:, 0]], dim=-1)  # swap to (x, y)
    grid_query = grid_query.unsqueeze(0).unsqueeze(2)  # (1, B, 1, 2)
    result = F.grid_sample(
        grid_img, grid_query, mode='bilinear',
        padding_mode='border', align_corners=True,
    )
    # result: (1, C, B, 1)
    return result.squeeze(0).squeeze(-1).T  # (B, C)


@torch.no_grad()
def simulate_oracle(z0, n_steps, dt, bz_img, sigma_img, z_lo, z_hi,
                    diffusion_net=None, dW=None, reflect=True):
    """Euler-Maruyama with oracle drift (+ optional oracle diffusion).

    If diffusion_net is provided, uses learned diffusion instead of oracle.
    If reflect=True, clamp trajectories to grid bounds (reflecting boundary).
    """
    B, d = z0.shape
    device = z0.device
    if dW is None:
        dW = torch.randn(B, n_steps, d, device=device)
    dW_scaled = dW * (dt ** 0.5)

    z_traj = torch.zeros(B, n_steps + 1, d, device=device)
    z_traj[:, 0] = z0
    z = z0.clone()

    for t in range(n_steps):
        b_z = query_grid(bz_img, z, z_lo, z_hi)  # (B, 2)
        if diffusion_net is not None:
            sigma_z = diffusion_net(z)  # (B, 2, 2)
        else:
            sigma_flat = query_grid(sigma_img, z, z_lo, z_hi)  # (B, 4)
            sigma_z = sigma_flat.reshape(B, 2, 2)
        noise = (sigma_z @ dW_scaled[:, t].unsqueeze(-1)).squeeze(-1)
        z = z + b_z * dt + noise
        if reflect:
            z = torch.max(z, z_lo)
            z = torch.min(z, z_hi)
        z_traj[:, t + 1] = z

    return z_traj


def compute_mfpt_metrics(ae, z_traj, gt_results, n_traj):
    """Decode z_traj, assign wells, compute MFPT metrics."""
    B = z_traj.shape[0]
    d = z_traj.shape[2]
    n_steps = z_traj.shape[1] - 1

    # Decode in chunks to get (u,v)
    chunk_size = 2000
    lr_uv = torch.zeros(B, n_steps + 1, 2, device=z_traj.device)
    with torch.no_grad():
        for start in range(0, n_steps + 1, chunk_size):
            end = min(start + chunk_size, n_steps + 1)
            z_chunk = z_traj[:, start:end].reshape(-1, d)
            x_chunk = ae.decoder(z_chunk)
            lr_uv[:, start:end] = x_chunk[:, :2].reshape(B, end - start, 2)

    # Censor
    ever_out = (lr_uv.abs() > CENSOR_BOUND).any(dim=-1).any(dim=-1)
    exit_frac = ever_out.float().mean().item()

    # Well assignment
    lr_assign = assign_wells_coreset_2d(lr_uv, WELLS_UV.to(z_traj.device))
    lr_assign[ever_out] = -1

    lr_pairwise_mfpt = compute_pairwise_mfpt(lr_assign, dt=MB_DT)
    lr_its = compute_implied_timescales(lr_assign, dt=MB_DT, lag_times=LAG_TIMES)
    lr_stat = compute_stationary_probabilities(lr_assign)

    # Pairwise MFPT error
    gt_pairwise_mfpt = gt_results['gt_pairwise_mfpt']
    pairwise_err_vals = []
    for i in range(3):
        for j in range(3):
            if i == j:
                continue
            gt_val = gt_pairwise_mfpt[i, j].item()
            lr_val = lr_pairwise_mfpt[i, j].item()
            if np.isfinite(gt_val) and gt_val > 0 and np.isfinite(lr_val):
                pairwise_err_vals.append(abs(lr_val - gt_val) / gt_val)
    mfpt_err = np.mean(pairwise_err_vals) if pairwise_err_vals else float('nan')

    # ITS t2 error
    gt_its = gt_results['gt_its']
    its_lag = 0.2
    lag_list = list(gt_its['lag_times'])
    idx = lag_list.index(its_lag) if its_lag in lag_list else -1
    gt_t2 = gt_its['t2'][idx] if idx >= 0 else float('nan')
    lr_t2 = lr_its['t2'][idx] if idx >= 0 else float('nan')
    if np.isfinite(gt_t2) and gt_t2 > 0 and np.isfinite(lr_t2):
        its_err = abs(lr_t2 - gt_t2) / gt_t2
    else:
        its_err = float('nan')

    # Stationary L1
    gt_stat = gt_results['gt_stat']
    stat_l1 = (gt_stat - lr_stat).abs().sum().item()

    return {
        'mfpt_err': mfpt_err,
        'its_t2_err': its_err,
        'stat_l1': stat_l1,
        'exit_frac': exit_frac,
    }


def run_one_condition(surface_name, D, seed, cond_label, lw, epochs,
                      sde_epochs, sde, train_data, x, v, Lambda,
                      n_train, gt_results, grid_res=50):
    """Train AE, compute oracle drift, run both learned and oracle MFPT."""
    print(f"\n  --- {cond_label} ---")
    torch.manual_seed(seed)
    np.random.seed(seed)

    # Stage 1: train AE
    ae, recon = train_ae_highd(
        FourierAugmentedSurface(surface_name, D),
        D, seed, n_train, epochs, lw, train_data,
    )
    print(f"    recon={recon:.6f}")

    # Coefficient errors
    torch.manual_seed(seed + 5000)
    eval_uv = (torch.rand(N_EVAL, 2, device=DEVICE) * 2 - 1) * TRAIN_BOUND
    e_mu, e_sigma = compute_coefficient_errors(
        ae, sde, FourierAugmentedSurface(surface_name, D), eval_uv, DEVICE,
    )
    print(f"    E_mu={e_mu:.4f}  E_Sigma={e_sigma:.4f}")

    # Build oracle drift/diffusion on dense grid
    print(f"    Building oracle drift on {grid_res}×{grid_res} grid...")
    surface = FourierAugmentedSurface(surface_name, D)
    bz_img, sigma_img, z_lo, z_hi, diag = build_oracle_fields(
        ae, surface, grid_res=grid_res, device=DEVICE, train_data=train_data,
    )
    print(f"    R1={diag['R1']:.2f}  R2={diag['R2']:.2f}  κ_g={diag['kappa_g']:.3f}")
    print(f"    Grid bounds: z0=[{z_lo[0]:.3f},{z_hi[0]:.3f}] z1=[{z_lo[1]:.3f},{z_hi[1]:.3f}]")

    # Train learned pipeline (Stage 2 + 3) — normal approach
    pipeline = train_pipeline(ae, x, v, Lambda, seed, n_train, sde_epochs,
                              drift_hidden=MB_DRIFT_HIDDEN)

    # Also train a standalone Stage 3 (diffusion only) for oracle drift condition
    d = 2
    bs = min(n_train, BATCH_SIZE)
    tmp = SDEPipelineTrainer(
        ae, DriftNet(d).to(DEVICE), DiffusionNet(d).to(DEVICE), device=DEVICE,
    )
    z_pre, dphi_pre, _ = tmp.precompute_decoder_derivatives(x)

    torch.manual_seed(seed + 300)
    diff_net_oracle = DiffusionNet(d).to(DEVICE)
    pipe_diff = SDEPipelineTrainer(ae, DriftNet(d).to(DEVICE), diff_net_oracle, device=DEVICE)
    pipe_diff.train_stage3_precomputed(
        z_pre, dphi_pre, Lambda.to(DEVICE),
        epochs=sde_epochs, lr=LR_SDE, batch_size=bs,
        print_interval=max(1, sde_epochs // 3),
    )

    # Simulation setup (shared)
    init_ambient = gt_results['init_ambient']
    with torch.no_grad():
        z0 = ae.encoder(init_ambient)

    n_traj = z0.shape[0]
    _COND_SEED_OFFSET = {"baseline": 0, "T": 1, "F": 2, "C": 3, "T+F": 4}

    results = []

    for drift_mode in ["learned", "oracle_drift+learned_diff", "oracle_drift+oracle_diff"]:
        torch.manual_seed(seed + 5678 + _COND_SEED_OFFSET.get(cond_label, 99))
        dW = torch.randn(n_traj, LONG_N_STEPS, 2, device=DEVICE)

        if drift_mode == "learned":
            z_traj, _ = pipeline.simulate(
                z0, LONG_N_STEPS, MB_DT, dW=dW,
                boundary=None, decode=False,
            )
        elif drift_mode == "oracle_drift+learned_diff":
            diff_net_oracle.eval()
            z_traj = simulate_oracle(
                z0, LONG_N_STEPS, MB_DT,
                bz_img, sigma_img, z_lo, z_hi,
                diffusion_net=diff_net_oracle, dW=dW,
            )
        else:  # oracle_drift+oracle_diff
            z_traj = simulate_oracle(
                z0, LONG_N_STEPS, MB_DT,
                bz_img, sigma_img, z_lo, z_hi,
                diffusion_net=None, dW=dW,
            )

        metrics = compute_mfpt_metrics(ae, z_traj, gt_results, n_traj)
        print(f"    {drift_mode:>35s}: MFPT={metrics['mfpt_err']:.1%}  "
              f"ITS={metrics['its_t2_err']:.1%}  exit={metrics['exit_frac']:.1%}")

        results.append({
            'surface': surface_name,
            'condition': cond_label,
            'drift_mode': drift_mode,
            'seed': seed,
            'E_mu': e_mu,
            'E_Sigma': e_sigma,
            'R1': diag['R1'],
            'R2': diag['R2'],
            'kappa_g': diag['kappa_g'],
            'recon': recon,
            'mfpt_err': metrics['mfpt_err'],
            'its_t2_err': metrics['its_t2_err'],
            'stat_l1': metrics['stat_l1'],
            'exit_frac': metrics['exit_frac'],
        })

    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-seeds", type=int, default=5)
    parser.add_argument("--epochs", type=int, default=4000)
    parser.add_argument("--sde-epochs", type=int, default=MB_SDE_EPOCHS)
    parser.add_argument("--base-seed", type=int, default=42)
    parser.add_argument("--D", type=int, default=11)
    parser.add_argument("--N", type=int, default=200)
    parser.add_argument("--grid-res", type=int, default=50)
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--surfaces", type=str, nargs="+",
                        default=["paraboloid", "hyperbolic_paraboloid"])
    parser.add_argument("--n-traj", type=int, default=None)
    args = parser.parse_args()

    if args.n_traj is not None:
        import experiments.data_driven_sde as _dds
        _dds.N_TRAJ = args.n_traj
        global N_TRAJ
        N_TRAJ = args.n_traj

    if args.output is None:
        args.output = f"mb_oracle_drift_d{args.D}.csv"

    conditions = {
        "baseline": LossWeights(),
        "T+F": LossWeights(tangent_bundle=1.0, diffeo=1.0),
    }
    seeds = [args.base_seed + i * 1000 for i in range(args.n_seeds)]

    print(f"Device: {DEVICE}")
    print(f"D={args.D}, N={args.N}, grid_res={args.grid_res}")
    print(f"Surfaces: {args.surfaces}")
    print(f"Seeds ({len(seeds)}): {seeds}")
    print(f"LONG_T={LONG_T}, LONG_N_STEPS={LONG_N_STEPS}, MB_DT={MB_DT}")

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
                [(-TRAIN_BOUND, TRAIN_BOUND), (-TRAIN_BOUND, TRAIN_BOUND)],
                n_samples=args.N, seed=seed, device=DEVICE,
                uv_override=uv_is,
            )
            x = train_data.samples.to(DEVICE)
            v = train_data.mu.to(DEVICE)
            Lambda = train_data.cov.to(DEVICE)
            sde = create_highd_lambdified_sde(
                surface, mb_local_drift_fn, mb_local_diffusion_fn,
            )

            # GT simulation (shared across conditions)
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
            gt_pairwise_mfpt = compute_pairwise_mfpt(gt_assign, dt=MB_DT)
            gt_its = compute_implied_timescales(gt_assign, dt=MB_DT, lag_times=LAG_TIMES)
            gt_stat = compute_stationary_probabilities(gt_assign)
            del dW_gt

            gt_results = {
                'gt_assign': gt_assign, 'gt_pairwise_mfpt': gt_pairwise_mfpt,
                'gt_its': gt_its, 'gt_stat': gt_stat,
                'init_ambient': init_ambient,
            }

            for cond_label, lw in conditions.items():
                rows = run_one_condition(
                    surface_name, args.D, seed, cond_label, lw,
                    args.epochs, args.sde_epochs,
                    sde, train_data, x, v, Lambda,
                    n_train=args.N, gt_results=gt_results,
                    grid_res=args.grid_res,
                )
                all_rows.extend(rows)

            # Save incrementally
            pd.DataFrame(all_rows).to_csv(args.output, index=False)

    elapsed = time.time() - t0
    print(f"\n\nTotal time: {elapsed / 60:.1f} min")

    df = pd.DataFrame(all_rows)
    df.to_csv(args.output, index=False)
    print(f"Saved {len(df)} rows to {args.output}")

    # ── Summary ──
    print(f"\n{'='*80}")
    print("ORACLE DRIFT SUMMARY")
    print(f"{'='*80}")

    for surface_name in args.surfaces:
        sub = df[df['surface'] == surface_name]
        print(f"\n  {surface_name}:")
        print(f"    {'condition':>10s} {'drift_mode':>35s}  {'mfpt_err':>10s} {'R1':>8s} {'R2':>8s} {'κ_g':>8s}")
        for cond in conditions:
            for dm in ["learned", "oracle_drift+learned_diff", "oracle_drift+oracle_diff"]:
                ss = sub[(sub['condition'] == cond) & (sub['drift_mode'] == dm)]
                if len(ss) == 0:
                    continue
                me = ss['mfpt_err'].values
                r1 = ss['R1'].values
                r2 = ss['R2'].values
                kg = ss['kappa_g'].values
                print(f"    {cond:>10s} {dm:>35s}  "
                      f"{np.nanmean(me):>5.1%}±{np.nanstd(me):>.1%} "
                      f"{np.nanmean(r1):>8.1f} {np.nanmean(r2):>8.1f} "
                      f"{np.nanmean(kg):>8.2f}")


if __name__ == "__main__":
    main()
