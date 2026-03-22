"""
Unified ablation study: chart + coefficient quality under all conditions.

Covers both dynamics (rotation, MB Langevin), all 4 surfaces, D=11 and D=201,
5 AE conditions + ATLAS baseline. Reports reconstruction, tangent error,
σ_min, E_μ, E_Σ, and Stage 2 drift loss.

This is the single source of truth for the paper's ablation table.

Usage:
    # Smoke test
    python -m experiments.full_ablation --n-seeds 1 --D 11 --dynamics mb \
        --surfaces paraboloid --ae-epochs 200 --sde-epochs 100

    # Full run (submit via SLURM)
    python -m experiments.full_ablation --n-seeds 10 --D 11 --dynamics mb
"""

import argparse
import time
import torch
import numpy as np
import pandas as pd

from src.numeric.losses import LossWeights
from src.numeric.geometry import compute_sigma_min, ambient_quadratic_variation_drift
from src.numeric.performance_stats import compute_losses_per_sample
from src.numeric.sde_nets import DriftNet, DiffusionNet
from src.numeric.sde_training import SDEPipelineTrainer
from src.numeric.highd_manifolds import (
    FourierAugmentedSurface,
    sample_from_highd_manifold,
    create_highd_lambdified_sde,
)
from experiments.highd_N_D_sweep import (
    N_EVAL, compute_coefficient_errors, hidden_dims_for_D,
    local_drift_fn as rot_drift_fn,
    local_diffusion_fn as rot_diffusion_fn,
)
from experiments.mb_dynamics import (
    mb_local_drift_fn, mb_local_diffusion_fn,
    train_ae_highd, importance_sample_mb,
)
from experiments.mb_dynamics import TRAIN_BOUND as MB_TRAIN_BOUND
from experiments.data_driven_sde import DEVICE, BATCH_SIZE, LR_SDE
from experiments.data_driven_sde import TRAIN_BOUND as ROT_TRAIN_BOUND
from experiments.mfpt_full_ablation import LAMBDA_SMOOTH
from experiments.atlas_benchmark import (
    build_atlas_from_data, atlas_weighted_drift_diffusion,
)

# ── Conditions ────────────────────────────────────────────────────────────

LAMBDA_C = 0.01

AE_CONDITIONS = {
    "baseline": LossWeights(),
    "T":        LossWeights(tangent_bundle=1.0),
    "F":        LossWeights(diffeo=1.0),
    "C":        LossWeights(contractive=LAMBDA_C),
    "T+F":      LossWeights(tangent_bundle=1.0, diffeo=1.0),
}

SURFACES = ["paraboloid", "hyperbolic_paraboloid", "quartic_dome", "sinusoidal"]

# ── Dynamics-specific config ──────────────────────────────────────────────

DYNAMICS_CONFIG = {
    "rotation": {
        "local_drift_fn": rot_drift_fn,
        "local_diffusion_fn": rot_diffusion_fn,
        "N_train": 50,
        "ae_epochs": 500,
        "sde_epochs": 300,
        "drift_hidden": None,  # default [64, 64]
        "sampling": "delta_net",
        "train_bound": ROT_TRAIN_BOUND,
    },
    "mb": {
        "local_drift_fn": mb_local_drift_fn,
        "local_diffusion_fn": mb_local_diffusion_fn,
        "N_train": 200,
        "ae_epochs": 4000,
        "sde_epochs": 3000,
        "drift_hidden": [256, 256, 256],
        "sampling": "delta_net",
        "train_bound": MB_TRAIN_BOUND,
    },
}

def run_one(surface_name, D, seed, cond_label, lw, dynamics, dyn_cfg,
            train_data, sde, ae_epochs, sde_epochs):
    """Run one (surface, D, seed, condition, dynamics) and return metrics."""
    print(f"    {cond_label}", end="", flush=True)
    torch.manual_seed(seed)
    np.random.seed(seed)

    surface = FourierAugmentedSurface(surface_name, D)
    N = train_data.samples.shape[0]

    # Stage 1: Train AE
    ae, recon = train_ae_highd(surface, D, seed, N, ae_epochs, lw, train_data)

    # Chart quality: tangent error, σ_min
    test_data = sample_from_highd_manifold(
        surface, dyn_cfg["local_drift_fn"], dyn_cfg["local_diffusion_fn"],
        [(-dyn_cfg["train_bound"], dyn_cfg["train_bound"])] * 2,
        n_samples=500, seed=seed + 10000, device=DEVICE,
    )
    losses_df = compute_losses_per_sample(ae, test_data)
    tangent_err = losses_df["tangent"].mean()
    sigma_min_vals = compute_sigma_min(ae, test_data.samples.to(DEVICE))

    # Coefficient errors E_mu, E_Sigma
    torch.manual_seed(seed + 5000)
    eval_uv = (torch.rand(N_EVAL, 2, device=DEVICE) * 2 - 1) * dyn_cfg["train_bound"]
    e_mu, e_sigma = compute_coefficient_errors(ae, sde, surface, eval_uv, DEVICE)

    row = {
        "surface": surface_name,
        "D": D,
        "dynamics": dynamics,
        "condition": cond_label,
        "seed": seed,
        "recon": recon,
        "tangent": tangent_err,
        "E_mu": e_mu,
        "E_Sigma": e_sigma,
        "sigma_min_p5": sigma_min_vals.quantile(0.05).item(),
        "sigma_min_median": sigma_min_vals.median().item(),
        "stage2_loss": float('nan'),
        "stage3_loss": float('nan'),
        "E_drift": float('nan'),
        "E_b": float('nan'),
        "E_Lambda": float('nan'),
    }

    # Stage 2 + 3 (always run, even for catastrophic conditions)
    x = train_data.samples.to(DEVICE)
    v = train_data.mu.to(DEVICE)
    Lambda = train_data.cov.to(DEVICE)
    d = 2
    bs = min(N, BATCH_SIZE)

    tmp = SDEPipelineTrainer(
        ae, DriftNet(d).to(DEVICE), DiffusionNet(d).to(DEVICE), device=DEVICE,
    )
    z_pre, dphi_pre, _ = tmp.precompute_decoder_derivatives(x)
    b_z_target, g = tmp.precompute_enc_pull_target(x, v, Lambda, dphi_pre)

    torch.manual_seed(seed + 100)
    drift_net = DriftNet(d, hidden_dims=dyn_cfg["drift_hidden"]).to(DEVICE)
    diff_net = DiffusionNet(d).to(DEVICE)
    pipe = SDEPipelineTrainer(ae, drift_net, diff_net, device=DEVICE)

    losses = pipe.train_stage2_regression(
        z_pre, b_z_target, g,
        epochs=sde_epochs, lr=LR_SDE, batch_size=bs,
        print_interval=0,  # silent
        lambda_smooth=LAMBDA_SMOOTH, aug_sigma=0.1,
    )
    row["stage2_loss"] = losses[-1] if losses else float('nan')

    # E_drift: drift-net fit error at eval points (metric-weighted)
    with torch.no_grad():
        x_eval = sde.chart(eval_uv)
        z_eval = ae.encoder(x_eval)
    dphi_eval = torch.func.vmap(torch.func.jacrev(ae.decoder))(z_eval).detach()
    g_eval = torch.bmm(dphi_eval.transpose(-1, -2), dphi_eval)
    tmp_eval = SDEPipelineTrainer(ae, drift_net, diff_net, device=DEVICE)
    z_e, dphi_e, _ = tmp_eval.precompute_decoder_derivatives(x_eval)
    b_eval = sde.ambient_drift(eval_uv.detach())
    Lambda_eval = sde.ambient_covariance(eval_uv.detach())
    mu_enc_eval, g_enc = tmp_eval.precompute_enc_pull_target(
        x_eval, b_eval, Lambda_eval, dphi_e,
    )
    with torch.no_grad():
        mu_hat = drift_net(z_eval)
        residual = mu_hat - mu_enc_eval
        # ||r||_g^2 = r^T g r = ||Dφ r||^2
        e_drift_per = torch.bmm(
            dphi_eval, residual.unsqueeze(-1)
        ).squeeze(-1).pow(2).sum(-1)
        valid_d = torch.isfinite(e_drift_per)
        row["E_drift"] = (
            e_drift_per[valid_d].median().item()
            if valid_d.sum() >= 2 else float('nan')
        )
    del dphi_eval, g_eval, z_e, dphi_e, mu_enc_eval, g_enc, mu_hat
    torch.cuda.empty_cache()

    # Stage 3: diffusion
    torch.manual_seed(seed + 200)
    diff_net2 = DiffusionNet(d).to(DEVICE)
    pipe2 = SDEPipelineTrainer(ae, drift_net, diff_net2, device=DEVICE)
    losses3 = pipe2.train_stage3_precomputed(
        z_pre, dphi_pre, Lambda,
        epochs=sde_epochs, lr=LR_SDE, batch_size=bs,
        print_interval=0,
    )
    row["stage3_loss"] = losses3[-1] if losses3 else float('nan')

    # True end-to-end coefficient errors at eval points
    with torch.no_grad():
        x_ev = sde.chart(eval_uv)
        z_ev = ae.encoder(x_ev).detach()
    z_ev.requires_grad_(True)
    dphi_ev = torch.func.vmap(torch.func.jacrev(ae.decoder))(z_ev).detach()

    # Decoder Hessian (batched to avoid OOM)
    D_dim = dphi_ev.shape[1]
    hess_bs2 = 32 if D_dim > 100 else len(z_ev)
    d2phi_chunks2 = []
    for i in range(0, len(z_ev), hess_bs2):
        d2phi_chunks2.append(
            torch.func.vmap(
                torch.func.jacrev(torch.func.jacrev(ae.decoder))
            )(z_ev[i:i+hess_bs2]).detach()
        )
        torch.cuda.empty_cache()
    d2phi_ev = torch.cat(d2phi_chunks2)

    b_true = sde.ambient_drift(eval_uv).detach()
    Lambda_true = sde.ambient_covariance(eval_uv).detach()

    with torch.no_grad():
        mu_hat_ev = drift_net(z_ev)
        sigma_hat_ev = diff_net2(z_ev)
        Sigma_hat = torch.bmm(sigma_hat_ev, sigma_hat_ev.transpose(-1, -2))

        # E_b: ||Dφ μ̂ + ½q(σ̂σ̂ᵀ) − b||²
        q_hat = ambient_quadratic_variation_drift(Sigma_hat, d2phi_ev)
        b_recon = torch.bmm(dphi_ev, mu_hat_ev.unsqueeze(-1)).squeeze(-1) + 0.5 * q_hat
        e_b_per = (b_recon - b_true).pow(2).sum(-1)

        # E_Lambda: ||Dφ σ̂σ̂ᵀ Dφᵀ − Λ||²_F
        Lambda_recon = torch.bmm(dphi_ev, torch.bmm(Sigma_hat, dphi_ev.transpose(-1, -2)))
        e_lam_per = ((Lambda_recon - Lambda_true) ** 2).sum(dim=(-1, -2))

        valid_c = torch.isfinite(e_b_per) & torch.isfinite(e_lam_per)
        row["E_b"] = e_b_per[valid_c].median().item() if valid_c.sum() >= 2 else float('nan')
        row["E_Lambda"] = e_lam_per[valid_c].median().item() if valid_c.sum() >= 2 else float('nan')

    del dphi_ev, d2phi_ev, mu_hat_ev, sigma_hat_ev, Sigma_hat, q_hat
    torch.cuda.empty_cache()

    print(f"  recon={recon:.4f}  E={e_mu:.4f}  E_b={row['E_b']:.6f}  E_Lam={row['E_Lambda']:.6f}")
    return row


def run_atlas(surface_name, D, seed, dynamics, dyn_cfg, train_data):
    """Run ATLAS baseline (no AE) and return metrics."""
    print(f"    ATLAS", end="", flush=True)

    x = train_data.samples.to(DEVICE)
    v = train_data.mu.to(DEVICE)
    Lambda = train_data.cov.to(DEVICE)

    charts = build_atlas_from_data(x, v, Lambda, d=2, device=DEVICE)

    # E_mu at held-out points
    surface = FourierAugmentedSurface(surface_name, D)
    torch.manual_seed(seed + 5000)
    eval_uv = (torch.rand(N_EVAL, 2, device=DEVICE) * 2 - 1) * dyn_cfg["train_bound"]
    eval_data = sample_from_highd_manifold(
        surface, dyn_cfg["local_drift_fn"], dyn_cfg["local_diffusion_fn"],
        [(-dyn_cfg["train_bound"], dyn_cfg["train_bound"])] * 2,
        n_samples=N_EVAL, seed=seed + 5000, device=DEVICE,
        uv_override=eval_uv,
    )
    x_eval = eval_data.samples.to(DEVICE)
    v_eval = eval_data.mu.to(DEVICE)

    x_proj, b_atlas, _, Lambda_atlas = atlas_weighted_drift_diffusion(x_eval, charts, d=2)
    e_mu = (b_atlas - v_eval).pow(2).sum(-1).median().item()
    # Per-dim recon to match AE's recon_per_dim
    recon = (x_proj - x_eval).pow(2).sum(-1).mean().item() / D

    # Tangent error: compare ATLAS tangent projector (from SVD of Lambda_atlas)
    # against true tangent projector P = U_d U_d^T from eval_data
    Lambda_eval = eval_data.cov.to(DEVICE)
    P_true = eval_data.p.to(DEVICE)  # true tangent projector (B, D, D)
    # ATLAS projector: SVD of blended Lambda → top-d left singular vectors
    U_atlas, _, _ = torch.linalg.svd(Lambda_atlas)
    P_atlas = U_atlas[:, :, :2] @ U_atlas[:, :, :2].transpose(-1, -2)
    tangent_err = ((P_atlas - P_true) ** 2).sum((-1, -2)).mean().item() * 0.5

    # E_Sigma: ||P_atlas Lambda P_atlas - Lambda_eval||_F^2
    P_Lam_P = P_atlas @ Lambda_eval @ P_atlas
    e_sigma = ((P_Lam_P - Lambda_eval) ** 2).sum((-1, -2)).median().item()

    # Stage 2 equivalent: drift approximation error at training points
    # (analogous to S2 training loss — how well does blended drift match true drift)
    _, b_atlas_train, _, Lambda_atlas_train = atlas_weighted_drift_diffusion(x, charts, d=2)
    stage2_equiv = (b_atlas_train - v).pow(2).sum(-1).mean().item()

    # Stage 3 equivalent: diffusion approximation error at training points
    stage3_equiv = ((Lambda_atlas_train - Lambda) ** 2).sum((-1, -2)).mean().item()

    # E_b and E_Lambda for ATLAS: direct comparison of blended coefficients
    # ATLAS produces ambient drift and covariance directly (no Itô correction needed)
    # e_mu (line above) already computes ||b_atlas - b_true||^2, so E_b = e_mu
    e_b_atlas = e_mu
    e_lam_atlas = ((Lambda_atlas - Lambda_eval) ** 2).sum((-1, -2)).median().item()

    print(f"  recon={recon:.4f}  tangent={tangent_err:.4f}  E_mu={e_mu:.4f}  E_b={e_b_atlas:.4f}  E_Lam={e_lam_atlas:.4f}")

    return {
        "surface": surface_name,
        "D": D,
        "dynamics": dynamics,
        "condition": "ATLAS",
        "seed": seed,
        "recon": recon,
        "tangent": tangent_err,
        "E_mu": e_mu,
        "E_Sigma": e_sigma,
        "sigma_min_p5": float('nan'),
        "sigma_min_median": float('nan'),
        "stage2_loss": stage2_equiv,
        "stage3_loss": stage3_equiv,
        "E_drift": float('nan'),  # N/A for ATLAS (no drift_net)
        "E_b": e_b_atlas,
        "E_Lambda": e_lam_atlas,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-seeds", type=int, default=10)
    parser.add_argument("--base-seed", type=int, default=42)
    parser.add_argument("--D", type=int, default=11)
    parser.add_argument("--dynamics", type=str, default="mb",
                        choices=["rotation", "mb"])
    parser.add_argument("--surfaces", type=str, nargs="+", default=None)
    parser.add_argument("--ae-epochs", type=int, default=None,
                        help="Override AE epochs (default: per dynamics)")
    parser.add_argument("--sde-epochs", type=int, default=None,
                        help="Override SDE epochs (default: per dynamics)")
    parser.add_argument("--output", type=str, default=None)
    args = parser.parse_args()

    dyn_cfg = DYNAMICS_CONFIG[args.dynamics].copy()
    if args.ae_epochs is not None:
        dyn_cfg["ae_epochs"] = args.ae_epochs
    if args.sde_epochs is not None:
        dyn_cfg["sde_epochs"] = args.sde_epochs

    surfaces = args.surfaces if args.surfaces else SURFACES
    seeds = [args.base_seed + i * 1000 for i in range(args.n_seeds)]

    if args.output is None:
        args.output = f"full_ablation_{args.dynamics}_d{args.D}.csv"

    print(f"Full ablation: {args.dynamics} dynamics, D={args.D}")
    print(f"Surfaces: {surfaces}")
    print(f"Conditions: {list(AE_CONDITIONS.keys())} + ATLAS")
    print(f"Seeds ({len(seeds)}): {seeds}")
    print(f"AE epochs: {dyn_cfg['ae_epochs']}, SDE epochs: {dyn_cfg['sde_epochs']}")
    print(f"N_train: {dyn_cfg['N_train']}")

    t0 = time.time()
    all_rows = []

    for surface_name in surfaces:
        for seed in seeds:
            print(f"\n{'='*60}")
            print(f"  {surface_name} | seed={seed} | {args.dynamics} | D={args.D}")
            print(f"{'='*60}")

            torch.manual_seed(seed)
            np.random.seed(seed)

            surface = FourierAugmentedSurface(surface_name, args.D)
            N = dyn_cfg["N_train"]

            # Generate training data
            bds = [(-dyn_cfg["train_bound"], dyn_cfg["train_bound"])] * 2
            if dyn_cfg["sampling"] == "delta_net":
                from src.numeric.highd_manifolds import delta_net_subsample
                uv_dn = delta_net_subsample(
                    surface, bds, n_landmarks=N,
                    n_candidates=max(10000, N * 100),
                    seed=seed, device=DEVICE,
                )
                train_data = sample_from_highd_manifold(
                    surface, dyn_cfg["local_drift_fn"], dyn_cfg["local_diffusion_fn"],
                    bds, n_samples=N, seed=seed, device=DEVICE,
                    uv_override=uv_dn,
                )
            elif dyn_cfg["sampling"] == "importance":
                uv_is = importance_sample_mb(N, seed=seed, device=DEVICE)
                train_data = sample_from_highd_manifold(
                    surface, dyn_cfg["local_drift_fn"], dyn_cfg["local_diffusion_fn"],
                    bds, n_samples=N, seed=seed, device=DEVICE,
                    uv_override=uv_is,
                )
            else:
                train_data = sample_from_highd_manifold(
                    surface, dyn_cfg["local_drift_fn"], dyn_cfg["local_diffusion_fn"],
                    bds, n_samples=N, seed=seed, device=DEVICE,
                )

            sde = create_highd_lambdified_sde(
                surface, dyn_cfg["local_drift_fn"], dyn_cfg["local_diffusion_fn"],
            )

            # ATLAS
            row = run_atlas(surface_name, args.D, seed, args.dynamics,
                           dyn_cfg, train_data)
            all_rows.append(row)

            # AE conditions
            for cond_label, lw in AE_CONDITIONS.items():
                row = run_one(
                    surface_name, args.D, seed, cond_label, lw,
                    args.dynamics, dyn_cfg, train_data, sde,
                    dyn_cfg["ae_epochs"], dyn_cfg["sde_epochs"],
                )
                all_rows.append(row)

            # Save incrementally
            pd.DataFrame(all_rows).to_csv(args.output, index=False)

    elapsed = time.time() - t0
    df = pd.DataFrame(all_rows)
    df.to_csv(args.output, index=False)
    print(f"\n\nTotal time: {elapsed / 60:.1f} min")
    print(f"Saved {len(df)} rows to {args.output}")

    # ── Summary ──
    print(f"\n{'='*80}")
    print(f"ABLATION SUMMARY ({args.dynamics}, D={args.D})")
    print(f"{'='*80}")

    conds = ["ATLAS"] + list(AE_CONDITIONS.keys())
    for surface_name in surfaces:
        sub = df[df["surface"] == surface_name]
        print(f"\n  {surface_name}:")
        print(f"    {'cond':>10s}  {'recon':>12s}  {'tangent':>12s}  "
              f"{'E_mu':>12s}  {'E_Sigma':>12s}  {'S2_loss':>12s}")
        for cond in conds:
            cs = sub[sub["condition"] == cond]
            if len(cs) == 0:
                continue
            r = cs["recon"].values
            t = cs["tangent"].values
            em = cs["E_mu"].values
            es = cs["E_Sigma"].values
            s2 = cs["stage2_loss"].values
            print(f"    {cond:>10s}  "
                  f"{np.nanmean(r):>5.4f}±{np.nanstd(r):>.4f}  "
                  f"{np.nanmean(t):>5.4f}±{np.nanstd(t):>.4f}  "
                  f"{np.nanmean(em):>5.4f}±{np.nanstd(em):>.4f}  "
                  f"{np.nanmean(es):>5.4f}±{np.nanstd(es):>.4f}  "
                  f"{np.nanmean(s2):>5.6f}")


if __name__ == "__main__":
    main()
