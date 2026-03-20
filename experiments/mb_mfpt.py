"""
MB inter-well MFPT + transition rate experiment.

4-5 conditions × 4 surfaces × D × 10 seeds.
Uses MB drift + state-dependent diffusion. Independent noise for GT vs learned.

Metrics per (surface, condition, seed):
  - E_mu, E_Sigma: coefficient errors
  - Pairwise MFPT (core-set based)
  - Implied timescales (ITS)
  - Stationary probabilities
  - W2@1.0 for reference

Output: mb_mfpt_d{D}.csv
"""

import argparse
import time
import torch
import numpy as np
import pandas as pd
from scipy import stats

from src.numeric.losses import LossWeights
from src.numeric.geometry import compute_sigma_min
from src.numeric.sde_nets import DriftNet, DiffusionNet
from src.numeric.sde_training import SDEPipelineTrainer
from src.numeric.highd_manifolds import (
    FourierAugmentedSurface,
    sample_from_highd_manifold,
    create_highd_lambdified_sde,
)

from experiments.data_driven_sde import (
    simulate_ground_truth, compute_w2,
    BATCH_SIZE, LR_SDE, DEVICE,
    N_TRAJ,
)
from experiments.highd_N_D_sweep import (
    hidden_dims_for_D, N_EVAL, compute_coefficient_errors,
)
from experiments.mfpt_full_ablation import train_pipeline

from experiments.mb_dynamics import (
    mb_local_drift_fn, mb_local_diffusion_fn,
    train_ae_highd,
    WELLS_UV, WELL_NAMES, TRAIN_BOUND,
    assign_wells_coreset_2d, compute_transition_matrix,
    compute_implied_timescales, compute_stationary_probabilities,
    compute_pairwise_mfpt, compute_exit_fraction,
    importance_sample_mb,
    CORE_RADIUS, CORE_DWELL,
)


# ── Conditions ────────────────────────────────────────────────────────────

LAMBDA_C = 0.01  # best from MB contractive sweep

ALL_CONDITIONS = {
    "baseline": LossWeights(),
    "T":        LossWeights(tangent_bundle=1.0),
    "F":        LossWeights(diffeo=1.0),
    "C":        LossWeights(contractive=LAMBDA_C),
    "T+F":      LossWeights(tangent_bundle=1.0, diffeo=1.0),
}

# D=201 drops C (per convention)
D201_CONDITIONS = {k: v for k, v in ALL_CONDITIONS.items() if k != "C"}

SURFACES = ["paraboloid", "hyperbolic_paraboloid", "quartic_dome", "sinusoidal"]
N_TRAIN = 200
MB_DRIFT_HIDDEN = [256, 256, 256]
MB_SDE_EPOCHS = 3000

# Simulation parameters
MB_DT = 0.005
LONG_T = 50.0
LONG_N_STEPS = int(LONG_T / MB_DT)  # 10000
LAG_TIMES = [0.05, 0.1, 0.2, 0.5]
CENSOR_BOUND = 1.0  # censor trajectories that leave [-1,1]² in decoded (u,v)


def run_one(surface_name, D, seed, cond_label, lw, epochs, sde_epochs,
            sde, wells_ambient, train_data, x, v, Lambda, n_train=50,
            gt_results=None):
    """Run one (surface, condition, seed) configuration.

    gt_results: precomputed GT metrics (shared across conditions for same surface+seed).
    """
    print(f"\n  --- {cond_label} ---")
    torch.manual_seed(seed)
    np.random.seed(seed)

    ae, recon = train_ae_highd(
        FourierAugmentedSurface(surface_name, D),
        D, seed, n_train, epochs, lw, train_data,
    )
    torch.cuda.empty_cache()  # free training cache before Hessian computation
    print(f"    recon={recon:.6f}")

    # sigma_min
    torch.manual_seed(seed + 7000)
    eval_uv_in = (torch.rand(300, 2, device=DEVICE) * 2 - 1) * TRAIN_BOUND
    x_in = sde.chart(eval_uv_in).to(DEVICE)
    sigma_min_vals = compute_sigma_min(ae, x_in)

    # Coefficient errors
    torch.manual_seed(seed + 5000)
    eval_uv = (torch.rand(N_EVAL, 2, device=DEVICE) * 2 - 1) * TRAIN_BOUND
    e_mu, e_sigma = compute_coefficient_errors(ae, sde, FourierAugmentedSurface(surface_name, D), eval_uv, DEVICE)
    print(f"    E_mu={e_mu:.4f}  E_Sigma={e_sigma:.4f}")

    # SDE pipeline
    pipeline = train_pipeline(ae, x, v, Lambda, seed, n_train, sde_epochs,
                              drift_hidden=MB_DRIFT_HIDDEN)
    torch.cuda.empty_cache()

    # Use precomputed GT results
    init_ambient = gt_results['init_ambient']
    gt_assign = gt_results['gt_assign']
    gt_pairwise_mfpt = gt_results['gt_pairwise_mfpt']
    gt_its = gt_results['gt_its']
    gt_stat = gt_results['gt_stat']
    gt_at_1 = gt_results['gt_at_1']

    # Learned trajectories (no boundary — run free, censor post-hoc)
    _COND_SEED_OFFSET = {"baseline": 0, "T": 1, "F": 2, "C": 3, "T+F": 4}
    torch.manual_seed(seed + 5678 + _COND_SEED_OFFSET.get(cond_label, 99))
    dW_lr = torch.randn(N_TRAJ, LONG_N_STEPS, 2, device=DEVICE)
    with torch.no_grad():
        z0 = ae.encoder(init_ambient)
    z_traj, _ = pipeline.simulate(z0, LONG_N_STEPS, MB_DT, dW=dW_lr,
                                  boundary=None, decode=False)

    # Decode in chunks to get (u,v) coords for well assignment
    B = z_traj.shape[0]
    d = z_traj.shape[2]
    chunk_size = 2000
    lr_uv = torch.zeros(B, LONG_N_STEPS + 1, 2, device=DEVICE)
    with torch.no_grad():
        for start in range(0, LONG_N_STEPS + 1, chunk_size):
            end = min(start + chunk_size, LONG_N_STEPS + 1)
            z_chunk = z_traj[:, start:end].reshape(-1, d)
            x_chunk = ae.decoder(z_chunk)
            lr_uv[:, start:end] = x_chunk[:, :2].reshape(B, end - start, 2)

    # Censor trajectories that leave [-CENSOR_BOUND, CENSOR_BOUND]² in decoded (u,v)
    ever_out = (lr_uv.abs() > CENSOR_BOUND).any(dim=-1).any(dim=-1)  # (B,)
    n_censored = ever_out.sum().item()
    exit_frac = n_censored / B
    # Mask censored trajectories: set their assignments to -1 (ignored by metrics)
    lr_uv_censored = lr_uv.clone()
    lr_uv_censored[ever_out] = float('nan')  # won't match any core

    # Core-set well assignment (censored trajectories get -1 everywhere)
    lr_assign = assign_wells_coreset_2d(lr_uv, WELLS_UV.to(DEVICE))
    lr_assign[ever_out] = -1  # exclude censored trajectories from all metrics

    # Pairwise MFPT (core-set based, only on non-censored trajectories)
    lr_pairwise_mfpt = compute_pairwise_mfpt(lr_assign, dt=MB_DT)

    # Implied timescales
    lr_its = compute_implied_timescales(lr_assign, dt=MB_DT, lag_times=LAG_TIMES)

    # Stationary probabilities
    lr_stat = compute_stationary_probabilities(lr_assign)

    # W2 at T=1.0
    if gt_at_1 is not None:
        with torch.no_grad():
            lr_at_1 = ae.decoder(z_traj[:, int(round(1.0 / MB_DT))])
        alive_all = torch.ones(B, dtype=torch.bool, device=DEVICE)
        w2 = compute_w2(lr_at_1, gt_at_1, alive_all, alive_all)
    else:
        w2 = float('nan')

    # W0 escape time: first confirmed passage from W0 to any other well
    def compute_w0_escape(assign, dt, dwell=CORE_DWELL):
        """Mean first-passage time from W0 to any other well.

        All trajectories start near W0; we use all of them (no core-set
        filter on the initial frame, since the Gaussian initialization
        may place some starts outside the tiny core radius).
        Requires `dwell` consecutive frames in another well to confirm escape.
        """
        B = assign.shape[0]
        escape_times = []
        for b in range(B):
            traj = assign[b]
            consecutive = 0
            candidate = -1
            for t in range(1, len(traj)):
                w = traj[t].item()
                if w > 0:  # in W1 or W2
                    if w == candidate:
                        consecutive += 1
                    else:
                        candidate = w
                        consecutive = 1
                    if consecutive >= dwell:
                        escape_times.append((t - dwell + 1) * dt)
                        break
                else:
                    candidate = -1
                    consecutive = 0
        return np.mean(escape_times) if escape_times else float('nan')

    gt_w0_escape = compute_w0_escape(gt_assign, MB_DT)
    lr_w0_escape = compute_w0_escape(lr_assign, MB_DT)
    w0_escape_err = abs(lr_w0_escape - gt_w0_escape) / gt_w0_escape if gt_w0_escape > 0 else float('nan')

    # Pairwise MFPT error (average over all finite pairs)
    pairwise_err_vals = []
    for i in range(3):
        for j in range(3):
            if i == j:
                continue
            gt_val = gt_pairwise_mfpt[i, j].item()
            lr_val = lr_pairwise_mfpt[i, j].item()
            if np.isfinite(gt_val) and gt_val > 0 and np.isfinite(lr_val):
                pairwise_err_vals.append(abs(lr_val - gt_val) / gt_val)
    pairwise_mfpt_err = np.mean(pairwise_err_vals) if pairwise_err_vals else float('nan')

    # Implied timescale error (fixed lag for fair comparison)
    its_lag = 0.2
    lag_list = list(gt_its['lag_times'])
    its_lag_idx = lag_list.index(its_lag) if its_lag in lag_list else -1
    gt_t2 = gt_its['t2'][its_lag_idx] if its_lag_idx >= 0 else float('nan')
    lr_t2 = lr_its['t2'][its_lag_idx] if its_lag_idx >= 0 else float('nan')
    if np.isfinite(gt_t2) and gt_t2 > 0 and np.isfinite(lr_t2):
        its_t2_err = abs(lr_t2 - gt_t2) / gt_t2
    else:
        its_t2_err = float('nan')

    # Stationary probability error (L1)
    stat_l1_err = (gt_stat - lr_stat).abs().sum().item()

    print(f"    Pairwise MFPT err={pairwise_mfpt_err:.1%}  "
          f"W0 escape err={w0_escape_err:.1%}  "
          f"(GT={gt_w0_escape:.2f}, LR={lr_w0_escape:.2f})")
    print(f"    Exit fraction={exit_frac:.1%}")

    return {
        "surface": surface_name,
        "condition": cond_label,
        "seed": seed,
        "recon_per_dim": recon,
        "E_mu": e_mu,
        "E_Sigma": e_sigma,
        "sigma_min_p5": sigma_min_vals.quantile(0.05).item(),
        "W2@1.0": w2,
        "gt_w0_escape": gt_w0_escape,
        "lr_w0_escape": lr_w0_escape,
        "w0_escape_err": w0_escape_err,
        "pairwise_mfpt_err": pairwise_mfpt_err,
        "gt_mfpt_01": gt_pairwise_mfpt[0, 1].item(),
        "gt_mfpt_02": gt_pairwise_mfpt[0, 2].item(),
        "gt_mfpt_10": gt_pairwise_mfpt[1, 0].item(),
        "gt_mfpt_12": gt_pairwise_mfpt[1, 2].item(),
        "gt_mfpt_20": gt_pairwise_mfpt[2, 0].item(),
        "gt_mfpt_21": gt_pairwise_mfpt[2, 1].item(),
        "lr_mfpt_01": lr_pairwise_mfpt[0, 1].item(),
        "lr_mfpt_02": lr_pairwise_mfpt[0, 2].item(),
        "lr_mfpt_10": lr_pairwise_mfpt[1, 0].item(),
        "lr_mfpt_12": lr_pairwise_mfpt[1, 2].item(),
        "lr_mfpt_20": lr_pairwise_mfpt[2, 0].item(),
        "lr_mfpt_21": lr_pairwise_mfpt[2, 1].item(),
        "its_t2_err": its_t2_err,
        "gt_t2": gt_t2,
        "lr_t2": lr_t2,
        "stat_l1_err": stat_l1_err,
        "gt_stat_0": gt_stat[0].item(),
        "gt_stat_1": gt_stat[1].item(),
        "gt_stat_2": gt_stat[2].item(),
        "lr_stat_0": lr_stat[0].item(),
        "lr_stat_1": lr_stat[1].item(),
        "lr_stat_2": lr_stat[2].item(),
        "exit_frac": exit_frac,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-seeds", type=int, default=10)
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--sde-epochs", type=int, default=MB_SDE_EPOCHS)
    parser.add_argument("--base-seed", type=int, default=42)
    parser.add_argument("--D", type=int, default=11)
    parser.add_argument("--N", type=int, default=N_TRAIN)
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--surfaces", type=str, nargs="+", default=None)
    parser.add_argument("--conditions", type=str, nargs="+", default=None,
                        help="Conditions to run (default: all for D=11, drop C for D=201)")
    parser.add_argument("--n-traj", type=int, default=None,
                        help="Override N_TRAJ for MFPT simulation (default: from data_driven_sde)")
    parser.add_argument("--uniform", action="store_true",
                        help="Use uniform sampling instead of importance sampling")
    args = parser.parse_args()

    if args.n_traj is not None:
        import experiments.data_driven_sde as _dds
        _dds.N_TRAJ = args.n_traj
        global N_TRAJ
        N_TRAJ = args.n_traj

    if args.output is None:
        args.output = f"mb_mfpt_d{args.D}.csv"

    surfaces = args.surfaces if args.surfaces else SURFACES
    seeds = [args.base_seed + i * 1000 for i in range(args.n_seeds)]

    # Select conditions based on D
    if args.conditions:
        conditions = {k: v for k, v in ALL_CONDITIONS.items() if k in args.conditions}
    elif args.D >= 201:
        conditions = D201_CONDITIONS
    else:
        conditions = ALL_CONDITIONS

    cond_names = list(conditions.keys())
    total = len(seeds) * len(surfaces) * len(cond_names)

    print(f"Device: {DEVICE}")
    print(f"D={args.D}, N={args.N}")
    print(f"Surfaces: {surfaces}")
    print(f"Conditions: {cond_names}")
    print(f"Seeds ({len(seeds)}): {seeds}")
    print(f"LONG_T={LONG_T}, LONG_N_STEPS={LONG_N_STEPS}, MB_DT={MB_DT}")
    print(f"Total runs: {total}\n")

    t0 = time.time()
    all_rows = []

    for surface_name in surfaces:
        for seed in seeds:
            print(f"\n{'='*60}")
            print(f"  {surface_name} | seed={seed}")
            print(f"{'='*60}")

            torch.manual_seed(seed)
            np.random.seed(seed)

            surface = FourierAugmentedSurface(surface_name, args.D)
            if args.uniform:
                # Uniform sampling
                train_data = sample_from_highd_manifold(
                    surface, mb_local_drift_fn, mb_local_diffusion_fn,
                    [(-TRAIN_BOUND, TRAIN_BOUND), (-TRAIN_BOUND, TRAIN_BOUND)],
                    n_samples=args.N, seed=seed, device=DEVICE,
                )
            else:
                # Importance-sample: 30% near saddle, 70% uniform
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
            sde = create_highd_lambdified_sde(surface, mb_local_drift_fn, mb_local_diffusion_fn)

            # Well centers in ambient space
            wells_ambient = sde.chart(WELLS_UV.to(DEVICE)).to(DEVICE)

            # GT simulation (shared across conditions for same surface+seed)
            torch.manual_seed(seed + 999)
            init_local = torch.randn(N_TRAJ, 2, device=DEVICE) * 0.1
            init_local[:, 0] += WELLS_UV[0, 0]
            init_local[:, 1] += WELLS_UV[0, 1]
            init_local.clamp_(-TRAIN_BOUND, TRAIN_BOUND)
            init_ambient = sde.chart(init_local).to(DEVICE)

            torch.manual_seed(seed + 1234)
            dW_gt = torch.randn(N_TRAJ, LONG_N_STEPS, 2, device=DEVICE)
            _, _, gt_local = simulate_ground_truth(
                init_local, sde, LONG_N_STEPS, MB_DT, dW_gt,
                boundary=None,
            )
            gt_uv = gt_local
            # Censor GT the same way as learned (matched treatment)
            gt_ever_out = (gt_uv.abs() > CENSOR_BOUND).any(dim=-1).any(dim=-1)
            gt_assign = assign_wells_coreset_2d(gt_uv, WELLS_UV.to(DEVICE))
            gt_assign[gt_ever_out] = -1
            gt_exit_frac = gt_ever_out.float().mean().item()
            gt_pairwise_mfpt = compute_pairwise_mfpt(gt_assign, dt=MB_DT)
            gt_its = compute_implied_timescales(gt_assign, dt=MB_DT, lag_times=LAG_TIMES)
            gt_stat = compute_stationary_probabilities(gt_assign)
            print(f"  GT exit fraction: {gt_exit_frac:.1%}")
            # GT ambient at T=1.0
            step_1 = int(round(1.0 / MB_DT))
            gt_at_1 = sde.chart(gt_local[:, step_1]) if step_1 < gt_local.shape[1] else None
            del dW_gt, gt_local  # free memory

            gt_results = {
                'gt_assign': gt_assign, 'gt_pairwise_mfpt': gt_pairwise_mfpt,
                'gt_its': gt_its, 'gt_stat': gt_stat, 'gt_at_1': gt_at_1,
                'init_local': init_local, 'init_ambient': init_ambient,
            }

            for cond_label, lw in conditions.items():
                row = run_one(
                    surface_name, args.D, seed, cond_label, lw,
                    args.epochs, args.sde_epochs,
                    sde, wells_ambient, train_data, x, v, Lambda,
                    n_train=args.N, gt_results=gt_results,
                )
                all_rows.append(row)

            # Save incrementally
            pd.DataFrame(all_rows).to_csv(args.output, index=False)

    elapsed = time.time() - t0
    print(f"\n\nTotal time: {elapsed / 60:.1f} min")

    df = pd.DataFrame(all_rows)
    df.to_csv(args.output, index=False)
    print(f"Saved {len(df)} rows to {args.output}")

    # ── Summary ───────────────────────────────────────────────────────
    print(f"\n\n{'='*80}")
    print(f"MB MFPT SUMMARY (D={args.D})")
    print(f"{'='*80}")

    for surface_name in surfaces:
        sub = df[df["surface"] == surface_name]
        print(f"\n  Surface: {surface_name}")
        print(f"    {'condition':>10s}  {'pw_mfpt_err':>12s}  {'ITS_t2_err':>12s}  "
              f"{'stat_L1':>10s}  {'exit_frac':>10s}  {'W2@1.0':>12s}  {'E_mu':>10s}")
        for cond in cond_names:
            cs = sub[sub["condition"] == cond]
            pw = cs["pairwise_mfpt_err"].values
            it = cs["its_t2_err"].values
            sl = cs["stat_l1_err"].values
            ef = cs["exit_frac"].values
            w2 = cs["W2@1.0"].values
            em = cs["E_mu"].values
            print(f"    {cond:>10s}  "
                  f"{np.nanmean(pw):>5.1%}±{np.nanstd(pw):>.1%}  "
                  f"{np.nanmean(it):>5.1%}±{np.nanstd(it):>.1%}  "
                  f"{np.nanmean(sl):>5.4f}  "
                  f"{np.nanmean(ef):>5.1%}  "
                  f"{np.nanmean(w2):>5.4f}±{np.nanstd(w2):>.4f}  "
                  f"{np.nanmean(em):>5.4f}")

    # Paired t-test: each condition vs baseline
    print(f"\n  Paired t-test vs baseline (negative = better):")
    for surface_name in surfaces:
        sub = df[df["surface"] == surface_name]
        print(f"\n    {surface_name}:")
        base = sub[sub["condition"] == "baseline"].sort_values("seed")
        for cond in [c for c in cond_names if c != "baseline"]:
            cond_s = sub[sub["condition"] == cond].sort_values("seed")
            if len(base) != len(cond_s) or len(base) < 2:
                continue

            for metric in ["pairwise_mfpt_err",
                           "its_t2_err", "stat_l1_err"]:
                bv = base[metric].values
                cv = cond_s[metric].values
                mask = np.isfinite(bv) & np.isfinite(cv)
                if mask.sum() < 2:
                    continue
                t_stat, p_val = stats.ttest_rel(cv[mask], bv[mask])
                diff = cv[mask] - bv[mask]
                delta_pct = diff.mean() / bv[mask].mean() * 100 if bv[mask].mean() != 0 else 0
                sig = "**" if p_val < 0.01 else "*" if p_val < 0.05 else "+" if p_val < 0.1 else ""
                wins = (cv[mask] < bv[mask]).sum()
                print(f"      {cond:>6s} {metric:>15s}: Δ={delta_pct:+.1f}% p={p_val:.3f}{sig} "
                      f"wins={wins}/{mask.sum()}")


if __name__ == "__main__":
    main()
