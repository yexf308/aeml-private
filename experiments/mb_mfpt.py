"""
MB inter-well MFPT + transition rate experiment.

4-5 conditions × 4 surfaces × D × 10 seeds.
Uses MB drift + state-dependent diffusion. Independent noise for GT vs learned.

Metrics per (surface, condition, seed):
  - E_mu, E_Sigma: coefficient errors
  - Inter-well MFPT (well 1 → any other)
  - Transition rate (#transitions / total_time)
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
    N_TRAJ, DT, BOUNDARY,
)
from experiments.highd_N_D_sweep import (
    hidden_dims_for_D, N_EVAL, compute_coefficient_errors,
)
from experiments.mfpt_full_ablation import train_pipeline

from experiments.mb_dynamics import (
    mb_local_drift_fn, mb_local_diffusion_fn,
    train_ae_highd,
    WELLS_UV, WELL_NAMES, TRAIN_BOUND,
    assign_wells_ambient, compute_interwell_mfpt, compute_transition_rate,
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
N_TRAIN = 50

# Simulation parameters
LONG_T = 20.0
LONG_N_STEPS = 2000  # = LONG_T / DT


def run_one(surface_name, D, seed, cond_label, lw, epochs, sde_epochs,
            sde, wells_ambient, train_data, x, v, Lambda, n_train=50):
    """Run one (surface, condition, seed) configuration."""
    print(f"\n  --- {cond_label} ---")
    torch.manual_seed(seed)
    np.random.seed(seed)

    ae, recon = train_ae_highd(
        FourierAugmentedSurface(surface_name, D),
        D, seed, n_train, epochs, lw, train_data,
    )
    print(f"    recon={recon:.6f}")

    # σ_min
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
    pipeline = train_pipeline(ae, x, v, Lambda, seed, n_train, sde_epochs)

    # Initial conditions at well 1 + noise
    torch.manual_seed(seed + 999)
    init_local = torch.randn(N_TRAJ, 2, device=DEVICE) * 0.1
    init_local[:, 0] += WELLS_UV[0, 0]
    init_local[:, 1] += WELLS_UV[0, 1]
    init_local.clamp_(-TRAIN_BOUND, TRAIN_BOUND)
    init_ambient = sde.chart(init_local).to(DEVICE)

    # GT trajectories (independent noise)
    torch.manual_seed(seed + 1234)
    dW_gt = torch.randn(N_TRAJ, LONG_N_STEPS, 2, device=DEVICE)
    gt_traj, gt_alive, _ = simulate_ground_truth(
        init_local, sde, LONG_N_STEPS, DT, dW_gt, BOUNDARY * 3,
    )

    # Learned trajectories (independent noise, deterministic seed per condition)
    _COND_SEED_OFFSET = {"baseline": 0, "T": 1, "F": 2, "C": 3, "T+F": 4}
    torch.manual_seed(seed + 5678 + _COND_SEED_OFFSET.get(cond_label, 99))
    dW_lr = torch.randn(N_TRAJ, LONG_N_STEPS, 2, device=DEVICE)
    with torch.no_grad():
        z0 = ae.encoder(init_ambient)
    _, x_traj = pipeline.simulate(z0, LONG_N_STEPS, DT, dW=dW_lr)

    # Inter-well metrics
    gt_assign = assign_wells_ambient(gt_traj, wells_ambient)
    lr_assign = assign_wells_ambient(x_traj, wells_ambient)

    gt_mfpt = compute_interwell_mfpt(gt_assign, start_well=0, dt=DT)
    lr_mfpt = compute_interwell_mfpt(lr_assign, start_well=0, dt=DT)
    gt_rate = compute_transition_rate(gt_assign, dt=DT)
    lr_rate = compute_transition_rate(lr_assign, dt=DT)

    # MFPT error
    if (np.isfinite(gt_mfpt['mean']) and gt_mfpt['mean'] > 0
            and np.isfinite(lr_mfpt['mean'])):
        mfpt_err = abs(lr_mfpt['mean'] - gt_mfpt['mean']) / gt_mfpt['mean']
    else:
        mfpt_err = float('nan')

    # Rate error (if learned has 0 rate but GT doesn't, that's 100% error)
    if gt_rate['rate'] > 0:
        rate_err = abs(lr_rate['rate'] - gt_rate['rate']) / gt_rate['rate']
    else:
        rate_err = float('nan')

    # W2 at T=1.0
    step_1 = int(round(1.0 / DT))
    both_alive = gt_alive[:, step_1] if step_1 < gt_alive.shape[1] else gt_alive[:, -1]
    w2 = compute_w2(x_traj[:, step_1], gt_traj[:, step_1], both_alive, both_alive)

    print(f"    MFPT: GT={gt_mfpt['mean']:.3f} learned={lr_mfpt['mean']:.3f} "
          f"err={mfpt_err:.1%}")
    print(f"    Rate: GT={gt_rate['rate']:.4f} learned={lr_rate['rate']:.4f} "
          f"err={rate_err:.1%}")

    return {
        "surface": surface_name,
        "condition": cond_label,
        "seed": seed,
        "recon_per_dim": recon,
        "E_mu": e_mu,
        "E_Sigma": e_sigma,
        "sigma_min_p5": sigma_min_vals.quantile(0.05).item(),
        "gt_mfpt": gt_mfpt["mean"],
        "lr_mfpt": lr_mfpt["mean"],
        "mfpt_err": mfpt_err,
        "gt_rate": gt_rate["rate"],
        "lr_rate": lr_rate["rate"],
        "rate_err": rate_err,
        "transition_frac": lr_mfpt["transition_frac"],
        "gt_transition_frac": gt_mfpt["transition_frac"],
        "gt_transitions": gt_rate["total_transitions"],
        "lr_transitions": lr_rate["total_transitions"],
        "W2@1.0": w2,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-seeds", type=int, default=10)
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--sde-epochs", type=int, default=300)
    parser.add_argument("--base-seed", type=int, default=42)
    parser.add_argument("--D", type=int, default=11)
    parser.add_argument("--N", type=int, default=N_TRAIN)
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--surfaces", type=str, nargs="+", default=None)
    parser.add_argument("--conditions", type=str, nargs="+", default=None,
                        help="Conditions to run (default: all for D=11, drop C for D=201)")
    args = parser.parse_args()

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
    print(f"LONG_T={LONG_T}, LONG_N_STEPS={LONG_N_STEPS}")
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
            train_data = sample_from_highd_manifold(
                surface, mb_local_drift_fn, mb_local_diffusion_fn,
                [(-TRAIN_BOUND, TRAIN_BOUND), (-TRAIN_BOUND, TRAIN_BOUND)],
                n_samples=args.N, seed=seed, device=DEVICE,
            )
            x = train_data.samples.to(DEVICE)
            v = train_data.mu.to(DEVICE)
            Lambda = train_data.cov.to(DEVICE)
            sde = create_highd_lambdified_sde(surface, mb_local_drift_fn, mb_local_diffusion_fn)

            # Well centers in ambient space
            wells_ambient = sde.chart(WELLS_UV.to(DEVICE)).to(DEVICE)

            for cond_label, lw in conditions.items():
                row = run_one(
                    surface_name, args.D, seed, cond_label, lw,
                    args.epochs, args.sde_epochs,
                    sde, wells_ambient, train_data, x, v, Lambda,
                    n_train=args.N,
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
        print(f"    {'condition':>10s}  {'MFPT_err':>12s}  {'rate_err':>12s}  "
              f"{'trans_frac':>10s}  {'W2@1.0':>12s}  {'E_mu':>10s}")
        for cond in cond_names:
            cs = sub[sub["condition"] == cond]
            me = cs["mfpt_err"].values
            re = cs["rate_err"].values
            tf = cs["transition_frac"].values
            w2 = cs["W2@1.0"].values
            em = cs["E_mu"].values
            print(f"    {cond:>10s}  {np.nanmean(me):>5.1%}±{np.nanstd(me):>.1%}  "
                  f"{np.nanmean(re):>5.1%}±{np.nanstd(re):>.1%}  "
                  f"{np.nanmean(tf):>5.0%}  "
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

            for metric in ["mfpt_err", "rate_err"]:
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
                print(f"      {cond:>6s} {metric:>10s}: Δ={delta_pct:+.1f}% p={p_val:.3f}{sig} "
                      f"wins={wins}/{mask.sum()}")


if __name__ == "__main__":
    main()
