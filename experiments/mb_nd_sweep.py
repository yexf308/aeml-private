"""
N×D Sweep with MB dynamics: encoder-pullback vs decoder-side drift.

Tests whether training set size interacts with K benefit at high D,
now using metastable MB drift instead of rotation drift.

Design mirrors highd_N_D_sweep.py but uses MB dynamics:
  - Shared-checkpoint fork: Phase 1 (T+F warmup) trained ONCE per seed
  - Phase 2 AE grouped: baseline (T+F) and K (T+F+K)
  - Parameters:
    - N in {20, 50, 100, 200}
    - D in {11, 201}
    - Surfaces: paraboloid, hyperbolic_paraboloid
    - Conditions: baseline (T+F), K (T+F+K), K+S (T+F+K + smooth)
    - 10 seeds

Output: mb_nd_sweep.csv
"""

import argparse
import copy
import time
import torch
import numpy as np
import pandas as pd
from scipy import stats

from src.numeric.losses import LossWeights
from src.numeric.sde_nets import DriftNet, DiffusionNet
from src.numeric.sde_training import SDEPipelineTrainer
from src.numeric.training import MultiModelTrainer, TrainingConfig
from src.numeric.geometry import regularized_metric_inverse, ambient_quadratic_variation_drift
from src.numeric.highd_manifolds import (
    FourierAugmentedSurface,
    sample_from_highd_manifold,
    create_highd_lambdified_sde,
)

from experiments.common import make_model_config
from experiments.data_driven_sde import (
    evaluate_pipeline,
    BATCH_SIZE, LR_AE, LR_SDE, DEVICE,
    N_TRAJ, N_STEPS, DT, BOUNDARY,
)
from experiments.highd_N_D_sweep import (
    hidden_dims_for_D, N_EVAL, compute_coefficient_errors,
    _train_phase2, _run_sde_stages,
    WARMUP_LW, BASELINE_LW, BASE_K_WEIGHT, D_REF,
    AUG_SIGMA, METRICS, paired_ttest,
)
from experiments.mb_dynamics import (
    mb_local_drift_fn, mb_local_diffusion_fn,
    TRAIN_BOUND,
)


# D configs: (K_fourier_pairs, D_ambient)
D_CONFIGS = [
    (4, 11),    # low-D
    (99, 201),  # high-D
]

SURFACES = ["paraboloid", "hyperbolic_paraboloid"]
N_VALUES = [20, 50, 100, 200]
LAMBDA_SMOOTH = 0.0


def make_conditions(D, lambda_smooth=LAMBDA_SMOOTH):
    """Create conditions with sqrt(D/D_ref) K weight scaling."""
    k_weight = BASE_K_WEIGHT * (D / D_REF) ** 0.5
    full_lw = LossWeights(tangent_bundle=1.0, diffeo=1.0, curvature=k_weight)
    return [
        ("baseline", BASELINE_LW, 0.0),
        ("K",        full_lw,     0.0),
        ("K+S",      full_lw,     lambda_smooth),
    ]


COND_LABELS = ["baseline", "K", "K+S"]


def run_single_fork(surface_name, D, seed, n_train=20, epochs_ae=500,
                    epochs_sde=300, lambda_smooth=LAMBDA_SMOOTH):
    """Run shared-checkpoint fork for one (surface, D, seed, n_train) with MB dynamics."""
    torch.manual_seed(seed)
    np.random.seed(seed)

    hdims = hidden_dims_for_D(D)
    surface = FourierAugmentedSurface(surface_name, D)
    batch_size = min(n_train, BATCH_SIZE)

    # Sample with MB drift + state-dependent diffusion
    train_data = sample_from_highd_manifold(
        surface, mb_local_drift_fn, mb_local_diffusion_fn,
        [(-TRAIN_BOUND, TRAIN_BOUND), (-TRAIN_BOUND, TRAIN_BOUND)],
        n_samples=n_train, seed=seed, device=DEVICE,
    )
    x = train_data.samples.to(DEVICE)
    v = train_data.mu.to(DEVICE)
    Lambda = train_data.cov.to(DEVICE)

    # ── Phase 1: T+F warmup (ONCE) ──
    phase1_epochs = max(1, epochs_ae // 2)

    trainer = MultiModelTrainer(TrainingConfig(
        epochs=epochs_ae, n_samples=n_train, input_dim=D, hidden_dim=hdims[0],
        latent_dim=2, learning_rate=LR_AE, batch_size=batch_size,
        test_size=0.03, print_interval=max(1, phase1_epochs // 5), device=DEVICE,
    ))
    mc = make_model_config("ae", WARMUP_LW, extrinsic_dim=D, hidden_dims=hdims)
    trainer.add_model(mc)
    loader = trainer.create_data_loader(train_data)

    for epoch in range(phase1_epochs):
        losses = trainer.train_epoch(loader)
        if (epoch + 1) % max(1, phase1_epochs // 5) == 0:
            print(f"    Phase1 Epoch {epoch+1}/{phase1_epochs}: loss={losses['ae']:.6f}")

    # Phase 1 quality gate
    with torch.no_grad():
        z_check = trainer.models["ae"].encoder(x)
        x_hat_check = trainer.models["ae"].decoder(z_check)
        recon_per_dim = ((x_hat_check - x) ** 2).sum(-1).mean().item() / D

    RECON_THRESHOLD = 0.1
    phase1_converged = recon_per_dim < RECON_THRESHOLD
    if not phase1_converged:
        print(f"    WARNING: Phase 1 recon_per_dim={recon_per_dim:.4f} > {RECON_THRESHOLD}")

    phase1_state = copy.deepcopy(trainer.models["ae"].state_dict())
    phase1_optim_state = copy.deepcopy(trainer.optimizers["ae"].state_dict())
    phase1_sched_state = copy.deepcopy(trainer.schedulers["ae"].state_dict())

    # Lambdify SDE for evaluation (MB drift)
    sde = create_highd_lambdified_sde(surface, mb_local_drift_fn, mb_local_diffusion_fn)

    # ── Phase 2 groups ──
    conditions = make_conditions(D, lambda_smooth)
    phase2_epochs = epochs_ae - phase1_epochs

    groups = {}
    for label, phase2_lw, lam_s in conditions:
        key = id(phase2_lw)
        if key not in groups:
            groups[key] = (phase2_lw, [])
        groups[key][1].append((label, lam_s))

    results = {}

    for phase2_lw, sde_conditions in groups.values():
        print(f"      Phase 2 AE: K_weight={phase2_lw.curvature:.4f}")

        t2 = MultiModelTrainer(TrainingConfig(
            epochs=epochs_ae, n_samples=n_train, input_dim=D, hidden_dim=hdims[0],
            latent_dim=2, learning_rate=LR_AE, batch_size=batch_size,
            test_size=0.03, print_interval=max(1, phase2_epochs // 5), device=DEVICE,
        ))
        mc2 = make_model_config("ae", WARMUP_LW, extrinsic_dim=D, hidden_dims=hdims)
        t2.add_model(mc2)
        t2._has_local_cov = True
        t2.models["ae"].load_state_dict(phase1_state)
        t2.optimizers["ae"].load_state_dict(phase1_optim_state)
        t2.schedulers["ae"].load_state_dict(phase1_sched_state)

        p2_loss = _train_phase2(t2, loader, mc2, phase2_lw, phase2_epochs,
                                phase1_converged)

        ae = t2.models["ae"]
        ae.eval()
        ae_loss = p2_loss if p2_loss is not None else losses["ae"]

        # Coefficient errors
        torch.manual_seed(seed + 5000)
        eval_uv = (torch.rand(N_EVAL, 2, device=DEVICE) * 2 - 1) * TRAIN_BOUND
        e_mu, e_sigma = compute_coefficient_errors(ae, sde, surface, eval_uv, DEVICE)
        print(f"        E_mu={e_mu:.4f}  E_Sigma={e_sigma:.4f}")

        for cond_label, lam_s in sde_conditions:
            smooth_tag = f" (smooth={lam_s})" if lam_s > 0 else ""
            print(f"      → {cond_label}{smooth_tag}")

            sde_results = _run_sde_stages(
                ae, x, v, Lambda, sde, surface, seed, n_train,
                epochs_sde, lam_s, drift_hidden=[128, 128],
            )
            results[cond_label] = {
                "ae_loss": ae_loss,
                "drift_loss": sde_results["drift_loss"],
                "diff_loss": sde_results["diff_loss"],
                "recon_per_dim": recon_per_dim,
                "phase1_converged": phase1_converged,
                "E_mu": e_mu,
                "E_Sigma": e_sigma,
                **{k: v for k, v in sde_results.items()
                   if k not in ("drift_loss", "diff_loss")},
            }

    return results


def print_summary(df):
    """Print grouped summary tables."""
    cond_labels = sorted(df["condition"].unique())

    print(f"\n\n{'='*150}")
    print("MB N×D SWEEP: TRAINING SET SIZE × AMBIENT DIMENSION INTERACTION")
    print(f"{'='*150}")

    for D_val in sorted(df["D"].unique()):
        for N_val in sorted(df["N"].unique()):
            df_dn = df[(df["D"] == D_val) & (df["N"] == N_val)]
            if len(df_dn) == 0:
                continue

            print(f"\n{'─'*150}")
            print(f"  D = {D_val}, N = {N_val}")
            print(f"{'─'*150}")

            print(f"\n  {'surface':>25s}  {'condition':>10s}  {'n':>3s}  ", end="")
            for m in METRICS:
                if m in df_dn.columns:
                    print(f"{'mean':>8s} {'+-std':>8s}  ", end="")
            print()

            for surface_name in SURFACES:
                for cond_label in cond_labels:
                    subset = df_dn[
                        (df_dn["surface"] == surface_name)
                        & (df_dn["condition"] == cond_label)
                    ]
                    n = len(subset)
                    print(f"  {surface_name:>25s}  {cond_label:>10s}  {n:>3d}  ", end="")
                    for m in METRICS:
                        if m in subset.columns:
                            vals = subset[m].values
                            print(f"{np.nanmean(vals):>8.4f} {np.nanstd(vals):>8.4f}  ", end="")
                    print()

            # Paired comparisons
            comparisons = [
                ("K vs baseline", "K", "baseline"),
                ("K+S vs baseline", "K+S", "baseline"),
            ]
            for comp_name, cond_a, cond_b in comparisons:
                print(f"\n  {comp_name}:")
                for surface_name in SURFACES:
                    a = df_dn[
                        (df_dn["surface"] == surface_name) & (df_dn["condition"] == cond_a)
                    ].sort_values("seed")
                    b = df_dn[
                        (df_dn["surface"] == surface_name) & (df_dn["condition"] == cond_b)
                    ].sort_values("seed")
                    if len(a) == 0 or len(b) == 0:
                        continue

                    print(f"    {surface_name:>25s}  ", end="")
                    for m in METRICS:
                        if m not in a.columns:
                            continue
                        av, bv = a[m].values, b[m].values
                        mask = np.isfinite(av) & np.isfinite(bv)
                        if mask.sum() < 2:
                            print(f"{'n/a':>8s} {'n/a':>8s}  ", end="")
                            continue
                        ac, bc = av[mask], bv[mask]
                        delta = (ac.mean() - bc.mean()) / bc.mean() * 100
                        mean_d, p_val = paired_ttest(ac, bc)
                        sig = "**" if p_val < 0.01 else "*" if p_val < 0.05 else "+" if p_val < 0.1 else ""
                        print(f"{delta:>+8.1f}% {p_val:>7.4f}{sig:<1s} ", end="")
                    print()


def main():
    parser = argparse.ArgumentParser(
        description="MB N×D Sweep: training set size × ambient dimension interaction",
    )
    parser.add_argument("--n-seeds", type=int, default=10)
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--sde-epochs", type=int, default=1000)
    parser.add_argument("--base-seed", type=int, default=42)
    parser.add_argument("--d-values", type=int, nargs="+", default=None)
    parser.add_argument("--n-values", type=int, nargs="+", default=None)
    parser.add_argument("--lambda-smooth", type=float, default=LAMBDA_SMOOTH)
    parser.add_argument("--output", type=str, default="mb_nd_sweep.csv")
    args = parser.parse_args()

    if args.d_values is not None:
        d_configs = [((D - 3) // 2, D) for D in args.d_values]
    else:
        d_configs = D_CONFIGS

    n_values = args.n_values if args.n_values is not None else N_VALUES
    seeds = [args.base_seed + i * 1000 for i in range(args.n_seeds)]

    print(f"Device: {DEVICE}")
    print(f"Seeds ({len(seeds)}): {seeds}")
    print(f"Surfaces: {SURFACES}")
    print(f"Conditions: {COND_LABELS}")
    print(f"D configs: {d_configs}")
    print(f"N values: {n_values}")
    print(f"Dynamics: Müller-Brown gradient + state-dependent diffusion")
    print(f"TRAIN_BOUND: {TRAIN_BOUND}")
    expected_rows = (len(n_values) * len(d_configs) * len(SURFACES)
                     * len(COND_LABELS) * len(seeds))
    print(f"Expected rows: {expected_rows}\n")

    t0 = time.time()
    all_rows = []

    for N in n_values:
        for K_pairs, D_val in d_configs:
            t_dn = time.time()
            for surface_name in SURFACES:
                for seed in seeds:
                    print(f"\n{'='*60}")
                    print(f"  N={N} | D={D_val} (K={K_pairs}) | {surface_name} | seed={seed}")
                    print(f"{'='*60}")

                    fork_results = run_single_fork(
                        surface_name, D_val, seed, n_train=N,
                        epochs_ae=args.epochs, epochs_sde=args.sde_epochs,
                        lambda_smooth=args.lambda_smooth,
                    )
                    for cond_label, metrics in fork_results.items():
                        all_rows.append({
                            "N": N,
                            "D": D_val,
                            "K_pairs": K_pairs,
                            "surface": surface_name,
                            "condition": cond_label,
                            "seed": seed,
                            **metrics,
                        })

            elapsed_dn = time.time() - t_dn
            print(f"\n  N={N}, D={D_val} completed in {elapsed_dn:.1f}s")

    df = pd.DataFrame(all_rows)
    df.to_csv(args.output, index=False)
    print(f"\nSaved {len(df)} rows to {args.output}")

    print_summary(df)
    print(f"\nTotal time: {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
