"""
N×D Sweep: Does training set size interact with K benefit at high D?

K regularization helps at N=20 (sparse) but not N=2000 (abundant) for D=3.
The stabilized high-D study shows K helps strongly at D=201 with N=20.
Question: does the "crossover N" where K stops helping shift with D?
Hypothesis: higher D extends the sparse regime to larger N.

Design:
  Shared-checkpoint fork (same as highd_K_study.py):
  Phase 1 (T+F warmup) trained ONCE per seed, then forked into conditions.

  Parameters:
    - N ∈ {20, 50, 100, 200}
    - D ∈ {11, 201} (low-D vs high-D extremes)
    - Surfaces: paraboloid, hyperbolic_paraboloid
    - Conditions: baseline (no Phase 2), K(Phat) (Phase 2 with curvature)
    - 5 seeds → 4 × 2 × 2 × 2 × 5 = 160 runs

  Hidden dims: D-dependent (<=11:[64,64], <=51:[128,128], else:[256,256]).

Usage:
    # Smoke test
    python -m experiments.highd_N_D_sweep --n-seeds 1 --epochs 50 --sde-epochs 50 --d-values 11 --n-values 20 50

    # Full GPU run
    srun --partition=dgx --gres=gpu:1 --time=02:00:00 \\
      python3 -u -m experiments.highd_N_D_sweep --n-seeds 5
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
from src.numeric.highd_manifolds import (
    FourierAugmentedSurface,
    sample_from_highd_manifold,
    create_highd_lambdified_sde,
)

from experiments.common import make_model_config
from experiments.data_driven_sde import (
    evaluate_pipeline,
    TRAIN_BOUND, BATCH_SIZE, LR_AE, LR_SDE, DEVICE,
    N_TRAJ, N_STEPS, DT, BOUNDARY,
)

# D configs: (K_fourier_pairs, D_ambient)
D_CONFIGS = [
    (4, 11),    # low-D
    (99, 201),  # high-D (Hessian-free curvature)
]

SURFACES = ["paraboloid", "hyperbolic_paraboloid"]
N_VALUES = [20, 50, 100, 200]


def hidden_dims_for_D(D: int) -> list:
    """Scale AE hidden dims with ambient dimension."""
    if D <= 11:
        return [64, 64]
    elif D <= 51:
        return [128, 128]
    else:
        return [256, 256]

# Phase 1: T+F warmup (no curvature)
WARMUP_LW = LossWeights(tangent_bundle=1.0, diffeo=1.0)
# Phase 2: T+F+K finetune
FULL_LW = LossWeights(tangent_bundle=1.0, diffeo=1.0, curvature=0.1)

# (label, phase2_loss_weights_or_None)
CONDITIONS = [
    ("baseline", None),
    ("K(Phat)", FULL_LW),
]

METRICS = ["MTE@0.1", "MTE@0.5", "MTE@1.0", "W2@1.0", "MMD@1.0"]


# ── Local dynamics (same as highd_K_study.py) ─────────────────────────────

def local_drift_fn(uv: torch.Tensor) -> torch.Tensor:
    """Rotation drift: (-v, u). Input: (B, 2), output: (B, 2)."""
    return torch.stack([-uv[:, 1], uv[:, 0]], dim=-1)


def local_diffusion_fn(uv: torch.Tensor) -> torch.Tensor:
    """State-dependent diffusion. Input: (B, 2), output: (B, 2, 2)."""
    u, v = uv[:, 0], uv[:, 1]
    B = uv.shape[0]
    sigma = torch.zeros(B, 2, 2, device=uv.device)
    sigma[:, 0, 0] = 1 + u ** 2 / 4
    sigma[:, 0, 1] = u + v
    sigma[:, 1, 1] = 1 + v ** 2 / 4
    return sigma


# ── Main experiment ─────────────────────────────────────────────────────────

def run_single_fork(surface_name, D, seed, n_train=20, epochs_ae=500, epochs_sde=300):
    """Run shared-checkpoint fork for one (surface, D, seed, n_train).

    Returns dict mapping condition_label -> metrics dict.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)

    hdims = hidden_dims_for_D(D)
    surface = FourierAugmentedSurface(surface_name, D)
    batch_size = min(n_train, BATCH_SIZE)

    # Sample data numerically (no SymPy)
    train_data = sample_from_highd_manifold(
        surface, local_drift_fn, local_diffusion_fn,
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

    RECON_THRESHOLD = 0.1  # per-dim MSE
    phase1_converged = recon_per_dim < RECON_THRESHOLD
    if not phase1_converged:
        print(f"    WARNING: Phase 1 recon_per_dim={recon_per_dim:.4f} > {RECON_THRESHOLD}")

    # Snapshot Phase 1 state
    phase1_state = copy.deepcopy(trainer.models["ae"].state_dict())
    phase1_optim_state = copy.deepcopy(trainer.optimizers["ae"].state_dict())
    phase1_sched_state = copy.deepcopy(trainer.schedulers["ae"].state_dict())

    # Lambdify SDE for evaluation (numeric, no SymPy)
    sde = create_highd_lambdified_sde(surface, local_drift_fn, local_diffusion_fn)

    # ── Fork into conditions ──
    phase2_epochs = epochs_ae - phase1_epochs
    results = {}

    for cond_label, phase2_lw in CONDITIONS:
        print(f"      Condition: {cond_label}")

        # Create fresh trainer, load Phase 1 checkpoint
        t2 = MultiModelTrainer(TrainingConfig(
            epochs=epochs_ae, n_samples=n_train, input_dim=D, hidden_dim=hdims[0],
            latent_dim=2, learning_rate=LR_AE, batch_size=batch_size,
            test_size=0.03, print_interval=max(1, phase2_epochs // 5), device=DEVICE,
        ))
        mc2 = make_model_config("ae", WARMUP_LW, extrinsic_dim=D, hidden_dims=hdims)
        t2.add_model(mc2)
        t2._has_local_cov = True  # Match Phase 1 loader layout
        t2.models["ae"].load_state_dict(phase1_state)
        t2.optimizers["ae"].load_state_dict(phase1_optim_state)
        t2.schedulers["ae"].load_state_dict(phase1_sched_state)

        if phase2_lw is not None and phase1_converged:
            warmup_frac = 0.2
            warmup_epochs = int(phase2_epochs * warmup_frac)
            for epoch in range(phase2_epochs):
                if epoch < warmup_epochs:
                    ramp = (epoch + 1) / warmup_epochs
                    lw_epoch = LossWeights(
                        tangent_bundle=phase2_lw.tangent_bundle,
                        diffeo=phase2_lw.diffeo,
                        curvature=phase2_lw.curvature * ramp,
                    )
                else:
                    lw_epoch = phase2_lw
                ep_losses = t2.train_epoch(loader, {mc2.name: lw_epoch})
                if (epoch + 1) % max(1, phase2_epochs // 5) == 0:
                    print(f"        Phase2 Epoch {epoch+1}/{phase2_epochs}: "
                          f"loss={ep_losses['ae']:.6f}")
        elif phase2_lw is not None:
            print(f"        Skipping Phase 2 (Phase 1 unconverged)")

        ae = t2.models["ae"]
        ae.eval()

        # Precompute decoder derivatives for SDE stages
        d = 2
        dummy = SDEPipelineTrainer(
            ae, DriftNet(d).to(DEVICE), DiffusionNet(d).to(DEVICE), device=DEVICE,
        )
        z_pre, dphi_pre, d2phi_pre = dummy.precompute_decoder_derivatives(x)

        # Stage 2: Drift
        torch.manual_seed(seed + 100)
        drift_net = DriftNet(d).to(DEVICE)
        dp = SDEPipelineTrainer(ae, drift_net, DiffusionNet(d).to(DEVICE), device=DEVICE)
        drift_losses = dp.train_stage2_precomputed(
            z_pre, dphi_pre, d2phi_pre, v, Lambda,
            epochs=epochs_sde, lr=LR_SDE, batch_size=batch_size,
            print_interval=max(1, epochs_sde // 5),
        )
        drift_net.eval()

        # Stage 3: Diffusion
        torch.manual_seed(seed + 200)
        diff_net = DiffusionNet(d).to(DEVICE)
        pipeline = SDEPipelineTrainer(ae, drift_net, diff_net, device=DEVICE)
        diff_losses = pipeline.train_stage3_precomputed(
            z_pre, dphi_pre, Lambda,
            epochs=epochs_sde, lr=LR_SDE, batch_size=batch_size,
            print_interval=max(1, epochs_sde // 5),
        )

        # Evaluate
        eval_results = evaluate_pipeline(pipeline, ae, sde, seed)
        results[cond_label] = {
            "ae_loss": ep_losses["ae"] if (phase2_lw is not None and phase1_converged) else losses["ae"],
            "drift_loss": drift_losses[-1],
            "diff_loss": diff_losses[-1],
            "recon_per_dim": recon_per_dim,
            "phase1_converged": phase1_converged,
            **eval_results,
        }

    return results


def paired_ttest(vals_a, vals_b):
    """Paired t-test: is mean(a - b) significantly different from 0?"""
    diffs = np.array(vals_a) - np.array(vals_b)
    n = len(diffs)
    if n < 2:
        return float("nan"), float("nan")
    mean_d = diffs.mean()
    std_d = diffs.std(ddof=1)
    if std_d < 1e-15:
        return mean_d, 0.0
    t_stat = mean_d / (std_d / np.sqrt(n))
    p_val = stats.t.sf(abs(t_stat), df=n - 1) * 2
    return mean_d, p_val


def print_summary(df):
    """Print grouped summary tables with paired t-tests, grouped by (D, N)."""
    cond_labels = [c[0] for c in CONDITIONS]

    print(f"\n\n{'='*120}")
    print("N×D SWEEP: TRAINING SET SIZE × AMBIENT DIMENSION INTERACTION")
    print(f"{'='*120}")

    for D_val in sorted(df["D"].unique()):
        for N_val in sorted(df["N"].unique()):
            df_dn = df[(df["D"] == D_val) & (df["N"] == N_val)]
            if len(df_dn) == 0:
                continue

            print(f"\n{'─'*120}")
            print(f"  D = {D_val}, N = {N_val}")
            print(f"{'─'*120}")

            # Mean ± std table
            print(f"\n  {'surface':>25s}  {'condition':>10s}  {'n':>3s}  ", end="")
            for m in METRICS:
                print(f"{'mean':>8s} {'+-std':>8s}  ", end="")
            print()
            print("  " + "-" * 110)

            for surface_name in SURFACES:
                for cond_label in cond_labels:
                    subset = df_dn[
                        (df_dn["surface"] == surface_name)
                        & (df_dn["condition"] == cond_label)
                    ]
                    n = len(subset)
                    print(f"  {surface_name:>25s}  {cond_label:>10s}  {n:>3d}  ", end="")
                    for m in METRICS:
                        vals = subset[m].values
                        print(f"{vals.mean():>8.4f} {vals.std():>8.4f}  ", end="")
                    print()

            # Paired comparisons: K(Phat) vs baseline
            print(f"\n  K(Phat) vs baseline:")
            print(f"    {'surface':>25s}  ", end="")
            for m in METRICS:
                print(f"{'delta':>8s} {'p-val':>8s}  ", end="")
            print("  win-rate  verdict")
            print("    " + "-" * 108)

            for surface_name in SURFACES:
                a = df_dn[
                    (df_dn["surface"] == surface_name) & (df_dn["condition"] == "K(Phat)")
                ].sort_values("seed")
                b = df_dn[
                    (df_dn["surface"] == surface_name) & (df_dn["condition"] == "baseline")
                ].sort_values("seed")

                print(f"    {surface_name:>25s}  ", end="")
                verdicts = []
                for m in METRICS:
                    mean_d, p_val = paired_ttest(a[m].values, b[m].values)
                    sig = "**" if p_val < 0.01 else "*" if p_val < 0.05 else "+" if p_val < 0.1 else ""
                    print(f"{mean_d:>+8.4f} {p_val:>7.4f}{sig:<1s} ", end="")
                    if p_val < 0.05:
                        verdicts.append(f"{m}: {'worse' if mean_d > 0 else 'better'}")

                # Win rate on MTE@1.0 (lower is better)
                a_mte = a["MTE@1.0"].values
                b_mte = b["MTE@1.0"].values
                wins = np.sum(a_mte < b_mte)
                total = len(a_mte)
                win_str = f"{wins}/{total}"

                verdict_str = "; ".join(verdicts) if verdicts else "n.s."
                print(f"  {win_str:>8s}  {verdict_str}")


def main():
    parser = argparse.ArgumentParser(
        description="N×D Sweep: training set size × ambient dimension interaction",
    )
    parser.add_argument("--n-seeds", type=int, default=5)
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--sde-epochs", type=int, default=300)
    parser.add_argument("--base-seed", type=int, default=42)
    parser.add_argument("--d-values", type=int, nargs="+", default=None,
                        help="D values to test (default: 11 201)")
    parser.add_argument("--n-values", type=int, nargs="+", default=None,
                        help="N (training set size) values to test (default: 20 50 100 200)")
    args = parser.parse_args()

    # Determine D configs
    if args.d_values is not None:
        d_configs = [(D - 3) // 2 for D in args.d_values]
        d_configs = list(zip(d_configs, args.d_values))
    else:
        d_configs = D_CONFIGS

    n_values = args.n_values if args.n_values is not None else N_VALUES

    seeds = [args.base_seed + i * 1000 for i in range(args.n_seeds)]
    cond_labels = [c[0] for c in CONDITIONS]

    print(f"Device: {DEVICE}")
    print(f"Seeds ({len(seeds)}): {seeds}")
    print(f"Surfaces: {SURFACES}")
    print(f"Conditions: {cond_labels}")
    print(f"D configs: {d_configs}")
    print(f"N values: {n_values}")
    print(f"Hidden dims: D-dependent (<=11:[64,64], <=51:[128,128], else:[256,256])")
    print(f"AE epochs: {args.epochs} (Phase 1: {args.epochs // 2}, Phase 2: {args.epochs - args.epochs // 2})")
    print(f"SDE epochs: {args.sde_epochs}")
    expected_rows = len(n_values) * len(d_configs) * len(SURFACES) * len(cond_labels) * len(seeds)
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

                        print(f"    {cond_label:>10s}: MTE@1.0={metrics['MTE@1.0']:.4f}  "
                              f"W2@1.0={metrics['W2@1.0']:.4f}  MMD@1.0={metrics['MMD@1.0']:.4f}")

            elapsed_dn = time.time() - t_dn
            print(f"\n  N={N}, D={D_val} completed in {elapsed_dn:.1f}s "
                  f"({elapsed_dn / (len(SURFACES) * len(seeds)):.1f}s per fork)")

    df = pd.DataFrame(all_rows)
    csv_path = "highd_N_D_sweep.csv"
    df.to_csv(csv_path, index=False)
    print(f"\nSaved {len(df)} rows to {csv_path}")

    print_summary(df)

    print(f"\nTotal time: {time.time() - t0:.1f}s")
    print("Done.")


if __name__ == "__main__":
    main()
