"""
N×D Sweep: Does training set size interact with K benefit at high D?

K regularization helps at N=20 (sparse) but not N=2000 (abundant) for D=3.
Question: does the "crossover N" where K stops helping shift with D?
Hypothesis: higher D extends the sparse regime to larger N.

Design:
  Shared-checkpoint fork:
  Phase 1 (T+F warmup) trained ONCE per seed, then forked into conditions.
  Phase 2 AE grouped: baseline (T+F) and K (T+F+K) each trained once.
  Stage 2 drift: K AE forked into {no smooth, smooth} for K and K+S.

  Parameters:
    - N in {20, 50, 100, 200}
    - D in {11, 201}
    - Surfaces: paraboloid, hyperbolic_paraboloid
    - Conditions: baseline (T+F), K (T+F+K sqrt-scaled), K+S (T+F+K + smooth)
    - 10 seeds

  K weight scaling: lambda_K = 0.1 * sqrt(D / D_ref), D_ref = 11.
  Jacobian smoothing: ||J_{drift_net}||_F^2 / d penalty in Stage 2.

Usage:
    # Smoke test
    python -m experiments.highd_N_D_sweep --n-seeds 1 --epochs 50 --sde-epochs 50 --d-values 11 --n-values 20 50

    # Full GPU run
    srun --partition=dgx --gres=gpu:1 --time=04:00:00 \
      python3 -u -m experiments.highd_N_D_sweep --n-seeds 10
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
# Phase 2 baseline: T+F continuation
BASELINE_LW = LossWeights(tangent_bundle=1.0, diffeo=1.0, curvature=0.0)

# K weight scaling: sqrt(D/D_ref) to compensate 1/D normalisation of L_K
BASE_K_WEIGHT = 0.1
D_REF = 11

# Drift smoothness regularisation
LAMBDA_SMOOTH = 0.0
AUG_SIGMA = 0.1

# Exported for plot_nd_sweep.py compatibility
FULL_LW = LossWeights(tangent_bundle=1.0, diffeo=1.0, curvature=BASE_K_WEIGHT)


def make_conditions(D, lambda_smooth=LAMBDA_SMOOTH):
    """Create conditions: (label, phase2_lw, lambda_smooth_stage2).

    K weight uses sqrt(D/D_ref) scaling.  Group-lasso theory
    (Yuan & Lin 2006) prescribes sqrt(group_size) scaling for L2-type
    penalties, keeping per-sample penalty magnitude constant across D.
    """
    k_weight = BASE_K_WEIGHT * (D / D_REF) ** 0.5
    full_lw = LossWeights(tangent_bundle=1.0, diffeo=1.0, curvature=k_weight)
    return [
        ("baseline", BASELINE_LW, 0.0),
        ("K",        full_lw,     0.0),
        ("K+S",      full_lw,     lambda_smooth),
    ]


COND_LABELS = ["baseline", "K", "K+S"]
METRICS = ["MTE@0.1", "MTE@0.5", "MTE@1.0", "W2@1.0", "sW2@0.5", "sW2@1.0",
           "MMD@1.0", "E_mu", "E_Sigma"]


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


N_EVAL = 200  # number of points for E_mu/E_Sigma evaluation


def compute_coefficient_errors(ae, sde, surface, eval_uv, device):
    """Compute ambient drift and covariance coefficient errors at static points.

    Returns (E_mu_median, E_Sigma_median) — medians over evaluation points.
    E_mu = ||Dφ·μ_learned + ½q - b||²  (roundtrip drift error)
    E_Sigma = ||P·Λ·P - Λ||²_F         (tangent covariance error)
    """
    ae.eval()
    b = sde.ambient_drift(eval_uv.detach())
    Lambda = sde.ambient_covariance(eval_uv.detach())
    b, Lambda = b.detach(), Lambda.detach()

    with torch.no_grad():
        x_eval = sde.chart(eval_uv)
        z = ae.encoder(x_eval)

    dphi = torch.func.vmap(torch.func.jacrev(ae.decoder))(z)

    def decoder_hessian(z_single):
        return torch.func.jacrev(torch.func.jacrev(ae.decoder))(z_single)

    # Batch Hessian computation to avoid OOM at high D
    D = dphi.shape[1]
    hess_bs = 32 if D > 100 else len(z)
    d2phi_chunks = []
    for i in range(0, len(z), hess_bs):
        d2phi_chunks.append(torch.func.vmap(decoder_hessian)(z[i:i+hess_bs]).detach())
        torch.cuda.empty_cache()
    d2phi = torch.cat(d2phi_chunks)

    dphi = dphi.detach()
    d2phi = d2phi.detach()
    z = z.detach()

    with torch.no_grad():
        dphi_T = dphi.transpose(-1, -2)
        g = torch.bmm(dphi_T, dphi)
        g_inv = regularized_metric_inverse(g)
        P = torch.bmm(torch.bmm(dphi, g_inv), dphi_T)

        P_Lambda_P = torch.bmm(torch.bmm(P, Lambda), P)
        cov_err = ((P_Lambda_P - Lambda) ** 2).sum(dim=(-1, -2))

        Sigma_learned = torch.bmm(
            g_inv, torch.bmm(dphi_T, torch.bmm(Lambda, torch.bmm(dphi, g_inv)))
        )
        q = ambient_quadratic_variation_drift(Sigma_learned, d2phi)

        residual = b - 0.5 * q
        mu_learned = torch.bmm(
            g_inv, torch.bmm(dphi_T, residual.unsqueeze(-1))
        ).squeeze(-1)

        b_recon = torch.bmm(dphi, mu_learned.unsqueeze(-1)).squeeze(-1) + 0.5 * q
        drift_err = ((b_recon - b) ** 2).sum(dim=-1)

        valid = torch.isfinite(drift_err) & torch.isfinite(cov_err)
        if valid.sum() < 2:
            return float('nan'), float('nan')

        return drift_err[valid].median().item(), cov_err[valid].median().item()


# ── Phase 2 AE training ───────────────────────────────────────────────────

def _train_phase2(trainer, loader, mc, phase2_lw, phase2_epochs, phase1_converged):
    """Run Phase 2 fine-tuning with ramped loss weights.

    Returns final epoch loss.
    """
    if phase2_lw is None or not phase1_converged or phase2_epochs <= 0:
        if not phase1_converged:
            print(f"        Skipping Phase 2 (Phase 1 unconverged)")
        return None

    warmup_frac = 0.2
    warmup_epochs = max(1, int(phase2_epochs * warmup_frac))
    ep_losses = None
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
        ep_losses = trainer.train_epoch(loader, {mc.name: lw_epoch})
        if (epoch + 1) % max(1, phase2_epochs // 5) == 0:
            print(f"        Phase2 Epoch {epoch+1}/{phase2_epochs}: "
                  f"loss={ep_losses['ae']:.6f}")
    return ep_losses["ae"] if ep_losses else None


def _run_sde_stages(ae, x, v, Lambda, sde, surface, seed, n_train,
                    epochs_sde, lambda_smooth, drift_hidden=None):
    """Train Stage 2 + 3, evaluate full pipeline.

    Returns metrics dict.
    """
    batch_size = min(n_train, BATCH_SIZE)
    d = 2

    # Precompute decoder derivatives
    dummy = SDEPipelineTrainer(
        ae, DriftNet(d).to(DEVICE), DiffusionNet(d).to(DEVICE), device=DEVICE,
    )
    z_pre, dphi_pre, d2phi_pre = dummy.precompute_decoder_derivatives(x)

    # Precompute encoder-pullback drift target: b_z = Dπ·v + ½Λ:∇²π
    print("  Precomputing encoder-pullback target...")
    b_z_target, g = dummy.precompute_enc_pull_target(x, v, Lambda, dphi_pre)

    # Stage 2: Drift (encoder-pullback regression)
    torch.manual_seed(seed + 100)
    drift_net = DriftNet(d, hidden_dims=drift_hidden).to(DEVICE)
    dp = SDEPipelineTrainer(ae, drift_net, DiffusionNet(d).to(DEVICE), device=DEVICE)
    drift_losses = dp.train_stage2_regression(
        z_pre, b_z_target, g,
        epochs=epochs_sde, lr=LR_SDE, batch_size=batch_size,
        print_interval=max(1, epochs_sde // 5),
        lambda_smooth=lambda_smooth, aug_sigma=AUG_SIGMA,
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
    return {
        "drift_loss": drift_losses[-1],
        "diff_loss": diff_losses[-1],
        **eval_results,
    }


# ── Main experiment ─────────────────────────────────────────────────────────

def run_single_fork(surface_name, D, seed, n_train=20, epochs_ae=500,
                    epochs_sde=300, lambda_smooth=LAMBDA_SMOOTH):
    """Run shared-checkpoint fork for one (surface, D, seed, n_train).

    Phase 2 AEs are grouped: baseline (T+F) trained once, K (T+F+K) trained
    once and shared between "K" and "K+S" conditions (which differ only in
    Stage 2 smoothness).

    Returns dict mapping condition_label -> metrics dict.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)

    hdims = hidden_dims_for_D(D)
    surface = FourierAugmentedSurface(surface_name, D)
    batch_size = min(n_train, BATCH_SIZE)

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

    RECON_THRESHOLD = 0.1
    phase1_converged = recon_per_dim < RECON_THRESHOLD
    if not phase1_converged:
        print(f"    WARNING: Phase 1 recon_per_dim={recon_per_dim:.4f} > {RECON_THRESHOLD}")

    # Snapshot Phase 1 state
    phase1_state = copy.deepcopy(trainer.models["ae"].state_dict())
    phase1_optim_state = copy.deepcopy(trainer.optimizers["ae"].state_dict())
    phase1_sched_state = copy.deepcopy(trainer.schedulers["ae"].state_dict())

    # Lambdify SDE for evaluation
    sde = create_highd_lambdified_sde(surface, local_drift_fn, local_diffusion_fn)

    # ── Phase 2 groups ──
    # Group conditions by Phase 2 loss weights to avoid duplicate AE training.
    # baseline: Phase 2 with T+F (BASELINE_LW), SDE without smooth
    # K:        Phase 2 with T+F+K (full_lw),    SDE without smooth
    # K+S:      Phase 2 with T+F+K (full_lw),    SDE with smooth
    conditions = make_conditions(D, lambda_smooth)
    phase2_epochs = epochs_ae - phase1_epochs

    # Build grouped structure: {phase2_lw_id: (phase2_lw, [(label, lam_s)])}
    groups = {}
    for label, phase2_lw, lam_s in conditions:
        key = id(phase2_lw)
        if key not in groups:
            groups[key] = (phase2_lw, [])
        groups[key][1].append((label, lam_s))

    results = {}

    for phase2_lw, sde_conditions in groups.values():
        print(f"      Phase 2 AE: K_weight={phase2_lw.curvature:.4f}")

        # Create fresh trainer, load Phase 1 checkpoint
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

        # Chart-quality coefficient errors (shared across SDE conditions)
        torch.manual_seed(seed + 5000)
        eval_uv = (torch.rand(N_EVAL, 2, device=DEVICE) * 2 - 1) * TRAIN_BOUND
        e_mu, e_sigma = compute_coefficient_errors(ae, sde, surface, eval_uv, DEVICE)
        print(f"        E_mu={e_mu:.4f}  E_Sigma={e_sigma:.4f}")

        # Run SDE stages for each condition in this group
        for cond_label, lam_s in sde_conditions:
            smooth_tag = f" (smooth={lam_s})" if lam_s > 0 else ""
            print(f"      → {cond_label}{smooth_tag}")

            sde_results = _run_sde_stages(
                ae, x, v, Lambda, sde, surface, seed, n_train,
                epochs_sde, lam_s,
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
    cond_labels = sorted(df["condition"].unique())

    print(f"\n\n{'='*150}")
    print("N×D SWEEP: TRAINING SET SIZE × AMBIENT DIMENSION INTERACTION")
    print(f"{'='*150}")

    for D_val in sorted(df["D"].unique()):
        for N_val in sorted(df["N"].unique()):
            df_dn = df[(df["D"] == D_val) & (df["N"] == N_val)]
            if len(df_dn) == 0:
                continue

            print(f"\n{'─'*150}")
            print(f"  D = {D_val}, N = {N_val}")
            print(f"{'─'*150}")

            # Mean ± std table
            print(f"\n  {'surface':>25s}  {'condition':>10s}  {'n':>3s}  ", end="")
            for m in METRICS:
                if m in df_dn.columns:
                    print(f"{'mean':>8s} {'+-std':>8s}  ", end="")
            print()
            print("  " + "-" * 140)

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
                ("K+S vs K", "K+S", "K"),
            ]
            for comp_name, cond_a, cond_b in comparisons:
                print(f"\n  {comp_name}:")
                print(f"    {'surface':>25s}  ", end="")
                for m in METRICS:
                    if m in df_dn.columns:
                        print(f"{'delta%':>8s} {'p-val':>8s}  ", end="")
                print("  win-rate  verdict")
                print("    " + "-" * 138)

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
                    verdicts = []
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
                        if p_val < 0.05:
                            verdicts.append(f"{m}: {'worse' if delta > 0 else 'better'}")

                    # Win rate on W2@1.0
                    if "W2@1.0" in a.columns:
                        a_w2 = a["W2@1.0"].values
                        b_w2 = b["W2@1.0"].values
                        wins = np.sum(a_w2 < b_w2)
                        total = len(a_w2)
                        win_str = f"{wins}/{total}"
                    else:
                        win_str = "n/a"

                    verdict_str = "; ".join(verdicts) if verdicts else "n.s."
                    print(f"  {win_str:>8s}  {verdict_str}")


def main():
    parser = argparse.ArgumentParser(
        description="N×D Sweep: training set size × ambient dimension interaction",
    )
    parser.add_argument("--n-seeds", type=int, default=10)
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--sde-epochs", type=int, default=300)
    parser.add_argument("--base-seed", type=int, default=42)
    parser.add_argument("--d-values", type=int, nargs="+", default=None,
                        help="D values to test (default: 11 201)")
    parser.add_argument("--n-values", type=int, nargs="+", default=None,
                        help="N (training set size) values to test (default: 20 50 100 200)")
    parser.add_argument("--lambda-smooth", type=float, default=LAMBDA_SMOOTH,
                        help="Jacobian smoothness weight for Stage 2 (default: 0.5)")
    parser.add_argument("--output", type=str, default="highd_N_D_sweep.csv",
                        help="Output CSV path")
    args = parser.parse_args()

    if args.d_values is not None:
        d_configs = [((D - 3) // 2, D) for D in args.d_values]
    else:
        d_configs = D_CONFIGS

    n_values = args.n_values if args.n_values is not None else N_VALUES

    seeds = [args.base_seed + i * 1000 for i in range(args.n_seeds)]
    conditions = make_conditions(11, args.lambda_smooth)

    print(f"Device: {DEVICE}")
    print(f"Seeds ({len(seeds)}): {seeds}")
    print(f"Surfaces: {SURFACES}")
    print(f"Conditions: {[c[0] for c in conditions]}")
    print(f"D configs: {d_configs}")
    print(f"N values: {n_values}")
    print(f"K weight: {BASE_K_WEIGHT} * sqrt(D/{D_REF})")
    print(f"Smooth: lambda={args.lambda_smooth}, aug_sigma={AUG_SIGMA}")
    print(f"Hidden dims: D-dependent (<=11:[64,64], <=51:[128,128], else:[256,256])")
    print(f"AE epochs: {args.epochs} (Phase 1: {args.epochs // 2}, "
          f"Phase 2: {args.epochs - args.epochs // 2})")
    print(f"SDE epochs: {args.sde_epochs}")
    expected_rows = (len(n_values) * len(d_configs) * len(SURFACES)
                     * len(conditions) * len(seeds))
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

                        print(f"    {cond_label:>10s}: W2={metrics['W2@1.0']:.4f}  "
                              f"E_mu={metrics['E_mu']:.4f}  E_Sig={metrics['E_Sigma']:.4f}")

            elapsed_dn = time.time() - t_dn
            print(f"\n  N={N}, D={D_val} completed in {elapsed_dn:.1f}s "
                  f"({elapsed_dn / (len(SURFACES) * len(seeds)):.1f}s per fork)")

    df = pd.DataFrame(all_rows)
    csv_path = args.output
    df.to_csv(csv_path, index=False)
    print(f"\nSaved {len(df)} rows to {csv_path}")

    print_summary(df)

    print(f"\nTotal time: {time.time() - t0:.1f}s")
    print("Done.")


if __name__ == "__main__":
    main()
