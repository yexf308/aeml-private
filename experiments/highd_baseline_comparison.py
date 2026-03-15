"""
High-D baseline comparison: unconstrained AE vs T+F AE with encoder-pullback drift.

Tests whether T+F geometric penalties improve trajectory quality over an
unconstrained autoencoder at higher ambient dimensions (D=11, D=201).

Design:
  Two conditions per (surface, D, N, seed):
    - "no_penalty": AE trained with reconstruction loss only, enc-pull drift
    - "T+F":        AE trained with T+F penalties, enc-pull drift

  Both use encoder-pullback drift + smoothness regularisation.

Usage:
    # Smoke test
    python -m experiments.highd_baseline_comparison --n-seeds 1 --epochs 50 --sde-epochs 50 --d-values 11 --n-values 20

    # Full GPU run
    srun --partition=dgx --gres=gpu:1 --time=06:00:00 \
      python3 -u -m experiments.highd_baseline_comparison --n-seeds 10
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
)

# Reuse helpers from highd_N_D_sweep
from experiments.highd_N_D_sweep import (
    D_CONFIGS, SURFACES, N_VALUES, N_EVAL,
    hidden_dims_for_D, local_drift_fn, local_diffusion_fn,
    compute_coefficient_errors, _run_sde_stages,
)

# ── Constants ──────────────────────────────────────────────────────────────

NO_PENALTY_LW = LossWeights()  # reconstruction only
TF_LW = LossWeights(tangent_bundle=1.0, diffeo=1.0)

LAMBDA_SMOOTH = 0.5
AUG_SIGMA = 0.1

COND_LABELS = ["no_penalty", "T+F"]
METRICS = ["E_mu", "E_Sigma", "W2@1.0", "MMD@1.0", "MTE@1.0"]


# ── Training helpers ───────────────────────────────────────────────────────

def _train_ae(surface, D, seed, n_train, epochs, loss_weights, train_data):
    """Train an AE with given loss weights for the full epoch budget."""
    hdims = hidden_dims_for_D(D)
    batch_size = min(n_train, BATCH_SIZE)

    trainer = MultiModelTrainer(TrainingConfig(
        epochs=epochs, n_samples=n_train, input_dim=D, hidden_dim=hdims[0],
        latent_dim=2, learning_rate=LR_AE, batch_size=batch_size,
        test_size=0.03, print_interval=max(1, epochs // 5), device=DEVICE,
    ))
    mc = make_model_config("ae", loss_weights, extrinsic_dim=D, hidden_dims=hdims)
    trainer.add_model(mc)
    loader = trainer.create_data_loader(train_data)

    for epoch in range(epochs):
        losses = trainer.train_epoch(loader, {mc.name: loss_weights})
        if (epoch + 1) % max(1, epochs // 5) == 0:
            print(f"      Epoch {epoch+1}/{epochs}: loss={losses['ae']:.6f}")

    ae = trainer.models["ae"]
    ae.eval()

    # Recon quality check
    x = train_data.samples.to(DEVICE)
    with torch.no_grad():
        z = ae.encoder(x)
        x_hat = ae.decoder(z)
        recon_per_dim = ((x_hat - x) ** 2).sum(-1).mean().item() / D

    return ae, recon_per_dim


def run_single(surface_name, D, seed, n_train, epochs_ae, epochs_sde):
    """Run both conditions for one (surface, D, N, seed).

    Returns dict mapping condition_label -> metrics dict.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)

    surface = FourierAugmentedSurface(surface_name, D)

    train_data = sample_from_highd_manifold(
        surface, local_drift_fn, local_diffusion_fn,
        [(-TRAIN_BOUND, TRAIN_BOUND), (-TRAIN_BOUND, TRAIN_BOUND)],
        n_samples=n_train, seed=seed, device=DEVICE,
    )
    x = train_data.samples.to(DEVICE)
    v = train_data.mu.to(DEVICE)
    Lambda = train_data.cov.to(DEVICE)

    sde = create_highd_lambdified_sde(surface, local_drift_fn, local_diffusion_fn)

    results = {}

    for cond_label, lw in [("no_penalty", NO_PENALTY_LW), ("T+F", TF_LW)]:
        print(f"\n    Condition: {cond_label}")

        # Reset RNG for AE training so each condition gets the same init
        torch.manual_seed(seed)
        np.random.seed(seed)

        ae, recon_per_dim = _train_ae(
            surface, D, seed, n_train, epochs_ae, lw, train_data,
        )
        print(f"      recon_per_dim={recon_per_dim:.6f}")

        # Coefficient errors
        torch.manual_seed(seed + 5000)
        eval_uv = (torch.rand(N_EVAL, 2, device=DEVICE) * 2 - 1) * TRAIN_BOUND
        e_mu, e_sigma = compute_coefficient_errors(ae, sde, surface, eval_uv, DEVICE)
        print(f"      E_mu={e_mu:.4f}  E_Sigma={e_sigma:.4f}")

        # SDE stages (encoder-pullback + smoothness)
        sde_results = _run_sde_stages(
            ae, x, v, Lambda, sde, surface, seed, n_train,
            epochs_sde, LAMBDA_SMOOTH,
        )

        results[cond_label] = {
            "ae_loss": 0.0,
            "recon_per_dim": recon_per_dim,
            "E_mu": e_mu,
            "E_Sigma": e_sigma,
            "drift_loss": sde_results["drift_loss"],
            "diff_loss": sde_results["diff_loss"],
            **{k: v for k, v in sde_results.items()
               if k not in ("drift_loss", "diff_loss")},
        }

    return results


# ── Summary ────────────────────────────────────────────────────────────────

def paired_ttest(vals_a, vals_b):
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
    print(f"\n\n{'='*120}")
    print("HIGH-D BASELINE COMPARISON: no_penalty vs T+F (both with enc-pull drift)")
    print(f"{'='*120}")

    for D_val in sorted(df["D"].unique()):
        for N_val in sorted(df["N"].unique()):
            df_dn = df[(df["D"] == D_val) & (df["N"] == N_val)]
            if len(df_dn) == 0:
                continue

            print(f"\n{'─'*120}")
            print(f"  D = {D_val}, N = {N_val}")
            print(f"{'─'*120}")

            # Mean ± std
            print(f"\n  {'surface':>25s}  {'condition':>12s}  {'n':>3s}", end="")
            for m in METRICS:
                if m in df_dn.columns:
                    print(f"  {m:>12s}", end="")
            print()

            for surface_name in SURFACES:
                for cond in COND_LABELS:
                    sub = df_dn[(df_dn["surface"] == surface_name) & (df_dn["condition"] == cond)]
                    n = len(sub)
                    print(f"  {surface_name:>25s}  {cond:>12s}  {n:>3d}", end="")
                    for m in METRICS:
                        if m in sub.columns:
                            vals = sub[m].values
                            print(f"  {np.nanmean(vals):>5.3f}±{np.nanstd(vals):>.3f}", end="")
                    print()

            # Paired comparison: T+F vs no_penalty
            print(f"\n  T+F vs no_penalty (negative = T+F is better):")
            print(f"    {'surface':>25s}", end="")
            for m in METRICS:
                print(f"  {'Δ%':>6s} {'p':>6s}", end="")
            print("  win-rate")

            for surface_name in SURFACES:
                a = df_dn[(df_dn["surface"] == surface_name) & (df_dn["condition"] == "T+F")].sort_values("seed")
                b = df_dn[(df_dn["surface"] == surface_name) & (df_dn["condition"] == "no_penalty")].sort_values("seed")
                if len(a) == 0 or len(b) == 0:
                    continue

                print(f"    {surface_name:>25s}", end="")
                for m in METRICS:
                    if m not in a.columns:
                        continue
                    av, bv = a[m].values, b[m].values
                    mask = np.isfinite(av) & np.isfinite(bv)
                    if mask.sum() < 2:
                        print(f"  {'n/a':>6s} {'n/a':>6s}", end="")
                        continue
                    ac, bc = av[mask], bv[mask]
                    delta = (ac.mean() - bc.mean()) / bc.mean() * 100
                    _, p_val = paired_ttest(ac, bc)
                    sig = "**" if p_val < 0.01 else "*" if p_val < 0.05 else "+" if p_val < 0.1 else ""
                    print(f"  {delta:>+5.1f}% {p_val:>.3f}{sig}", end="")

                # Win rate on W2
                if "W2@1.0" in a.columns:
                    wins = np.sum(a["W2@1.0"].values < b["W2@1.0"].values)
                    print(f"  {wins}/{len(a)}", end="")
                print()


# ── CLI ────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="High-D baseline comparison: no_penalty vs T+F",
    )
    parser.add_argument("--n-seeds", type=int, default=10)
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--sde-epochs", type=int, default=300)
    parser.add_argument("--base-seed", type=int, default=42)
    parser.add_argument("--d-values", type=int, nargs="+", default=None)
    parser.add_argument("--n-values", type=int, nargs="+", default=None)
    parser.add_argument("--output", type=str, default="highd_baseline_comparison.csv")
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
    print(f"Smooth: lambda={LAMBDA_SMOOTH}, aug_sigma={AUG_SIGMA}")
    expected = len(n_values) * len(d_configs) * len(SURFACES) * 2 * len(seeds)
    print(f"Expected rows: {expected}\n")

    t0 = time.time()
    all_rows = []

    for N in n_values:
        for K_pairs, D_val in d_configs:
            for surface_name in SURFACES:
                for seed in seeds:
                    print(f"\n{'='*60}")
                    print(f"  N={N} | D={D_val} | {surface_name} | seed={seed}")
                    print(f"{'='*60}")

                    results = run_single(
                        surface_name, D_val, seed, n_train=N,
                        epochs_ae=args.epochs, epochs_sde=args.sde_epochs,
                    )

                    for cond_label, metrics in results.items():
                        all_rows.append({
                            "N": N,
                            "D": D_val,
                            "K_pairs": K_pairs,
                            "surface": surface_name,
                            "condition": cond_label,
                            "seed": seed,
                            **metrics,
                        })

                    # Save incrementally
                    pd.DataFrame(all_rows).to_csv(args.output, index=False)

    elapsed = time.time() - t0
    print(f"\n\nTotal time: {elapsed / 60:.1f} min")

    df = pd.DataFrame(all_rows)
    df.to_csv(args.output, index=False)
    print(f"Saved {len(df)} rows to {args.output}")

    print_summary(df)


if __name__ == "__main__":
    main()
