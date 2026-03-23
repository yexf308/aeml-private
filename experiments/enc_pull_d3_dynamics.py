"""
Encoder-pullback drift fitting on D=3 surfaces for the paper.

Compares three conditions on 4 surfaces with 10 seeds:
  1. baseline:   AE with no geometric penalties + encoder-pullback drift
  2. T+F:        AE with tangent+diffeo + decoder-side drift (Lemma 2.3)
  3. T+F (enc):  AE with tangent+diffeo + encoder-pullback drift (new method)

For each (surface, seed):
  - Train a T+F AE (shared between decoder-side and encoder-pullback)
  - Train a baseline AE (no penalties)
  - Fork Stage 2 drift into 3 conditions
  - Share Stage 3 diffusion per AE
  - Evaluate W2, MMD, MTE at t=1.0, and E_mu coefficient error

Usage:
    # Smoke test
    python -m experiments.enc_pull_d3_dynamics --n-seeds 1 --ae-epochs 50 --sde-epochs 50

    # Full run
    python -m experiments.enc_pull_d3_dynamics --n-seeds 10

    # GPU
    srun --partition=dgx --gres=gpu:1 --time=04:00:00 \
      python3 -u -m experiments.enc_pull_d3_dynamics --n-seeds 10
"""

import argparse
import copy
import time
import torch
import numpy as np
import pandas as pd
import sympy as sp

from src.numeric.datagen import sample_from_manifold
from src.numeric.losses import LossWeights
from src.numeric.sde_nets import DriftNet, DiffusionNet
from src.numeric.sde_training import SDEPipelineTrainer
from src.numeric.training import MultiModelTrainer, TrainingConfig
from src.numeric.geometry import (
    regularized_metric_inverse,
    ambient_quadratic_variation_drift,
)
from src.symbolic.manifold_sdes import ManifoldSDE
from src.symbolic.riemannian import RiemannianManifold
from src.symbolic.surfaces import surface

from experiments.common import SURFACE_MAP, make_model_config
from experiments.data_driven_sde import (
    TRAIN_BOUND, BATCH_SIZE, LR_AE, LR_SDE, DEVICE,
    N_TRAJ, N_STEPS, DT, BOUNDARY,
    evaluate_pipeline,
)
from experiments.trajectory_fidelity_study import (
    lambdify_sde,
    simulate_ground_truth,
    compute_mte_at_step,
    compute_w2,
    compute_mmd,
)

# ──────────────────────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────────────────────

SURFACES = ["paraboloid", "hyperbolic_paraboloid", "quartic_dome", "sinusoidal"]
N_TRAIN = 20
STOP_BOUND = 1.0
N_EVAL = 200  # points for E_mu evaluation

# Drift smoothness regularisation
LAMBDA_SMOOTH = 0.5
AUG_SIGMA = 0.1

# Condition labels in display order
CONDITION_LABELS = ["baseline", "T+F (dec)", "T+F (enc)"]

# AE loss configs
BASELINE_LW = LossWeights()  # no geometric penalties
TF_LW = LossWeights(tangent_bundle=1.0, diffeo=1.0)


# ──────────────────────────────────────────────────────────────
# Manifold SDE factory
# ──────────────────────────────────────────────────────────────

def create_manifold_sde(surface_name):
    """Create manifold SDE with non-trivial (state-dependent) dynamics."""
    u, v = sp.symbols("u v", real=True)
    local_coord, chart = surface(SURFACE_MAP[surface_name], u, v)
    manifold = RiemannianManifold(local_coord, chart)
    local_drift = sp.Matrix([-v, u])
    local_diffusion = sp.Matrix([[1 + u**2 / 4, u + v], [0, 1 + v**2 / 4]])
    return ManifoldSDE(manifold, local_drift=local_drift, local_diffusion=local_diffusion)


# ──────────────────────────────────────────────────────────────
# AE training
# ──────────────────────────────────────────────────────────────

def train_autoencoder(train_data, loss_weights, reg_name, epochs, seed):
    """Train an autoencoder with given regularization weights."""
    torch.manual_seed(seed)
    np.random.seed(seed)

    print(f"\n  Stage 1: AE ({reg_name}, {epochs} epochs)")
    trainer = MultiModelTrainer(TrainingConfig(
        epochs=epochs,
        n_samples=N_TRAIN,
        input_dim=3,
        hidden_dim=64,
        latent_dim=2,
        learning_rate=LR_AE,
        batch_size=BATCH_SIZE,
        test_size=0.03,
        print_interval=max(1, epochs // 5),
        device=DEVICE,
    ))
    trainer.add_model(make_model_config("ae", loss_weights, hidden_dims=[64, 64]))
    loader = trainer.create_data_loader(train_data)
    for epoch in range(epochs):
        losses = trainer.train_epoch(loader)
        if (epoch + 1) % max(1, epochs // 5) == 0:
            print(f"    Epoch {epoch+1}: loss={losses['ae']:.6f}")

    ae = trainer.models["ae"]
    ae.eval()
    return ae


# ──────────────────────────────────────────────────────────────
# Coefficient error (E_mu) at static points
# ──────────────────────────────────────────────────────────────

def compute_coefficient_errors(ae, sde, eval_uv):
    """Compute ambient drift/covariance coefficient errors at static points.

    Returns (E_mu_median, E_Sigma_median) -- medians over evaluation points.
    E_mu   = ||Dphi . mu_learned + 0.5*q - b||^2   (roundtrip drift error)
    E_Sigma = ||P . Lambda . P - Lambda||^2_F        (tangent covariance error)
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

    d2phi = torch.func.vmap(decoder_hessian)(z)

    dphi = dphi.detach()
    d2phi = d2phi.detach()
    z = z.detach()

    with torch.no_grad():
        dphi_T = dphi.transpose(-1, -2)
        g = torch.bmm(dphi_T, dphi)
        g_inv = regularized_metric_inverse(g)
        P = torch.bmm(torch.bmm(dphi, g_inv), dphi_T)

        # Covariance error
        P_Lambda_P = torch.bmm(torch.bmm(P, Lambda), P)
        cov_err = ((P_Lambda_P - Lambda) ** 2).sum(dim=(-1, -2))

        # Drift error
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


# ──────────────────────────────────────────────────────────────
# Single (surface, seed) run
# ──────────────────────────────────────────────────────────────

def run_single(surface_name, seed, ae_epochs, sde_epochs, sde):
    """Run all 3 conditions for one (surface, seed).

    Returns list of row dicts (one per condition).
    """
    torch.manual_seed(seed)
    np.random.seed(seed)

    manifold_sde = create_manifold_sde(surface_name)

    # ── Sample training data ──
    train_data = sample_from_manifold(
        manifold_sde,
        [(-TRAIN_BOUND, TRAIN_BOUND), (-TRAIN_BOUND, TRAIN_BOUND)],
        n_samples=N_TRAIN, seed=seed, device=DEVICE,
    )
    x = train_data.samples.to(DEVICE)
    v = train_data.mu.to(DEVICE)
    Lambda = train_data.cov.to(DEVICE)

    d = 2
    batch_size = min(N_TRAIN, BATCH_SIZE)

    # ── Evaluation points for E_mu (shared across conditions) ──
    torch.manual_seed(seed + 5000)
    eval_uv = (torch.rand(N_EVAL, 2, device=DEVICE) * 2 - 1) * TRAIN_BOUND

    # ================================================================
    # Train T+F AE (shared between decoder-side and encoder-pullback)
    # ================================================================
    ae_tf = train_autoencoder(train_data, TF_LW, "T+F", ae_epochs, seed)

    # Precompute decoder derivatives for T+F AE
    print("  Precomputing decoder derivatives (T+F)...")
    dummy = SDEPipelineTrainer(
        ae_tf, DriftNet(d).to(DEVICE), DiffusionNet(d).to(DEVICE), device=DEVICE,
    )
    z_pre_tf, dphi_pre_tf, d2phi_pre_tf = dummy.precompute_decoder_derivatives(x)

    # Precompute encoder-pullback target
    print("  Precomputing encoder-pullback target (T+F)...")
    b_z_enc_target_tf, g_tf = dummy.precompute_enc_pull_target(
        x, v, Lambda, dphi_pre_tf,
    )

    # Stage 3 diffusion (shared for T+F AE conditions)
    torch.manual_seed(seed + 200)
    diff_net_tf = DiffusionNet(d).to(DEVICE)
    diff_pipeline_tf = SDEPipelineTrainer(
        ae_tf, DriftNet(d).to(DEVICE), diff_net_tf, device=DEVICE,
    )
    print(f"  Stage 3: Diffusion net (T+F AE, {sde_epochs} epochs)")
    diff_pipeline_tf.train_stage3_precomputed(
        z_pre_tf, dphi_pre_tf, Lambda,
        epochs=sde_epochs, lr=LR_SDE,
        batch_size=batch_size, print_interval=max(1, sde_epochs // 5),
    )
    diff_net_tf.eval()
    diff_state_tf = copy.deepcopy(diff_net_tf.state_dict())

    # E_mu for T+F AE
    e_mu_tf, e_sigma_tf = compute_coefficient_errors(ae_tf, sde, eval_uv)

    # ================================================================
    # Train baseline AE (no penalties)
    # ================================================================
    ae_bl = train_autoencoder(train_data, BASELINE_LW, "baseline", ae_epochs, seed)

    # Precompute decoder derivatives for baseline AE
    print("  Precomputing decoder derivatives (baseline)...")
    dummy_bl = SDEPipelineTrainer(
        ae_bl, DriftNet(d).to(DEVICE), DiffusionNet(d).to(DEVICE), device=DEVICE,
    )
    z_pre_bl, dphi_pre_bl, _ = dummy_bl.precompute_decoder_derivatives(x)

    # Precompute encoder-pullback target for baseline AE
    print("  Precomputing encoder-pullback target (baseline)...")
    b_z_enc_target_bl, g_bl = dummy_bl.precompute_enc_pull_target(
        x, v, Lambda, dphi_pre_bl,
    )

    # Stage 3 diffusion (baseline AE)
    torch.manual_seed(seed + 200)
    diff_net_bl = DiffusionNet(d).to(DEVICE)
    diff_pipeline_bl = SDEPipelineTrainer(
        ae_bl, DriftNet(d).to(DEVICE), diff_net_bl, device=DEVICE,
    )
    print(f"  Stage 3: Diffusion net (baseline AE, {sde_epochs} epochs)")
    diff_pipeline_bl.train_stage3_precomputed(
        z_pre_bl, dphi_pre_bl, Lambda,
        epochs=sde_epochs, lr=LR_SDE,
        batch_size=batch_size, print_interval=max(1, sde_epochs // 5),
    )
    diff_net_bl.eval()
    diff_state_bl = copy.deepcopy(diff_net_bl.state_dict())

    # E_mu for baseline AE
    e_mu_bl, e_sigma_bl = compute_coefficient_errors(ae_bl, sde, eval_uv)

    # ================================================================
    # Fork Stage 2: 3 drift conditions
    # ================================================================
    rows = []

    conditions = [
        # (label, ae, z_pre, dphi_pre, d2phi_pre, enc_target, g, diff_state, e_mu, e_sigma, drift_method)
        ("baseline",   ae_bl, z_pre_bl, dphi_pre_bl, None, b_z_enc_target_bl, g_bl,
         diff_state_bl, e_mu_bl, e_sigma_bl, "enc_pull"),
        ("T+F (dec)",  ae_tf, z_pre_tf, dphi_pre_tf, d2phi_pre_tf, None, g_tf,
         diff_state_tf, e_mu_tf, e_sigma_tf, "decoder"),
        ("T+F (enc)",  ae_tf, z_pre_tf, dphi_pre_tf, None, b_z_enc_target_tf, g_tf,
         diff_state_tf, e_mu_tf, e_sigma_tf, "enc_pull"),
    ]

    for (cond_label, ae, z_pre, dphi_pre, d2phi_pre,
         enc_target, g, d_state, e_mu, e_sigma, drift_method) in conditions:
        t0 = time.time()
        print(f"\n  --- Condition: {cond_label} (drift={drift_method}) ---")

        torch.manual_seed(seed + 100)
        drift_net = DriftNet(d).to(DEVICE)

        if drift_method == "enc_pull":
            pipeline = SDEPipelineTrainer(
                ae, drift_net, DiffusionNet(d).to(DEVICE), device=DEVICE,
            )
            print(f"    Stage 2: Drift net (encoder pullback, {sde_epochs} epochs)")
            drift_losses = pipeline.train_stage2_regression(
                z_pre, enc_target, g,
                epochs=sde_epochs, lr=LR_SDE,
                batch_size=batch_size, print_interval=max(1, sde_epochs // 5),
                lambda_smooth=LAMBDA_SMOOTH, aug_sigma=AUG_SIGMA,
            )
        elif drift_method == "decoder":
            pipeline = SDEPipelineTrainer(
                ae, drift_net, DiffusionNet(d).to(DEVICE), device=DEVICE,
            )
            print(f"    Stage 2: Drift net (decoder-side, {sde_epochs} epochs)")
            drift_losses = pipeline.train_stage2_precomputed(
                z_pre, dphi_pre, d2phi_pre, v, Lambda,
                epochs=sde_epochs, lr=LR_SDE,
                batch_size=batch_size, print_interval=max(1, sde_epochs // 5),
                lambda_smooth=LAMBDA_SMOOTH, aug_sigma=AUG_SIGMA,
            )

        drift_net.eval()

        # Assemble final pipeline with shared diffusion
        diff_net_copy = DiffusionNet(d).to(DEVICE)
        diff_net_copy.load_state_dict(d_state)
        diff_net_copy.eval()
        pipeline = SDEPipelineTrainer(ae, drift_net, diff_net_copy, device=DEVICE)

        # Evaluate trajectory fidelity
        print("    Evaluating trajectory fidelity...")
        eval_res = evaluate_pipeline(pipeline, ae, sde, seed)
        elapsed = time.time() - t0

        w2 = eval_res.get("W2@1.0", float("nan"))
        mte = eval_res.get("MTE@1.0", float("nan"))
        mmd = eval_res.get("MMD@1.0", float("nan"))
        print(f"    {cond_label}: W2={w2:.4f}  MTE={mte:.4f}  MMD={mmd:.4f}  "
              f"E_mu={e_mu:.6f}  ({elapsed:.0f}s)")

        rows.append({
            "surface": surface_name,
            "seed": seed,
            "condition": cond_label,
            "E_mu": e_mu,
            "E_Sigma": e_sigma,
            "drift_loss": drift_losses[-1],
            **eval_res,
        })

    return rows


# ──────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Encoder-pullback drift fitting on D=3 surfaces",
    )
    parser.add_argument("--n-seeds", type=int, default=10)
    parser.add_argument("--ae-epochs", type=int, default=500)
    parser.add_argument("--sde-epochs", type=int, default=500)
    parser.add_argument("--base-seed", type=int, default=42)
    parser.add_argument("--surfaces", type=str, nargs="+",
                        default=SURFACES,
                        choices=list(SURFACE_MAP.keys()))
    parser.add_argument("--output", type=str, default="enc_pull_d3_dynamics.csv")
    args = parser.parse_args()

    seeds = [args.base_seed + i * 1000 for i in range(args.n_seeds)]

    print(f"Device: {DEVICE}")
    print(f"Surfaces: {args.surfaces}")
    print(f"Seeds ({len(seeds)}): {seeds}")
    print(f"AE epochs: {args.ae_epochs}, SDE epochs: {args.sde_epochs}")
    print(f"N_TRAIN={N_TRAIN}, N_TRAJ={N_TRAJ}, T_MAX={N_STEPS * DT:.1f}, DT={DT}")

    t_total = time.time()
    all_rows = []

    for surf in args.surfaces:
        print(f"\n{'='*70}")
        print(f"Surface: {surf}")
        print(f"{'='*70}")

        # Lambdify SDE once per surface (expensive symbolic computation)
        print("  Lambdifying SDE...")
        manifold_sde = create_manifold_sde(surf)
        sde = lambdify_sde(manifold_sde)

        for seed in seeds:
            print(f"\n{'='*60}")
            print(f"  {surf} | seed={seed}")
            print(f"{'='*60}")
            rows = run_single(surf, seed, args.ae_epochs, args.sde_epochs, sde)
            all_rows.extend(rows)

    df = pd.DataFrame(all_rows)
    df.to_csv(args.output, index=False)
    print(f"\nSaved {len(df)} rows to {args.output}")

    # ──────────────────────────────────────────────────────────────
    # Summary table
    # ──────────────────────────────────────────────────────────────
    print(f"\n{'='*90}")
    print("SUMMARY (median over seeds)")
    print(f"{'='*90}")

    key_metrics = ["W2@1.0", "MMD@1.0", "MTE@1.0", "E_mu"]

    for surf in args.surfaces:
        sub = df[df.surface == surf]
        if sub.empty:
            continue
        print(f"\n  {surf}:")
        header = f"  {'condition':>12s}"
        for m in key_metrics:
            header += f"  {m:>10s}"
        print(header)
        print(f"  {'-'*12}" + f"  {'-'*10}" * len(key_metrics))

        for cond in CONDITION_LABELS:
            sc = sub[sub.condition == cond]
            if sc.empty:
                continue
            line = f"  {cond:>12s}"
            for m in key_metrics:
                if m in sc.columns:
                    val = sc[m].median()
                    line += f"  {val:10.4f}"
                else:
                    line += f"  {'nan':>10s}"
            print(line)

    # ──────────────────────────────────────────────────────────────
    # Paired comparisons: T+F (enc) vs T+F (dec)
    # ──────────────────────────────────────────────────────────────
    if len(seeds) >= 3:
        print(f"\n{'='*90}")
        print("PAIRED COMPARISONS: T+F (enc) vs T+F (dec)  [Wilcoxon signed-rank]")
        print(f"{'='*90}")

        from scipy.stats import wilcoxon

        for surf in args.surfaces:
            sub = df[df.surface == surf]
            enc = sub[sub.condition == "T+F (enc)"].sort_values("seed")
            dec = sub[sub.condition == "T+F (dec)"].sort_values("seed")

            if len(enc) != len(dec) or len(enc) < 3:
                continue

            print(f"\n  {surf}:")
            for m in key_metrics:
                if m not in enc.columns or m not in dec.columns:
                    continue
                vals_enc = enc[m].values
                vals_dec = dec[m].values
                valid = np.isfinite(vals_enc) & np.isfinite(vals_dec)
                if valid.sum() < 3:
                    continue
                med_enc = np.median(vals_enc[valid])
                med_dec = np.median(vals_dec[valid])
                delta_pct = (med_enc - med_dec) / (abs(med_dec) + 1e-15) * 100
                try:
                    _, p = wilcoxon(vals_enc[valid], vals_dec[valid])
                except ValueError:
                    p = float("nan")
                sign = "-" if delta_pct < 0 else "+"
                print(f"    {m:>10s}: enc={med_enc:.4f}  dec={med_dec:.4f}  "
                      f"delta={sign}{abs(delta_pct):.1f}%  p={p:.3f}")

    print(f"\nTotal time: {time.time() - t_total:.1f}s")
    print("Done.")


if __name__ == "__main__":
    main()
