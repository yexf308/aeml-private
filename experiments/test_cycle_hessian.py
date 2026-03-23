#!/usr/bin/env python3
"""Test cycle_hessian regularization effect on D²(π∘φ) and enc_pull_v2.

Trains AE with and without L_H = ||D²(π∘φ)||²_F regularization, then measures:
  1. ||D²(π∘φ)|| at training points (should decrease with L_H)
  2. enc_pull vs enc_pull_v2 target agreement (should improve with L_H)
  3. W2 for enc_pull and enc_pull_v2 conditions (v2 should catch up)

Usage:
    # Smoke test
    python3 -u -m experiments.test_cycle_hessian --n-seeds 1 --ae-epochs 50 --sde-epochs 50

    # Full
    python3 -u -m experiments.test_cycle_hessian --n-seeds 5
"""
import argparse
import copy
import time
import torch
import numpy as np
import pandas as pd

from src.numeric.losses import LossWeights, cycle_hessian_penalty
from src.numeric.sde_nets import DriftNet, DiffusionNet
from src.numeric.sde_training import SDEPipelineTrainer
from src.numeric.training import MultiModelTrainer, TrainingConfig
from src.numeric.geometry import (
    regularized_metric_inverse,
    curvature_drift_explicit_full,
)
from src.numeric.highd_manifolds import (
    FourierAugmentedSurface,
    sample_from_highd_manifold,
    create_highd_lambdified_sde,
)

from experiments.common import make_model_config
from experiments.data_driven_sde import (
    TRAIN_BOUND, BATCH_SIZE, LR_AE, LR_SDE, DEVICE,
)
from experiments.highd_N_D_sweep import (
    WARMUP_LW, BASE_K_WEIGHT, D_REF,
    LAMBDA_SMOOTH, AUG_SIGMA, N_EVAL,
    local_drift_fn, local_diffusion_fn,
    hidden_dims_for_D, compute_coefficient_errors,
    _train_phase2, evaluate_pipeline,
)


SURFACES = ["paraboloid", "hyperbolic_paraboloid"]
# L_H weights to test: 0 (baseline) and several strengths
LH_WEIGHTS = [0.0, 0.01, 0.05, 0.1]
# Drift conditions to test for each AE
DRIFT_CONDITIONS = ["current", "enc_pull", "enc_pull_v2"]


def measure_d2_pi_phi(ae, z):
    """Compute ||D²(π∘φ)(z)||_F at each point."""
    def _g(z_single):
        x_hat = ae.decoder(z_single.unsqueeze(0)).squeeze(0)
        return ae.encoder(x_hat.unsqueeze(0)).squeeze(0)

    D2g = torch.func.vmap(torch.func.hessian(_g))(z).detach()  # (N, d, d, d)
    return D2g.flatten(1).norm(dim=1)  # (N,)


def compute_enc_pull_targets(ae, x, v, Lambda, dphi, d2phi, dpi):
    """Compute enc_pull and enc_pull_v2 targets for comparison."""
    N = x.shape[0]

    # enc_pull: Dπ·v + ½Λ:∇²π
    ito_corrections = []
    for i in range(0, N, 8):
        x_b = x[i:i+8]
        Lambda_b = Lambda[i:i+8]
        d2pi_b = ae.encoder_hessian(x_b).detach()
        corr_b = 0.5 * torch.einsum('brs, birs -> bi', Lambda_b, d2pi_b)
        ito_corrections.append(corr_b)
    ito_correction = torch.cat(ito_corrections)
    b_z_enc = (dpi @ v.unsqueeze(-1)).squeeze(-1) + ito_correction

    # enc_pull_v2: Dπ·(v - q_φ) where q_φ = ½(pinv·Λ·pinv^T):d²φ
    g = dphi.mT @ dphi
    ginv = regularized_metric_inverse(g)
    pinv = ginv @ dphi.mT
    Sigma_z = pinv @ Lambda @ pinv.mT
    Sigma_z = 0.5 * (Sigma_z + Sigma_z.mT)
    q_phi = curvature_drift_explicit_full(d2phi, Sigma_z)
    b_z_v2 = (dpi @ (v - q_phi).unsqueeze(-1)).squeeze(-1)

    return b_z_enc.detach(), b_z_v2.detach()


def run_single(surface_name, D, N, seed, ae_epochs, sde_epochs, lh_weight):
    """Run one (surface, seed, lh_weight) combination."""
    torch.manual_seed(seed)
    np.random.seed(seed)

    surface = FourierAugmentedSurface(surface_name, D)
    sde = create_highd_lambdified_sde(surface, local_drift_fn, local_diffusion_fn)
    k_weight = BASE_K_WEIGHT * (D / D_REF) ** 0.5
    # Phase 2 LW includes cycle_hessian
    full_lw = LossWeights(
        tangent_bundle=1.0, diffeo=1.0, curvature=k_weight,
        cycle_hessian=lh_weight,
    )
    hdims = hidden_dims_for_D(D)
    batch_size = min(N, BATCH_SIZE)
    d = 2

    # ── Sample data ──
    train_data = sample_from_highd_manifold(
        surface, local_drift_fn, local_diffusion_fn,
        [(-TRAIN_BOUND, TRAIN_BOUND), (-TRAIN_BOUND, TRAIN_BOUND)],
        n_samples=N, seed=seed, device=DEVICE,
    )
    x = train_data.samples.to(DEVICE)
    v = train_data.mu.to(DEVICE)
    Lambda = train_data.cov.to(DEVICE)

    # ── Stage 1: AE training (3-phase) ──
    # Phase 1: warmup T+F only
    # Phase 2: T+F+K (standard two-phase)
    # Phase 3: T+F+K+L_H (if lh_weight > 0)
    phase1_epochs = ae_epochs // 3
    phase2_epochs = ae_epochs // 3
    phase3_epochs = ae_epochs - phase1_epochs - phase2_epochs

    tc = TrainingConfig(
        epochs=ae_epochs, n_samples=N, input_dim=D,
        hidden_dim=hdims[0], latent_dim=2, learning_rate=LR_AE,
        batch_size=batch_size, test_size=0.03, print_interval=0,
        device=DEVICE,
    )
    trainer = MultiModelTrainer(tc)
    mc = make_model_config("ae", WARMUP_LW, extrinsic_dim=D, hidden_dims=hdims)
    trainer.add_model(mc)
    loader = trainer.create_data_loader(train_data)

    # Phase 1: warmup (T+F only)
    for epoch in range(phase1_epochs):
        trainer.train_epoch(loader)

    with torch.no_grad():
        z_check = trainer.models["ae"].encoder(x)
        x_hat = trainer.models["ae"].decoder(z_check)
        recon_pd = ((x_hat - x) ** 2).sum(-1).mean().item() / D
    phase1_ok = recon_pd < 0.1

    # Phase 2: T+F+K (no L_H yet)
    k_only_lw = LossWeights(
        tangent_bundle=1.0, diffeo=1.0, curvature=k_weight,
    )
    _train_phase2(trainer, loader, mc, k_only_lw, phase2_epochs, phase1_ok)

    # Phase 3: T+F+K+L_H (if lh_weight > 0, otherwise just more T+F+K)
    _train_phase2(trainer, loader, mc, full_lw, phase3_epochs, True)

    ae = trainer.models["ae"]
    ae.eval()

    # ── Measure D²(π∘φ) ──
    with torch.no_grad():
        z_pre_raw = ae.encoder(x)
    d2_norm = measure_d2_pi_phi(ae, z_pre_raw)
    d2_mean = d2_norm.mean().item()

    # ── Precompute decoder derivatives ──
    dummy = SDEPipelineTrainer(
        ae, DriftNet(d).to(DEVICE), DiffusionNet(d).to(DEVICE), device=DEVICE,
    )
    z_pre, dphi_pre, d2phi_pre = dummy.precompute_decoder_derivatives(x)

    # ── Encoder Jacobian ──
    dpi = torch.func.vmap(torch.func.jacrev(ae.encoder))(x).detach()

    # ── Compute enc_pull and v2 targets ──
    b_z_enc, b_z_v2 = compute_enc_pull_targets(
        ae, x, v, Lambda, dphi_pre, d2phi_pre, dpi,
    )
    # Target agreement
    target_diff = ((b_z_enc - b_z_v2) ** 2).sum(-1).sqrt()
    target_norm = (b_z_enc ** 2).sum(-1).sqrt()
    target_rel_diff = (target_diff / (target_norm + 1e-8)).mean().item()

    print(f"  L_H={lh_weight:.2f}  ||D²(π∘φ)||={d2_mean:.4f}  "
          f"enc vs v2 target rel_diff={target_rel_diff:.2%}")

    # ── Stage 3: Diffusion (shared) ──
    torch.manual_seed(seed + 200)
    diff_net = DiffusionNet(d).to(DEVICE)
    diff_pipeline = SDEPipelineTrainer(ae, DriftNet(d).to(DEVICE), diff_net, device=DEVICE)
    diff_pipeline.train_stage3_precomputed(
        z_pre, dphi_pre, Lambda,
        epochs=sde_epochs, lr=LR_SDE, batch_size=batch_size,
        print_interval=0,
    )
    diff_net.eval()
    diff_state = copy.deepcopy(diff_net.state_dict())

    # ── Metric for enc_pull regression target ──
    g = (dphi_pre.mT @ dphi_pre).detach()

    # ── Fork Stage 2 ──
    rows = []
    for cond in DRIFT_CONDITIONS:
        t0 = time.time()
        torch.manual_seed(seed + 100)
        drift_net = DriftNet(d).to(DEVICE)

        if cond == "current":
            pipeline = SDEPipelineTrainer(ae, drift_net, DiffusionNet(d).to(DEVICE), device=DEVICE)
            pipeline.train_stage2_precomputed(
                z_pre, dphi_pre, d2phi_pre, v, Lambda,
                epochs=sde_epochs, lr=LR_SDE, batch_size=batch_size,
                print_interval=0,
                lambda_smooth=LAMBDA_SMOOTH, aug_sigma=AUG_SIGMA,
            )
        elif cond == "enc_pull":
            pipeline = SDEPipelineTrainer(ae, drift_net, DiffusionNet(d).to(DEVICE), device=DEVICE)
            pipeline.train_stage2_regression(
                z_pre, b_z_enc, g,
                epochs=sde_epochs, lr=LR_SDE, batch_size=batch_size,
                print_interval=0,
                lambda_smooth=LAMBDA_SMOOTH, aug_sigma=AUG_SIGMA,
            )
        elif cond == "enc_pull_v2":
            pipeline = SDEPipelineTrainer(ae, drift_net, DiffusionNet(d).to(DEVICE), device=DEVICE)
            pipeline.train_stage2_encoder_pullback(
                x, z_pre, dphi_pre, d2phi_pre, dpi, v, Lambda,
                epochs=sde_epochs, lr=LR_SDE, batch_size=batch_size,
                print_interval=0,
                lambda_smooth=LAMBDA_SMOOTH, aug_sigma=AUG_SIGMA,
            )

        drift_net.eval()

        # Attach shared diffusion
        diff_net_copy = DiffusionNet(d).to(DEVICE)
        diff_net_copy.load_state_dict(diff_state)
        diff_net_copy.eval()
        pipeline = SDEPipelineTrainer(ae, drift_net, diff_net_copy, device=DEVICE)

        # Evaluate
        eval_res = evaluate_pipeline(pipeline, ae, sde, seed)
        elapsed = time.time() - t0

        # Fit error vs enc_pull target
        with torch.no_grad():
            b_z_learned = drift_net(z_pre)
        fit_err = ((b_z_learned - b_z_enc) ** 2).sum(-1).sqrt()
        true_norm = (b_z_enc ** 2).sum(-1).sqrt()
        fit_err_rel = (fit_err / (true_norm + 1e-8)).mean().item()

        w2 = eval_res.get("W2@1.0", float("nan"))
        print(f"    {cond:>12s}: W2={w2:.4f}  fit_err={fit_err_rel:.2%}  ({elapsed:.0f}s)")

        rows.append({
            "surface": surface_name, "D": D, "N": N, "seed": seed,
            "lh_weight": lh_weight,
            "condition": cond,
            "d2_pi_phi": d2_mean,
            "target_rel_diff": target_rel_diff,
            "fit_err_rel": fit_err_rel,
            **eval_res,
        })

    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-seeds", type=int, default=5)
    parser.add_argument("--ae-epochs", type=int, default=400)
    parser.add_argument("--sde-epochs", type=int, default=300)
    parser.add_argument("--D", type=int, default=11)
    parser.add_argument("--N", type=int, default=20)
    args = parser.parse_args()

    seeds = [42 + 1000 * i for i in range(args.n_seeds)]
    all_rows = []

    for surface_name in SURFACES:
        for seed in seeds:
            for lh_weight in LH_WEIGHTS:
                print(f"\n{'='*60}")
                print(f"{surface_name} D={args.D} N={args.N} seed={seed} L_H={lh_weight}")
                print(f"{'='*60}")
                rows = run_single(
                    surface_name, args.D, args.N, seed,
                    args.ae_epochs, args.sde_epochs, lh_weight,
                )
                all_rows.extend(rows)

    df = pd.DataFrame(all_rows)
    csv_path = "cycle_hessian_test.csv"
    df.to_csv(csv_path, index=False)
    print(f"\nSaved {len(df)} rows to {csv_path}")

    # ── Summary ──
    print(f"\n{'='*70}")
    print("SUMMARY (median over seeds)")
    print(f"{'='*70}")

    for surface_name in SURFACES:
        mask = df["surface"] == surface_name
        print(f"\n{surface_name} D={args.D} N={args.N}:")

        # D²(π∘φ) by L_H weight
        print(f"\n  ||D²(π∘φ)|| by L_H weight:")
        for lh in LH_WEIGHTS:
            sub = df[mask & (df["lh_weight"] == lh)]
            d2_vals = sub.groupby("seed")["d2_pi_phi"].first()
            td_vals = sub.groupby("seed")["target_rel_diff"].first()
            print(f"    L_H={lh:.2f}:  ||D²||={d2_vals.median():.4f}  "
                  f"enc-v2 diff={td_vals.median():.2%}")

        # W2 by (L_H weight, condition)
        print(f"\n  W2@1.0 by (L_H, condition):")
        header = f"  {'L_H':>5s}"
        for cond in DRIFT_CONDITIONS:
            header += f"  {cond:>12s}"
        print(header)

        for lh in LH_WEIGHTS:
            row_str = f"  {lh:>5.1f}"
            for cond in DRIFT_CONDITIONS:
                sub = df[mask & (df["lh_weight"] == lh) & (df["condition"] == cond)]
                w2_med = sub["W2@1.0"].median()
                row_str += f"  {w2_med:>12.4f}"
            print(row_str)


if __name__ == "__main__":
    main()
