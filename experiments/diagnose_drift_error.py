#!/usr/bin/env python3
"""Diagnose drift error: target quality vs fit quality vs pushforward quality.

Decomposes the 52% drift error at training points into:
  (a) Target quality: Is the implied latent drift from tangential loss correct?
      Compare b_z^true (true latent drift) vs b_z^implied (pullback of observed v).
  (b) Fit quality: Does drift_net match the target b_z^implied?
  (c) Pushforward quality: Does Dphi @ b_z^learned recover the tangential part of v?

Also checks: is the AE chart distortion making the pullback drift too complex?

Usage:
    python3 -u -m experiments.diagnose_drift_error
    python3 -u -m experiments.diagnose_drift_error --n-seeds 5
"""
import argparse
import torch
import numpy as np

from src.numeric.losses import LossWeights
from src.numeric.sde_nets import DriftNet, DiffusionNet
from src.numeric.sde_training import SDEPipelineTrainer
from src.numeric.training import MultiModelTrainer, TrainingConfig
from src.numeric.geometry import regularized_metric_inverse
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
    LAMBDA_SMOOTH, AUG_SIGMA,
    local_drift_fn, local_diffusion_fn,
    hidden_dims_for_D, _train_phase2,
)


def diagnose_single(surface_name, D, N, seed, ae_epochs, sde_epochs, lambda_smooth=LAMBDA_SMOOTH):
    """Run full pipeline and decompose drift error."""
    torch.manual_seed(seed)
    np.random.seed(seed)

    K_pairs = {11: 4, 201: 99}[D]
    surface = FourierAugmentedSurface(surface_name, D)
    sde = create_highd_lambdified_sde(surface, local_drift_fn, local_diffusion_fn)
    k_weight = BASE_K_WEIGHT * (D / D_REF) ** 0.5
    full_lw = LossWeights(tangent_bundle=1.0, diffeo=1.0, curvature=k_weight)
    hdims = hidden_dims_for_D(D)
    batch_size = min(N, BATCH_SIZE)

    # ── Sample training data ──
    train_data = sample_from_highd_manifold(
        surface, local_drift_fn, local_diffusion_fn,
        [(-TRAIN_BOUND, TRAIN_BOUND), (-TRAIN_BOUND, TRAIN_BOUND)],
        n_samples=N, seed=seed, device=DEVICE,
    )
    x = train_data.samples.to(DEVICE)
    v = train_data.mu.to(DEVICE)      # true ambient drift at training points
    Lambda = train_data.cov.to(DEVICE)

    # Also get the local coordinates (u,v) for the training points
    uv_train = train_data.local_samples.to(DEVICE)  # (N, 2) — the (u,v) coordinates

    # ── Stage 1: AE training (K+S condition) ──
    phase1_epochs = ae_epochs // 2
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

    for epoch in range(phase1_epochs):
        trainer.train_epoch(loader)

    with torch.no_grad():
        z_check = trainer.models["ae"].encoder(x)
        x_hat = trainer.models["ae"].decoder(z_check)
        recon_pd = ((x_hat - x) ** 2).sum(-1).mean().item() / D
    phase1_ok = recon_pd < 0.1
    _train_phase2(trainer, loader, mc, full_lw, ae_epochs - phase1_epochs, phase1_ok)

    ae = trainer.models["ae"]
    ae.eval()

    # ── Precompute derivatives ──
    d = 2
    dummy = SDEPipelineTrainer(
        ae, DriftNet(d).to(DEVICE), DiffusionNet(d).to(DEVICE), device=DEVICE,
    )
    z_pre, dphi_pre, d2phi_pre = dummy.precompute_decoder_derivatives(x)

    # ── Stage 2: Train drift ──
    torch.manual_seed(seed + 100)
    drift_net = DriftNet(d).to(DEVICE)
    dp = SDEPipelineTrainer(ae, drift_net, DiffusionNet(d).to(DEVICE), device=DEVICE)
    dp.train_stage2_precomputed(
        z_pre, dphi_pre, d2phi_pre, v, Lambda,
        epochs=sde_epochs, lr=LR_SDE, batch_size=batch_size,
        print_interval=0,
        lambda_smooth=lambda_smooth, aug_sigma=AUG_SIGMA,
    )
    drift_net.eval()

    # ──────────────────────────────────────────────────────────────
    # DIAGNOSTIC: Decompose drift error at training points
    # ──────────────────────────────────────────────────────────────
    with torch.no_grad():
        z = ae.encoder(x).detach()
    dphi = ae.decoder.jacobian_network(z).detach()
    d2phi = ae.decoder.hessian_network(z).detach()

    g = dphi.mT @ dphi                        # (N, d, d)
    ginv = regularized_metric_inverse(g)       # (N, d, d)
    pinv = ginv @ dphi.mT                      # (N, d, D)

    P_hat = dphi @ pinv                        # (N, D, D)
    P_hat = 0.5 * (P_hat + P_hat.mT)

    # 1. TRUE LOCAL DRIFT: b_z^true = local_drift_fn(uv)
    b_z_true_local = local_drift_fn(uv_train)  # (N, 2) — in (u,v) coords

    # 2. IMPLIED LATENT DRIFT from tangential loss target:
    #    The tangential loss minimizes ||P(Dphi @ b_z - (v - q))||^2
    #    The target for b_z is: b_z^target = pinv @ (v - q)
    #    (This is the tangential component, pulled back to latent)
    from src.numeric.geometry import curvature_drift_explicit_full

    # Need Sigma_z_obs for curvature correction
    Lambda_tan = P_hat @ Lambda @ P_hat
    Lambda_tan = 0.5 * (Lambda_tan + Lambda_tan.mT)
    Sigma_z_obs = pinv @ Lambda_tan @ pinv.mT
    Sigma_z_obs = 0.5 * (Sigma_z_obs + Sigma_z_obs.mT)

    q = curvature_drift_explicit_full(d2phi, Sigma_z_obs)  # (N, D)

    # TRUE curvature correction from the ground-truth surface
    true_hess = train_data.hessians.to(DEVICE)       # (N, D, 2, 2)
    true_local_cov = train_data.local_cov.to(DEVICE) # (N, 2, 2)
    from src.numeric.geometry import ambient_quadratic_variation_drift
    q_true = 0.5 * ambient_quadratic_variation_drift(true_local_cov, true_hess)  # (N, D), with ½

    v_minus_q = v - q                          # (N, D) — tangential drift target in ambient
    b_z_target = (pinv @ v_minus_q.unsqueeze(-1)).squeeze(-1)  # (N, d) — pullback to latent

    # Target with TRUE curvature correction (oracle)
    v_minus_q_true = v - q_true               # = J @ mu_local (purely tangential)
    b_z_target_true_q = (pinv @ v_minus_q_true.unsqueeze(-1)).squeeze(-1)  # (N, d)

    # 3. LEARNED DRIFT
    with torch.no_grad():
        b_z_learned = drift_net(z)             # (N, d)

    # 4. PUSHFORWARD of learned drift to ambient
    with torch.no_grad():
        v_learned = (dphi @ b_z_learned.unsqueeze(-1)).squeeze(-1) + q  # (N, D)

    # ── Compute error metrics ──
    # (a) Chart quality: are z coordinates close to true (u,v)?
    # Check if encoder(chart(uv)) ≈ identity (up to affine transform)
    # by looking at the Jacobian of z w.r.t. uv
    z_np = z.cpu().numpy()
    uv_np = uv_train.cpu().numpy()
    # Correlation between z and uv (should be ~1 if chart is good)
    from scipy.stats import pearsonr
    corr_00 = pearsonr(z_np[:, 0], uv_np[:, 0])[0]
    corr_01 = pearsonr(z_np[:, 0], uv_np[:, 1])[0]
    corr_10 = pearsonr(z_np[:, 1], uv_np[:, 0])[0]
    corr_11 = pearsonr(z_np[:, 1], uv_np[:, 1])[0]

    # Which pairing is best?
    if abs(corr_00) + abs(corr_11) > abs(corr_01) + abs(corr_10):
        chart_corr = (abs(corr_00), abs(corr_11))
        pairing = "z0↔u, z1↔v"
    else:
        chart_corr = (abs(corr_01), abs(corr_10))
        pairing = "z0↔v, z1↔u"

    # (b) Target quality: b_z^target vs b_z^true
    # But they're in different coordinate systems! b_z^true is in (u,v),
    # b_z^target is in z-coords. We need to transform one.
    #
    # If the chart φ maps uv → x, and encoder maps x → z,
    # then z ≈ h(uv) for some function h.
    # True latent drift in z-coords: b_z^true_in_z = Dh @ b_z^true_local
    #
    # We can compute Dh ≈ D(encoder ∘ chart)(uv) numerically.
    # But simpler: compare in ambient space where both should agree.

    # Ambient target: Dphi @ b_z^target + q = v (by construction)
    # So target quality in ambient is EXACT at training points.
    # Let's verify:
    v_from_target = (dphi @ b_z_target.unsqueeze(-1)).squeeze(-1) + q
    target_residual = v_from_target - v
    target_ambient_err = (target_residual ** 2).sum(-1).sqrt()

    # (c) Fit quality: b_z^learned vs b_z^target (in latent coords)
    fit_residual = b_z_learned - b_z_target
    fit_err_abs = (fit_residual ** 2).sum(-1).sqrt()
    target_norm = (b_z_target ** 2).sum(-1).sqrt()
    fit_err_rel = fit_err_abs / (target_norm + 1e-8)

    # (d) Pushforward quality: v_learned vs v (in ambient)
    ambient_residual = v_learned - v
    # Project to tangent:
    ambient_residual_tan = (P_hat @ ambient_residual.unsqueeze(-1)).squeeze(-1)
    ambient_err_full = (ambient_residual ** 2).sum(-1).sqrt()
    ambient_err_tan = (ambient_residual_tan ** 2).sum(-1).sqrt()
    v_norm = (v ** 2).sum(-1).sqrt()
    ambient_err_rel = ambient_err_full / (v_norm + 1e-8)

    # (e) Is the curvature correction q large relative to v?
    q_norm = (q ** 2).sum(-1).sqrt()
    q_rel = q_norm / (v_norm + 1e-8)

    # (f) Tangential drift loss at convergence (the actual loss minimized)
    # This is ||P(Dphi @ b_z_learned - (v-q))||^2 / D
    residual_raw = (dphi @ b_z_learned.unsqueeze(-1)).squeeze(-1) - (v - q)
    tan_res = (P_hat @ residual_raw.unsqueeze(-1)).squeeze(-1)
    tang_loss = (tan_res ** 2).sum(-1).mean().item() / D

    # (g) Normal component of v: ||(I-P)v||
    I_mat = torch.eye(D, device=DEVICE).unsqueeze(0)
    N_hat = I_mat - P_hat
    v_normal = (N_hat @ v.unsqueeze(-1)).squeeze(-1)
    v_tan_component = (P_hat @ v.unsqueeze(-1)).squeeze(-1)
    v_normal_norm = (v_normal ** 2).sum(-1).sqrt()
    v_tan_norm = (v_tan_component ** 2).sum(-1).sqrt()

    # Print diagnostics
    print(f"\n{'='*70}")
    print(f"DRIFT ERROR DIAGNOSTIC: {surface_name} D={D} N={N} seed={seed}")
    print(f"{'='*70}")

    print(f"\n1. CHART QUALITY")
    print(f"   Recon per dim: {recon_pd:.6f}")
    print(f"   Best pairing: {pairing}")
    print(f"   Correlation: ({chart_corr[0]:.4f}, {chart_corr[1]:.4f})")

    q_true_norm = (q_true ** 2).sum(-1).sqrt()
    q_error = ((q - q_true) ** 2).sum(-1).sqrt()
    q_error_rel = q_error / (q_true_norm + 1e-8)

    print(f"\n2. CURVATURE CORRECTION")
    print(f"   ||q_learned||: {q_norm.mean():.4f} ± {q_norm.std():.4f}")
    print(f"   ||q_true||:    {q_true_norm.mean():.4f} ± {q_true_norm.std():.4f}")
    print(f"   ||v||:         {v_norm.mean():.4f} ± {v_norm.std():.4f}")
    print(f"   ||q||/||v||:   {q_rel.mean():.4f} ± {q_rel.std():.4f}")
    print(f"   ||q_err||:     {q_error.mean():.4f} ± {q_error.std():.4f}")
    print(f"   q_err relative: {q_error_rel.mean():.2%} ± {q_error_rel.std():.2%}")
    print(f"   ||q_err||/||v||: {(q_error / (v_norm + 1e-8)).mean():.4f}")

    print(f"\n3. AMBIENT DRIFT STRUCTURE")
    print(f"   ||v_tan||:    {v_tan_norm.mean():.4f} ± {v_tan_norm.std():.4f}")
    print(f"   ||v_normal||: {v_normal_norm.mean():.4f} ± {v_normal_norm.std():.4f}")
    print(f"   normal/total: {(v_normal_norm / (v_norm + 1e-8)).mean():.4f}")

    # Compare targets: with learned q vs true q
    target_diff = b_z_target - b_z_target_true_q
    target_diff_norm = (target_diff ** 2).sum(-1).sqrt()
    target_true_norm = (b_z_target_true_q ** 2).sum(-1).sqrt()
    target_diff_rel = target_diff_norm / (target_true_norm + 1e-8)

    # How close is the oracle target to the true local drift?
    # (This measures chart distortion only — in the limit of perfect chart, should be ~0)
    # Note: b_z_target_true_q is in z-coords, b_z_true_local is in (u,v) coords.
    # These are different coord systems so we compare in AMBIENT space.
    v_from_oracle_target = (dphi @ b_z_target_true_q.unsqueeze(-1)).squeeze(-1) + q_true
    oracle_vs_true_ambient = ((v_from_oracle_target - v) ** 2).sum(-1).sqrt()
    # This should be ||N_hat @ v|| = normal component (since v_from_oracle = P@v + q_true - q_true... wait)
    # Actually: v - q_true = J@mu_local (tangential), so
    # Dphi @ pinv @ (J@mu_local) + q_true = P_hat @ (J@mu_local) + q_true
    # v = J@mu_local + q_true, so error = (P_hat - I)(J@mu_local) = -N_hat @ J@mu_local
    # This is the chart error — how well the AE's tangent plane matches the true tangent plane

    print(f"\n   CURVATURE vs CHART ERROR DECOMPOSITION")
    print(f"   ||b_z_target(oracle)||:  {target_true_norm.mean():.4f}")
    print(f"   ||b_z_target(learned)||: {(b_z_target**2).sum(-1).sqrt().mean():.4f}")
    print(f"   Target shift from q error:  {target_diff_norm.mean():.4f}  ({target_diff_rel.mean():.2%} relative)")
    print(f"   Oracle ambient error (chart distortion): {oracle_vs_true_ambient.mean():.4f}")

    print(f"\n4. TARGET QUALITY (ambient: Dphi@b_z_target + q vs v)")
    print(f"   Should be ~0 (target is exact pullback of tangential part)")
    print(f"   ||target - v||: {target_ambient_err.mean():.6f} ± {target_ambient_err.std():.6f}")

    print(f"\n5. FIT QUALITY (latent: b_z_learned vs b_z_target)")
    print(f"   ||b_z_target||:  {target_norm.mean():.4f} ± {target_norm.std():.4f}")
    print(f"   ||b_z_learned||: {(b_z_learned**2).sum(-1).sqrt().mean():.4f}")
    print(f"   ||fit_err||_abs: {fit_err_abs.mean():.4f} ± {fit_err_abs.std():.4f}")
    print(f"   ||fit_err||_rel: {fit_err_rel.mean():.2%} ± {fit_err_rel.std():.2%}")
    print(f"   Tangential loss:  {tang_loss:.6f}")

    print(f"\n6. PUSHFORWARD QUALITY (ambient: Dphi@b_z_learned + q vs v)")
    print(f"   ||error||_full:  {ambient_err_full.mean():.4f} ± {ambient_err_full.std():.4f}")
    print(f"   ||error||_tan:   {ambient_err_tan.mean():.4f} ± {ambient_err_tan.std():.4f}")
    print(f"   relative error:  {ambient_err_rel.mean():.2%}")

    # (h) Detailed look at b_z_target complexity
    print(f"\n7. DRIFT COMPLEXITY IN LATENT SPACE")
    print(f"   b_z_target stats:")
    for i in range(d):
        vals = b_z_target[:, i]
        print(f"     dim {i}: mean={vals.mean():.4f} std={vals.std():.4f} "
              f"range=[{vals.min():.4f}, {vals.max():.4f}]")
    print(f"   b_z_learned stats:")
    for i in range(d):
        vals = b_z_learned[:, i]
        print(f"     dim {i}: mean={vals.mean():.4f} std={vals.std():.4f} "
              f"range=[{vals.min():.4f}, {vals.max():.4f}]")

    # (i) Is the drift net fitting bias or missing structure?
    # Compute per-point signed error
    print(f"\n8. ERROR STRUCTURE")
    for i in range(d):
        err_i = (b_z_learned[:, i] - b_z_target[:, i])
        print(f"   dim {i}: bias={err_i.mean():.4f}  std={err_i.std():.4f}  "
              f"RMSE={err_i.pow(2).mean().sqrt():.4f}")

    return {
        "surface": surface_name, "D": D, "N": N, "seed": seed,
        "recon_pd": recon_pd,
        "chart_corr_min": min(chart_corr),
        "q_over_v": q_rel.mean().item(),
        "fit_err_rel": fit_err_rel.mean().item(),
        "ambient_err_rel": ambient_err_rel.mean().item(),
        "tang_loss": tang_loss,
        "v_normal_frac": (v_normal_norm / (v_norm + 1e-8)).mean().item(),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-seeds", type=int, default=1)
    parser.add_argument("--ae-epochs", type=int, default=400)
    parser.add_argument("--sde-epochs", type=int, default=300)
    parser.add_argument("--D", type=int, default=11)
    parser.add_argument("--N", type=int, default=20)
    args = parser.parse_args()

    seeds = [42 + 1000 * i for i in range(args.n_seeds)]
    rows = []

    for lam_s in [LAMBDA_SMOOTH, 0.0]:
        print(f"\n\n{'#'*70}")
        print(f"# LAMBDA_SMOOTH = {lam_s}")
        print(f"{'#'*70}")
        for surf in ["paraboloid", "hyperbolic_paraboloid"]:
            for seed in seeds:
                row = diagnose_single(surf, args.D, args.N, seed,
                                      args.ae_epochs, args.sde_epochs,
                                      lambda_smooth=lam_s)
                row["lambda_smooth"] = lam_s
                rows.append(row)

    if len(rows) > 2:
        import pandas as pd
        df = pd.DataFrame(rows)
        print("\n\n=== SUMMARY ===")
        for surf in df.surface.unique():
            sub = df[df.surface == surf]
            print(f"\n{surf}:")
            for col in ["fit_err_rel", "ambient_err_rel", "q_over_v", "v_normal_frac"]:
                vals = sub[col]
                print(f"  {col}: {vals.median():.4f} ({vals.min():.4f} – {vals.max():.4f})")


if __name__ == "__main__":
    main()
