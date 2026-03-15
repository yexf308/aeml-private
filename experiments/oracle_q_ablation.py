#!/usr/bin/env python3
"""Oracle q ablation: diagnose and fix the decoder Hessian bottleneck.

6-condition fork experiment. Trains AE once per (surface, seed), then
forks Stage 2 into conditions that vary how the drift target is computed:

  1. no_q:        pinv @ v                     (is learned q helping at all?)
  2. current:     pinv @ (v - q_learned)        (status quo baseline)
  3. oracle_q:    pinv @ (v - q_true)           (fixes q, keeps decoder-side)
  4. enc_pull:    Dπ·v + ½Λ:∇²π                (encoder-side Itô, true oracle)
  5. enc_pull_v2: Dπ·(v - q_φ)                 (avoids encoder Hessian via D²(π∘φ)=0)
  6. direct_1st:  D(enc∘chart)·μ_local          (1st-order only, no Itô correction)

Stage 3 diffusion is trained once per AE and shared across all drift branches.

Usage:
    # Smoke test
    python3 -u -m experiments.oracle_q_ablation --n-seeds 1 --ae-epochs 50 --sde-epochs 50

    # Full
    python3 -u -m experiments.oracle_q_ablation --n-seeds 5

    # With D=201
    python3 -u -m experiments.oracle_q_ablation --n-seeds 5 --D 201
"""
import argparse
import copy
import time
import torch
import numpy as np
import pandas as pd

from src.numeric.losses import LossWeights
from src.numeric.sde_nets import DriftNet, DiffusionNet
from src.numeric.sde_training import SDEPipelineTrainer
from src.numeric.training import MultiModelTrainer, TrainingConfig
from src.numeric.geometry import (
    regularized_metric_inverse,
    ambient_quadratic_variation_drift,
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
D_CONFIGS = {11: 4, 201: 99}
CONDITION_LABELS = ["no_q", "current", "oracle_q", "enc_pull", "enc_pull_v2", "direct_1st"]


def precompute_oracle_targets(ae, train_data, surface, z_pre, dphi_pre, device):
    """Precompute all oracle drift targets.

    Returns dict with keys: q_true, q_no (zeros), b_z_enc_pull, b_z_direct_1st, g.
    """
    x = train_data.samples.to(device)
    v = train_data.mu.to(device)
    Lambda = train_data.cov.to(device)
    uv = train_data.local_samples.to(device)
    true_hess = train_data.hessians.to(device)
    true_local_cov = train_data.local_cov.to(device)

    N, D = x.shape
    d = z_pre.shape[1]

    # ── q_true: ground-truth curvature correction (with ½ factor) ──
    q_true = 0.5 * ambient_quadratic_variation_drift(true_local_cov, true_hess)

    # ── q_no: zero (for "no q" condition) ──
    q_no = torch.zeros_like(v)

    # ── Metric tensor from decoder ──
    g = dphi_pre.mT @ dphi_pre  # (N, d, d)

    # ── Encoder pullback: b_z = Dπ·v + ½Λ:∇²π ──
    # Encoder Jacobian: (N, d, D)
    with torch.no_grad():
        z = ae.encoder(x)

    # Compute encoder Jacobian via vmap(jacrev)
    dpi = torch.func.vmap(torch.func.jacrev(ae.encoder))(x).detach()  # (N, d, D)

    # Compute encoder Hessian and contract with Lambda
    # ∇²π has shape (B, d, D, D) — use ae.encoder_hessian which handles batching.
    # Contract: ½Λ:∇²π = ½ Σ_rs Λ_rs ∂²π_i/∂x_r∂x_s → (B, d)
    ito_corrections = []
    batch_sz = 8  # Small batches to manage memory for D=201
    for i in range(0, N, batch_sz):
        x_b = x[i:i + batch_sz]
        Lambda_b = Lambda[i:i + batch_sz]
        d2pi_b = ae.encoder_hessian(x_b).detach()  # (B, d, D, D)
        # Contract: einsum('brs, birs -> bi', Λ, ∇²π)
        corr_b = 0.5 * torch.einsum('brs, birs -> bi', Lambda_b, d2pi_b)
        ito_corrections.append(corr_b)
    ito_correction = torch.cat(ito_corrections)  # (N, d)

    # b_z = Dπ·v + ½Λ:∇²π
    b_z_enc_pull = (dpi @ v.unsqueeze(-1)).squeeze(-1) + ito_correction  # (N, d)
    b_z_enc_pull = b_z_enc_pull.detach()

    # ── Direct 1st-order: D(enc∘chart)·μ_local ──
    # Compute Jacobian of h = encoder ∘ chart at uv points
    def _h(uv_single):
        """encoder(chart(uv)) for a single point."""
        x_pt = surface.chart(uv_single.unsqueeze(0)).squeeze(0)
        return ae.encoder(x_pt.unsqueeze(0)).squeeze(0)

    Dh = torch.func.vmap(torch.func.jacrev(_h))(uv).detach()  # (N, d, 2)

    mu_local = local_drift_fn(uv)  # (N, 2)
    b_z_direct_1st = (Dh @ mu_local.unsqueeze(-1)).squeeze(-1)  # (N, d)
    b_z_direct_1st = b_z_direct_1st.detach()

    return {
        "q_true": q_true.detach(),
        "q_no": q_no,
        "b_z_enc_pull": b_z_enc_pull,
        "b_z_direct_1st": b_z_direct_1st,
        "g": g.detach(),
        "dpi": dpi.detach(),
    }


def run_single(surface_name, D, N, seed, ae_epochs, sde_epochs):
    """Run all 6 conditions for one (surface, D, N, seed)."""
    torch.manual_seed(seed)
    np.random.seed(seed)

    surface = FourierAugmentedSurface(surface_name, D)
    sde = create_highd_lambdified_sde(surface, local_drift_fn, local_diffusion_fn)
    k_weight = BASE_K_WEIGHT * (D / D_REF) ** 0.5
    full_lw = LossWeights(tangent_bundle=1.0, diffeo=1.0, curvature=k_weight)
    hdims = hidden_dims_for_D(D)
    batch_size = min(N, BATCH_SIZE)

    # ── Sample data ──
    train_data = sample_from_highd_manifold(
        surface, local_drift_fn, local_diffusion_fn,
        [(-TRAIN_BOUND, TRAIN_BOUND), (-TRAIN_BOUND, TRAIN_BOUND)],
        n_samples=N, seed=seed, device=DEVICE,
    )
    x = train_data.samples.to(DEVICE)
    v = train_data.mu.to(DEVICE)
    Lambda = train_data.cov.to(DEVICE)

    # ── Stage 1: AE (K condition) ──
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

    # ── Precompute decoder derivatives ──
    d = 2
    dummy = SDEPipelineTrainer(
        ae, DriftNet(d).to(DEVICE), DiffusionNet(d).to(DEVICE), device=DEVICE,
    )
    z_pre, dphi_pre, d2phi_pre = dummy.precompute_decoder_derivatives(x)

    # ── Precompute oracle targets ──
    print(f"  Precomputing oracle targets...")
    oracle = precompute_oracle_targets(ae, train_data, surface, z_pre, dphi_pre, DEVICE)

    # ── Stage 3: Diffusion (shared across all drift conditions) ──
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

    # ── Coefficient errors (chart quality, shared) ──
    torch.manual_seed(seed + 5000)
    eval_uv = (torch.rand(N_EVAL, 2, device=DEVICE) * 2 - 1) * TRAIN_BOUND
    e_mu_chart, e_sigma_chart = compute_coefficient_errors(ae, sde, surface, eval_uv, DEVICE)

    # ── Fork Stage 2 into 5 conditions ──
    rows = []
    for cond in CONDITION_LABELS:
        t0 = time.time()
        torch.manual_seed(seed + 100)
        drift_net = DriftNet(d).to(DEVICE)

        if cond == "no_q":
            pipeline = SDEPipelineTrainer(ae, drift_net, DiffusionNet(d).to(DEVICE), device=DEVICE)
            pipeline.train_stage2_precomputed(
                z_pre, dphi_pre, d2phi_pre, v, Lambda,
                epochs=sde_epochs, lr=LR_SDE, batch_size=batch_size,
                print_interval=0,
                lambda_smooth=LAMBDA_SMOOTH, aug_sigma=AUG_SIGMA,
                q_override=oracle["q_no"],
            )
        elif cond == "current":
            pipeline = SDEPipelineTrainer(ae, drift_net, DiffusionNet(d).to(DEVICE), device=DEVICE)
            pipeline.train_stage2_precomputed(
                z_pre, dphi_pre, d2phi_pre, v, Lambda,
                epochs=sde_epochs, lr=LR_SDE, batch_size=batch_size,
                print_interval=0,
                lambda_smooth=LAMBDA_SMOOTH, aug_sigma=AUG_SIGMA,
            )
        elif cond == "oracle_q":
            pipeline = SDEPipelineTrainer(ae, drift_net, DiffusionNet(d).to(DEVICE), device=DEVICE)
            pipeline.train_stage2_precomputed(
                z_pre, dphi_pre, d2phi_pre, v, Lambda,
                epochs=sde_epochs, lr=LR_SDE, batch_size=batch_size,
                print_interval=0,
                lambda_smooth=LAMBDA_SMOOTH, aug_sigma=AUG_SIGMA,
                q_override=oracle["q_true"],
            )
        elif cond == "enc_pull":
            pipeline = SDEPipelineTrainer(ae, drift_net, DiffusionNet(d).to(DEVICE), device=DEVICE)
            pipeline.train_stage2_regression(
                z_pre, oracle["b_z_enc_pull"], oracle["g"],
                epochs=sde_epochs, lr=LR_SDE, batch_size=batch_size,
                print_interval=0,
                lambda_smooth=LAMBDA_SMOOTH, aug_sigma=AUG_SIGMA,
            )
        elif cond == "enc_pull_v2":
            pipeline = SDEPipelineTrainer(ae, drift_net, DiffusionNet(d).to(DEVICE), device=DEVICE)
            pipeline.train_stage2_encoder_pullback(
                x, z_pre, dphi_pre, d2phi_pre, oracle["dpi"], v, Lambda,
                epochs=sde_epochs, lr=LR_SDE, batch_size=batch_size,
                print_interval=0,
                lambda_smooth=LAMBDA_SMOOTH, aug_sigma=AUG_SIGMA,
            )
        elif cond == "direct_1st":
            pipeline = SDEPipelineTrainer(ae, drift_net, DiffusionNet(d).to(DEVICE), device=DEVICE)
            pipeline.train_stage2_regression(
                z_pre, oracle["b_z_direct_1st"], oracle["g"],
                epochs=sde_epochs, lr=LR_SDE, batch_size=batch_size,
                print_interval=0,
                lambda_smooth=LAMBDA_SMOOTH, aug_sigma=AUG_SIGMA,
            )

        drift_net.eval()

        # Attach shared diffusion net
        diff_net_copy = DiffusionNet(d).to(DEVICE)
        diff_net_copy.load_state_dict(diff_state)
        diff_net_copy.eval()
        pipeline = SDEPipelineTrainer(ae, drift_net, diff_net_copy, device=DEVICE)

        # Evaluate
        eval_res = evaluate_pipeline(pipeline, ae, sde, seed)
        elapsed = time.time() - t0

        # Drift fit diagnostic at training points
        with torch.no_grad():
            b_z_learned = drift_net(z_pre)

        # Compare against encoder pullback target (the "true" latent drift)
        b_z_true = oracle["b_z_enc_pull"]
        fit_err = ((b_z_learned - b_z_true) ** 2).sum(-1).sqrt()
        true_norm = (b_z_true ** 2).sum(-1).sqrt()
        fit_err_rel = (fit_err / (true_norm + 1e-8)).mean().item()

        w2 = eval_res.get("W2@1.0", float("nan"))
        print(f"  {cond:>12s}: W2={w2:.4f}  fit_err={fit_err_rel:.2%}  ({elapsed:.0f}s)")

        rows.append({
            "surface": surface_name, "D": D, "N": N, "seed": seed,
            "condition": cond,
            "E_mu_chart": e_mu_chart, "E_Sigma_chart": e_sigma_chart,
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

    for surf in SURFACES:
        for seed in seeds:
            print(f"\n{'='*60}")
            print(f"{surf} D={args.D} N={args.N} seed={seed}")
            print(f"{'='*60}")
            rows = run_single(surf, args.D, args.N, seed,
                              args.ae_epochs, args.sde_epochs)
            all_rows.extend(rows)

    df = pd.DataFrame(all_rows)
    outfile = "oracle_q_ablation.csv"
    df.to_csv(outfile, index=False)
    print(f"\nSaved {len(df)} rows to {outfile}")

    # ── Summary ──
    print("\n" + "=" * 70)
    print("SUMMARY (median over seeds)")
    print("=" * 70)

    for surf in SURFACES:
        sub = df[df.surface == surf]
        if sub.empty:
            continue
        print(f"\n{surf} D={args.D} N={args.N}:")
        print(f"  {'condition':>12s}  {'W2@1.0':>8s}  {'fit_err':>8s}  {'MTE@0.5':>8s}  {'sW2@0.5':>8s}")
        print(f"  {'-'*12}  {'-'*8}  {'-'*8}  {'-'*8}  {'-'*8}")
        for cond in CONDITION_LABELS:
            sc = sub[sub.condition == cond]
            if sc.empty:
                continue
            w2 = sc["W2@1.0"].median()
            fit = sc["fit_err_rel"].median()
            mte = sc["MTE@0.5"].median() if "MTE@0.5" in sc else float("nan")
            sw2 = sc["sW2@0.5"].median() if "sW2@0.5" in sc else float("nan")
            print(f"  {cond:>12s}  {w2:8.4f}  {fit:7.2%}  {mte:8.4f}  {sw2:8.4f}")

        # Paired comparison: each condition vs "current"
        print(f"\n  Paired vs current (Wilcoxon signed-rank):")
        from scipy.stats import wilcoxon
        current = sub[sub.condition == "current"].sort_values("seed")
        for cond in ["no_q", "oracle_q", "enc_pull", "direct_1st"]:
            other = sub[sub.condition == cond].sort_values("seed")
            if len(other) != len(current) or len(current) < 3:
                continue
            w2_c = current["W2@1.0"].values
            w2_o = other["W2@1.0"].values
            delta = (np.median(w2_o) - np.median(w2_c)) / np.median(w2_c) * 100
            try:
                _, p = wilcoxon(w2_o, w2_c)
            except ValueError:
                p = float("nan")
            print(f"    {cond:>12s}: ΔW2={delta:+.1f}%  p={p:.3f}")


if __name__ == "__main__":
    main()
