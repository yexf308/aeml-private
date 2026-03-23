#!/usr/bin/env python3
"""Test whether K's negative effect on enc_pull is due to drift network capacity.

Hypothesis: K makes the enc_pull target b_z = Dπ·v + ½Λ:∇²π harder to regress
because it changes the AE's geometry. If the drift network is too small, it can't
fit the more complex target. With a larger network, the gap should close.

Tests DriftNet hidden_dims = [64,64] (default) vs [128,128,128] (3x capacity),
for both baseline (T+F) and K (T+F+K) AEs.

Usage:
    # Quick smoke test
    python3 -u -m experiments.test_drift_capacity --n-seeds 1 --ae-epochs 50 --sde-epochs 50

    # Full (GPU)
    python3 -u -m experiments.test_drift_capacity --n-seeds 5
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
    hidden_dims_for_D, _train_phase2, evaluate_pipeline,
)


SURFACES = ["paraboloid", "hyperbolic_paraboloid"]
DRIFT_HIDDEN_CONFIGS = {
    "small": [64, 64],        # default
    "large": [128, 128, 128], # 3-layer, 2x width
}


def run_one(surface_name, D, N, seed, ae_epochs, sde_epochs, k_weight, drift_label):
    """Run one (surface, seed, k_weight, drift_size) combination."""
    torch.manual_seed(seed)
    np.random.seed(seed)

    surface = FourierAugmentedSurface(surface_name, D)
    sde = create_highd_lambdified_sde(surface, local_drift_fn, local_diffusion_fn)
    hdims = hidden_dims_for_D(D)
    batch_size = min(N, BATCH_SIZE)
    d = 2

    # Sample data
    train_data = sample_from_highd_manifold(
        surface, local_drift_fn, local_diffusion_fn,
        [(-TRAIN_BOUND, TRAIN_BOUND), (-TRAIN_BOUND, TRAIN_BOUND)],
        n_samples=N, seed=seed, device=DEVICE,
    )
    x = train_data.samples.to(DEVICE)
    v = train_data.mu.to(DEVICE)
    Lambda = train_data.cov.to(DEVICE)

    # Phase 1: T+F warmup
    phase1_epochs = ae_epochs // 2
    phase2_epochs = ae_epochs - phase1_epochs

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

    # Quality gate
    with torch.no_grad():
        z_check = trainer.models["ae"].encoder(x)
        x_hat = trainer.models["ae"].decoder(z_check)
        recon_pd = ((x_hat - x) ** 2).sum(-1).mean().item() / D
    phase1_ok = recon_pd < 0.1

    # Phase 2
    if k_weight > 0:
        phase2_lw = LossWeights(
            tangent_bundle=1.0, diffeo=1.0,
            curvature=k_weight * (D / D_REF) ** 0.5,
        )
    else:
        phase2_lw = LossWeights(tangent_bundle=1.0, diffeo=1.0, curvature=0.0)

    _train_phase2(trainer, loader, mc, phase2_lw, phase2_epochs, phase1_ok)

    ae = trainer.models["ae"]
    ae.eval()

    # Precompute derivatives and enc_pull target
    drift_hdims = DRIFT_HIDDEN_CONFIGS[drift_label]
    dummy = SDEPipelineTrainer(
        ae, DriftNet(d, drift_hdims).to(DEVICE), DiffusionNet(d).to(DEVICE),
        device=DEVICE,
    )
    z_pre, dphi_pre, d2phi_pre = dummy.precompute_decoder_derivatives(x)
    b_z_target, g = dummy.precompute_enc_pull_target(x, v, Lambda, dphi_pre)

    # Stage 2: Drift regression with enc_pull target
    torch.manual_seed(seed + 100)
    drift_net = DriftNet(d, drift_hdims).to(DEVICE)
    dp = SDEPipelineTrainer(ae, drift_net, DiffusionNet(d).to(DEVICE), device=DEVICE)
    drift_losses = dp.train_stage2_regression(
        z_pre, b_z_target, g,
        epochs=sde_epochs, lr=LR_SDE, batch_size=batch_size,
        print_interval=max(1, sde_epochs // 5),
        lambda_smooth=LAMBDA_SMOOTH, aug_sigma=AUG_SIGMA,
    )
    drift_net.eval()

    # Stage 3: Diffusion (shared config)
    torch.manual_seed(seed + 200)
    diff_net = DiffusionNet(d).to(DEVICE)
    pipeline = SDEPipelineTrainer(ae, drift_net, diff_net, device=DEVICE)
    pipeline.train_stage3_precomputed(
        z_pre, dphi_pre, Lambda,
        epochs=sde_epochs, lr=LR_SDE, batch_size=batch_size,
        print_interval=0,
    )

    # Evaluate
    eval_res = evaluate_pipeline(pipeline, ae, sde, seed)

    # Final fit error
    with torch.no_grad():
        b_z_learned = drift_net(z_pre)
    fit_err = ((b_z_learned - b_z_target) ** 2).sum(-1).sqrt()
    true_norm = (b_z_target ** 2).sum(-1).sqrt()
    fit_err_rel = (fit_err / (true_norm + 1e-8)).mean().item()

    w2 = eval_res.get("W2@1.0", float("nan"))
    print(f"  K={k_weight:.1f} drift={drift_label:>5s}: "
          f"W2={w2:.4f}  fit_err={fit_err_rel:.2%}  "
          f"final_loss={drift_losses[-1]:.6f}")

    return {
        "surface": surface_name, "D": D, "N": N, "seed": seed,
        "k_weight": k_weight,
        "drift_size": drift_label,
        "drift_loss": drift_losses[-1],
        "fit_err_rel": fit_err_rel,
        **eval_res,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-seeds", type=int, default=5)
    parser.add_argument("--ae-epochs", type=int, default=500)
    parser.add_argument("--sde-epochs", type=int, default=300)
    parser.add_argument("--D", type=int, default=11)
    parser.add_argument("--N", type=int, default=20)
    args = parser.parse_args()

    seeds = [42 + 1000 * i for i in range(args.n_seeds)]
    k_weights = [0.0, BASE_K_WEIGHT]  # baseline vs K
    all_rows = []

    for surface_name in SURFACES:
        for seed in seeds:
            for k_w in k_weights:
                for drift_label in DRIFT_HIDDEN_CONFIGS:
                    print(f"\n{'='*60}")
                    print(f"{surface_name} D={args.D} N={args.N} seed={seed} "
                          f"K={k_w:.1f} drift={drift_label}")
                    print(f"{'='*60}")
                    row = run_one(
                        surface_name, args.D, args.N, seed,
                        args.ae_epochs, args.sde_epochs, k_w, drift_label,
                    )
                    all_rows.append(row)

    df = pd.DataFrame(all_rows)
    csv_path = "drift_capacity_test.csv"
    df.to_csv(csv_path, index=False)
    print(f"\nSaved {len(df)} rows to {csv_path}")

    # Summary
    print(f"\n{'='*80}")
    print("SUMMARY: Does larger drift network close the K gap?")
    print(f"{'='*80}")

    for surface_name in SURFACES:
        mask = df["surface"] == surface_name
        print(f"\n{surface_name}:")
        print(f"  {'K':>5s} {'drift':>6s}  {'W2@1.0':>8s} {'fit_err':>8s} {'drift_loss':>10s}")
        print(f"  {'-'*50}")
        for k_w in k_weights:
            for drift_label in DRIFT_HIDDEN_CONFIGS:
                sub = df[mask & (df["k_weight"] == k_w) &
                         (df["drift_size"] == drift_label)]
                w2_med = sub["W2@1.0"].median()
                fit_med = sub["fit_err_rel"].median()
                dl_med = sub["drift_loss"].median()
                print(f"  {k_w:>5.1f} {drift_label:>6s}  "
                      f"{w2_med:>8.4f} {fit_med:>8.2%} {dl_med:>10.6f}")

        # Key comparison: does the K penalty (gap) shrink with larger net?
        print(f"\n  K effect (K - baseline) on W2:")
        for drift_label in DRIFT_HIDDEN_CONFIGS:
            base_vals = []
            k_vals = []
            for seed in seeds:
                base = df[mask & (df["k_weight"] == 0.0) &
                          (df["drift_size"] == drift_label) &
                          (df["seed"] == seed)]["W2@1.0"]
                k = df[mask & (df["k_weight"] == BASE_K_WEIGHT) &
                       (df["drift_size"] == drift_label) &
                       (df["seed"] == seed)]["W2@1.0"]
                if len(base) == 1 and len(k) == 1:
                    base_vals.append(base.values[0])
                    k_vals.append(k.values[0])
            if len(base_vals) >= 2:
                diffs = np.array(k_vals) - np.array(base_vals)
                print(f"    {drift_label:>6s}: median delta = {np.median(diffs):+.4f}  "
                      f"(K worse in {(diffs > 0).sum()}/{len(diffs)} seeds)")


if __name__ == "__main__":
    main()
