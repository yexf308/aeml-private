#!/usr/bin/env python3
"""Test: latent-coordinate diffusion loss vs ambient diffusion loss.

For each (surface, D, N, seed), trains AE + drift once (K+S condition),
then forks Stage 3 comparing ambient vs latent diffusion loss.

Usage:
    # Smoke test
    python3 -u -m experiments.test_diffusion_smoothness --n-seeds 1 --ae-epochs 50 --sde-epochs 50

    # Full (D=11)
    python3 -u -m experiments.test_diffusion_smoothness

    # With D=201
    python3 -u -m experiments.test_diffusion_smoothness --d-values 11 201
"""
import argparse
import copy
import time
import torch
import numpy as np
import pandas as pd
from scipy import stats

from src.numeric.sde_nets import DriftNet, DiffusionNet
from src.numeric.sde_training import SDEPipelineTrainer
from src.numeric.training import MultiModelTrainer, TrainingConfig
from src.numeric.losses import LossWeights
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-seeds", type=int, default=5)
    parser.add_argument("--ae-epochs", type=int, default=400)
    parser.add_argument("--sde-epochs", type=int, default=300)
    parser.add_argument("--d-values", type=int, nargs="+", default=[11])
    parser.add_argument("--n-values", type=int, nargs="+", default=[20])
    args = parser.parse_args()

    seeds = [42 + 1000 * i for i in range(args.n_seeds)]
    rows = []

    for surf_name in SURFACES:
        for D in args.d_values:
            K_pairs = D_CONFIGS[D]
            surface = FourierAugmentedSurface(surf_name, D)
            sde = create_highd_lambdified_sde(surface, local_drift_fn, local_diffusion_fn)
            k_weight = BASE_K_WEIGHT * (D / D_REF) ** 0.5
            full_lw = LossWeights(tangent_bundle=1.0, diffeo=1.0, curvature=k_weight)
            hdims = hidden_dims_for_D(D)

            for N in args.n_values:
                batch_size = min(N, BATCH_SIZE)

                for seed in seeds:
                    torch.manual_seed(seed)
                    np.random.seed(seed)

                    # ── Sample training data ──
                    train_data = sample_from_highd_manifold(
                        surface, local_drift_fn, local_diffusion_fn,
                        [(-TRAIN_BOUND, TRAIN_BOUND), (-TRAIN_BOUND, TRAIN_BOUND)],
                        n_samples=N, seed=seed, device=DEVICE,
                    )
                    x = train_data.samples.to(DEVICE)
                    v = train_data.mu.to(DEVICE)
                    Lambda = train_data.cov.to(DEVICE)

                    # ── AE Phase 1: T+F warmup ──
                    phase1_epochs = args.ae_epochs // 2
                    tc = TrainingConfig(
                        epochs=args.ae_epochs, n_samples=N, input_dim=D,
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

                    # ── AE Phase 2: T+F+K ──
                    phase2_epochs = args.ae_epochs - phase1_epochs
                    with torch.no_grad():
                        z_check = trainer.models["ae"].encoder(x)
                        x_hat = trainer.models["ae"].decoder(z_check)
                        recon_pd = ((x_hat - x) ** 2).sum(-1).mean().item() / D
                    phase1_ok = recon_pd < 0.1
                    _train_phase2(trainer, loader, mc, full_lw, phase2_epochs, phase1_ok)

                    ae = trainer.models["ae"]
                    ae.eval()

                    # ── Precompute decoder derivatives ──
                    d = 2
                    dummy = SDEPipelineTrainer(
                        ae, DriftNet(d).to(DEVICE), DiffusionNet(d).to(DEVICE),
                        device=DEVICE,
                    )
                    z_pre, dphi_pre, d2phi_pre = dummy.precompute_decoder_derivatives(x)

                    # ── Stage 2: drift (K+S) — once per seed ──
                    torch.manual_seed(seed + 100)
                    drift_net = DriftNet(d).to(DEVICE)
                    dp = SDEPipelineTrainer(
                        ae, drift_net, DiffusionNet(d).to(DEVICE), device=DEVICE,
                    )
                    dp.train_stage2_precomputed(
                        z_pre, dphi_pre, d2phi_pre, v, Lambda,
                        epochs=args.sde_epochs, lr=LR_SDE, batch_size=batch_size,
                        print_interval=0,
                        lambda_smooth=LAMBDA_SMOOTH, aug_sigma=AUG_SIGMA,
                    )
                    drift_net.eval()
                    drift_state = copy.deepcopy(drift_net.state_dict())

                    # Chart-quality coefficient errors
                    torch.manual_seed(seed + 5000)
                    eval_uv = (torch.rand(N_EVAL, 2, device=DEVICE) * 2 - 1) * TRAIN_BOUND
                    e_mu, e_sigma_chart = compute_coefficient_errors(
                        ae, sde, surface, eval_uv, DEVICE,
                    )

                    # ── Stage 3: fork ambient vs latent ──
                    for loss_mode in ["ambient", "latent"]:
                        t0 = time.time()
                        torch.manual_seed(seed + 200)
                        diff_net = DiffusionNet(d).to(DEVICE)
                        pipeline = SDEPipelineTrainer(ae, drift_net, diff_net, device=DEVICE)
                        drift_net.load_state_dict(drift_state)

                        use_latent = (loss_mode == "latent")
                        pipeline.train_stage3_precomputed(
                            z_pre, dphi_pre, Lambda,
                            epochs=args.sde_epochs, lr=LR_SDE,
                            batch_size=batch_size, print_interval=0,
                            use_latent_loss=use_latent,
                        )

                        # Evaluate diffusion quality
                        diff_net.eval()
                        with torch.no_grad():
                            Sigma_learned = diff_net.covariance(z_pre)
                            Lambda_pred = dphi_pre @ Sigma_learned @ dphi_pre.mT
                            e_sig = ((Lambda_pred - Lambda) ** 2).sum((-1, -2)).mean().sqrt().item() / D

                        # Trajectory evaluation
                        eval_res = evaluate_pipeline(pipeline, ae, sde, seed)

                        elapsed = time.time() - t0
                        w2 = eval_res.get("W2@1.0", float("nan"))
                        cond = f"K+S ({loss_mode})"
                        print(f"  {surf_name:>25s} D={D:3d} N={N:3d} seed={seed} "
                              f"{cond:>18s}: W2={w2:.3f}  E_Sig={e_sig:.4f}  ({elapsed:.0f}s)")

                        rows.append({
                            "surface": surf_name, "D": D, "N": N,
                            "seed": seed, "loss_mode": loss_mode,
                            "E_mu": e_mu, "E_Sigma_chart": e_sigma_chart,
                            "E_Sigma_diff": e_sig,
                            **eval_res,
                        })

    df = pd.DataFrame(rows)
    df.to_csv("latent_vs_ambient_diff.csv", index=False)
    print(f"\nSaved {len(df)} rows")

    # ── Summary ──
    print("\n=== Summary (median over seeds) ===")
    for surf in SURFACES:
        for D in args.d_values:
            for N in args.n_values:
                print(f"\n{surf} D={D} N={N}:")
                sub = df[(df.surface == surf) & (df.D == D) & (df.N == N)]
                for mode in ["ambient", "latent"]:
                    sc = sub[sub.loss_mode == mode]
                    if len(sc) == 0:
                        continue
                    print(f"  {mode:>8s}: W2={sc['W2@1.0'].median():.3f}  "
                          f"E_Sig={sc['E_Sigma_diff'].median():.4f}")

                # Paired t-test
                amb = sub[sub.loss_mode == "ambient"].sort_values("seed")
                lat = sub[sub.loss_mode == "latent"].sort_values("seed")
                if len(amb) == len(lat) and len(amb) > 1:
                    w2_a = amb["W2@1.0"].values
                    w2_l = lat["W2@1.0"].values
                    t, p = stats.ttest_rel(w2_l, w2_a)
                    delta = (np.median(w2_l) - np.median(w2_a)) / np.median(w2_a) * 100
                    print(f"  latent vs ambient: ΔW2={delta:+.1f}%  p={p:.3f}")

                    es_a = amb["E_Sigma_diff"].values
                    es_l = lat["E_Sigma_diff"].values
                    t2, p2 = stats.ttest_rel(es_l, es_a)
                    delta2 = (np.median(es_l) - np.median(es_a)) / np.median(es_a) * 100
                    print(f"  latent vs ambient: ΔE_Sig={delta2:+.1f}%  p={p2:.3f}")


if __name__ == "__main__":
    main()
