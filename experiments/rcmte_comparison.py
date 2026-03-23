"""
Quick reference-chart MTE comparison: baseline vs T+F, enc-pull vs dec-side.

Uses D=11, N=20, 5 seeds on paraboloid to validate that rcMTE correctly
shows T+F > baseline (which W2 fails to show due to decoder stretch).

Usage:
    python -m experiments.rcmte_comparison --n-seeds 5
    python -m experiments.rcmte_comparison --n-seeds 10 --D 201
"""

import argparse
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
)
from experiments.highd_N_D_sweep import (
    hidden_dims_for_D, local_drift_fn, local_diffusion_fn,
    compute_coefficient_errors, N_EVAL,
)

LAMBDA_SMOOTH = 0.0
AUG_SIGMA = 0.1


def _train_ae(surface, D, seed, N, epochs, lw, train_data):
    hdims = hidden_dims_for_D(D)
    bs = min(N, BATCH_SIZE)
    trainer = MultiModelTrainer(TrainingConfig(
        epochs=epochs, n_samples=N, input_dim=D, hidden_dim=hdims[0],
        latent_dim=2, learning_rate=LR_AE, batch_size=bs,
        test_size=0.03, print_interval=max(1, epochs // 5), device=DEVICE,
    ))
    mc = make_model_config("ae", lw, extrinsic_dim=D, hidden_dims=hdims)
    trainer.add_model(mc)
    loader = trainer.create_data_loader(train_data)
    for epoch in range(epochs):
        trainer.train_epoch(loader, {mc.name: lw})
    ae = trainer.models["ae"]
    ae.eval()
    return ae


def _train_sde_enc_pull(ae, x, v, Lambda, seed, N, epochs_sde):
    d, bs = 2, min(N, BATCH_SIZE)
    tmp = SDEPipelineTrainer(ae, DriftNet(d).to(DEVICE), DiffusionNet(d).to(DEVICE), device=DEVICE)
    z_pre, dphi_pre, _ = tmp.precompute_decoder_derivatives(x)
    b_z_target, g = tmp.precompute_enc_pull_target(x, v, Lambda, dphi_pre)

    torch.manual_seed(seed + 100)
    drift_net = DriftNet(d).to(DEVICE)
    pipe = SDEPipelineTrainer(ae, drift_net, DiffusionNet(d).to(DEVICE), device=DEVICE)
    pipe.train_stage2_regression(
        z_pre, b_z_target, g,
        epochs=epochs_sde, lr=LR_SDE, batch_size=bs,
        print_interval=max(1, epochs_sde // 3),
        lambda_smooth=LAMBDA_SMOOTH, aug_sigma=AUG_SIGMA,
    )

    torch.manual_seed(seed + 200)
    diff_net = DiffusionNet(d).to(DEVICE)
    pipe2 = SDEPipelineTrainer(ae, drift_net, diff_net, device=DEVICE)
    pipe2.train_stage3_precomputed(
        z_pre, dphi_pre, Lambda,
        epochs=epochs_sde, lr=LR_SDE, batch_size=bs,
        print_interval=max(1, epochs_sde // 3),
    )
    return pipe2


def _train_sde_dec_side(ae, x, v, Lambda, seed, N, epochs_sde):
    """Train SDE using decoder-side drift target (for comparison)."""
    d, bs = 2, min(N, BATCH_SIZE)
    tmp = SDEPipelineTrainer(ae, DriftNet(d).to(DEVICE), DiffusionNet(d).to(DEVICE), device=DEVICE)
    z_pre, dphi_pre, d2phi_pre = tmp.precompute_decoder_derivatives(x)

    torch.manual_seed(seed + 100)
    drift_net = DriftNet(d).to(DEVICE)
    diff_net_dummy = DiffusionNet(d).to(DEVICE)
    pipe = SDEPipelineTrainer(ae, drift_net, diff_net_dummy, device=DEVICE)
    pipe.train_stage2_precomputed(
        z_pre, dphi_pre, d2phi_pre,
        v, Lambda,
        epochs=epochs_sde, lr=LR_SDE, batch_size=bs,
        print_interval=max(1, epochs_sde // 3),
    )

    torch.manual_seed(seed + 200)
    diff_net = DiffusionNet(d).to(DEVICE)
    pipe2 = SDEPipelineTrainer(ae, drift_net, diff_net, device=DEVICE)
    pipe2.train_stage3_precomputed(
        z_pre, dphi_pre, Lambda,
        epochs=epochs_sde, lr=LR_SDE, batch_size=bs,
        print_interval=max(1, epochs_sde // 3),
    )
    return pipe2


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-seeds", type=int, default=5)
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--sde-epochs", type=int, default=300)
    parser.add_argument("--base-seed", type=int, default=42)
    parser.add_argument("--D", type=int, default=11)
    parser.add_argument("--N", type=int, default=20)
    parser.add_argument("--output", type=str, default="rcmte_comparison.csv")
    args = parser.parse_args()

    seeds = [args.base_seed + i * 1000 for i in range(args.n_seeds)]
    surfaces = ["paraboloid", "hyperbolic_paraboloid"]

    print(f"Device: {DEVICE}")
    print(f"D={args.D}, N={args.N}")
    print(f"Seeds ({len(seeds)}): {seeds}")
    print(f"Surfaces: {surfaces}")
    print()

    t0 = time.time()
    all_rows = []

    for surface_name in surfaces:
        for seed in seeds:
            print(f"\n{'='*60}")
            print(f"  {surface_name} | D={args.D} | seed={seed}")
            print(f"{'='*60}")

            torch.manual_seed(seed)
            np.random.seed(seed)

            surface = FourierAugmentedSurface(surface_name, args.D)
            train_data = sample_from_highd_manifold(
                surface, local_drift_fn, local_diffusion_fn,
                [(-TRAIN_BOUND, TRAIN_BOUND), (-TRAIN_BOUND, TRAIN_BOUND)],
                n_samples=args.N, seed=seed, device=DEVICE,
            )
            sde = create_highd_lambdified_sde(surface, local_drift_fn, local_diffusion_fn)
            x = train_data.samples.to(DEVICE)
            v = train_data.mu.to(DEVICE)
            Lambda = train_data.cov.to(DEVICE)

            for ae_label, lw in [("baseline", LossWeights()), ("T+F", LossWeights(tangent_bundle=1.0, diffeo=1.0))]:
                print(f"\n  --- AE: {ae_label} ---")
                torch.manual_seed(seed)
                np.random.seed(seed)
                ae = _train_ae(surface, args.D, seed, args.N, args.epochs, lw, train_data)

                # Coefficient errors
                torch.manual_seed(seed + 5000)
                eval_uv = (torch.rand(N_EVAL, 2, device=DEVICE) * 2 - 1) * TRAIN_BOUND
                e_mu, e_sigma = compute_coefficient_errors(ae, sde, surface, eval_uv, DEVICE)
                print(f"    E_mu={e_mu:.4f}  E_Sigma={e_sigma:.4f}")

                for drift_label, train_fn in [("enc-pull", _train_sde_enc_pull), ("dec-side", _train_sde_dec_side)]:
                    print(f"\n    Drift: {drift_label}")
                    pipe = train_fn(ae, x, v, Lambda, seed, args.N, args.sde_epochs)
                    results = evaluate_pipeline(pipe, ae, sde, seed)
                    row = {
                        "surface": surface_name,
                        "D": args.D,
                        "N": args.N,
                        "seed": seed,
                        "ae_condition": ae_label,
                        "drift_method": drift_label,
                        "E_mu": e_mu,
                        "E_Sigma": e_sigma,
                        **results,
                    }
                    all_rows.append(row)

            pd.DataFrame(all_rows).to_csv(args.output, index=False)

    elapsed = time.time() - t0
    print(f"\n\nTotal time: {elapsed / 60:.1f} min")

    df = pd.DataFrame(all_rows)
    df.to_csv(args.output, index=False)
    print(f"Saved {len(df)} rows to {args.output}")

    # Summary
    print(f"\n\n{'='*80}")
    print("rcMTE COMPARISON SUMMARY")
    print(f"{'='*80}")

    metrics = ["rcMTE@1.0", "MTE@1.0", "W2@1.0"]

    for surface_name in surfaces:
        print(f"\n  {surface_name}:")
        sub = df[df["surface"] == surface_name]

        # Print means
        print(f"    {'ae':>8s}  {'drift':>10s}  {'rcMTE@1.0':>12s}  {'MTE@1.0':>12s}  {'W2@1.0':>12s}  {'E_mu':>12s}")
        for ae_cond in ["baseline", "T+F"]:
            for drift in ["enc-pull", "dec-side"]:
                s = sub[(sub["ae_condition"] == ae_cond) & (sub["drift_method"] == drift)]
                if len(s) == 0:
                    continue
                print(f"    {ae_cond:>8s}  {drift:>10s}  "
                      f"{s['rcMTE@1.0'].mean():.4f}±{s['rcMTE@1.0'].std():.3f}  "
                      f"{s['MTE@1.0'].mean():.4f}±{s['MTE@1.0'].std():.3f}  "
                      f"{s['W2@1.0'].mean():.4f}±{s['W2@1.0'].std():.3f}  "
                      f"{s['E_mu'].mean():.4f}±{s['E_mu'].std():.3f}")

        # Paired comparisons
        print(f"\n    Paired comparisons (negative = first is better):")

        # T+F vs baseline (enc-pull)
        tf_enc = sub[(sub["ae_condition"] == "T+F") & (sub["drift_method"] == "enc-pull")].sort_values("seed")
        bl_enc = sub[(sub["ae_condition"] == "baseline") & (sub["drift_method"] == "enc-pull")].sort_values("seed")
        if len(tf_enc) > 1 and len(bl_enc) > 1:
            for m in metrics:
                delta = (tf_enc[m].values - bl_enc[m].values)
                pct = delta.mean() / bl_enc[m].values.mean() * 100
                _, p = stats.ttest_rel(tf_enc[m].values, bl_enc[m].values)
                sig = "**" if p < 0.01 else "*" if p < 0.05 else "+" if p < 0.1 else ""
                print(f"      T+F vs baseline (enc-pull) {m:>12s}: Δ={pct:+.1f}%  p={p:.3f}{sig}")

        # enc-pull vs dec-side (T+F AE)
        tf_enc = sub[(sub["ae_condition"] == "T+F") & (sub["drift_method"] == "enc-pull")].sort_values("seed")
        tf_dec = sub[(sub["ae_condition"] == "T+F") & (sub["drift_method"] == "dec-side")].sort_values("seed")
        if len(tf_enc) > 1 and len(tf_dec) > 1:
            for m in metrics:
                delta = (tf_enc[m].values - tf_dec[m].values)
                pct = delta.mean() / tf_dec[m].values.mean() * 100
                _, p = stats.ttest_rel(tf_enc[m].values, tf_dec[m].values)
                sig = "**" if p < 0.01 else "*" if p < 0.05 else "+" if p < 0.1 else ""
                print(f"      enc-pull vs dec-side (T+F) {m:>12s}: Δ={pct:+.1f}%  p={p:.3f}{sig}")


if __name__ == "__main__":
    main()
