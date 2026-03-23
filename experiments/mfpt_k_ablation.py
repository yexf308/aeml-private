"""
MFPT for K conditions: T+F+K and T+K, to test whether curvature penalty
helps MFPT (which is insensitive to decoder stretch unlike W2).

Same setup as mfpt_full_ablation.py but only K conditions.
Two-phase AE training required for K (warmup T+F, then add K).

Usage:
    python -m experiments.mfpt_k_ablation --epochs 50 --sde-epochs 50 --n-seeds 1
    python -m experiments.mfpt_k_ablation
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
    simulate_ground_truth, compute_w2,
    TRAIN_BOUND, BATCH_SIZE, LR_AE, LR_SDE, DEVICE,
    N_TRAJ, DT, BOUNDARY,
)
from experiments.highd_N_D_sweep import (
    hidden_dims_for_D, local_drift_fn, local_diffusion_fn,
)
from experiments.mfpt_full_ablation import (
    compute_mfpt_ambient, train_pipeline, RADII, MFPT_N_STEPS, SURFACES,
)

# ── K conditions (two-phase AE) ──────────────────────────────────────────

CONDITIONS = {
    "T+F":     (LossWeights(tangent_bundle=1.0, diffeo=1.0), None),
    "T+F+K":   (LossWeights(tangent_bundle=1.0, diffeo=1.0),
                 LossWeights(tangent_bundle=1.0, diffeo=1.0, curvature=1.0)),
    "T+K":     (LossWeights(tangent_bundle=1.0),
                 LossWeights(tangent_bundle=1.0, curvature=1.0)),
}

LAMBDA_SMOOTH = 0.5


def train_ae_two_phase(surface, D, seed, N, epochs, phase1_lw, phase2_lw, train_data):
    """Two-phase AE training. Phase 1: warmup, Phase 2: add K."""
    hdims = hidden_dims_for_D(D)
    batch_size = min(N, BATCH_SIZE)

    trainer = MultiModelTrainer(TrainingConfig(
        epochs=epochs, n_samples=N, input_dim=D, hidden_dim=hdims[0],
        latent_dim=2, learning_rate=LR_AE, batch_size=batch_size,
        test_size=0.03, print_interval=max(1, epochs // 5), device=DEVICE,
    ))
    mc = make_model_config("ae", phase1_lw, extrinsic_dim=D, hidden_dims=hdims)
    trainer.add_model(mc)
    loader = trainer.create_data_loader(train_data)

    # Phase 1
    phase1_epochs = epochs if phase2_lw is None else int(epochs * 0.6)
    phase2_epochs = epochs - phase1_epochs

    for epoch in range(phase1_epochs):
        losses = trainer.train_epoch(loader, {mc.name: phase1_lw})
        if (epoch + 1) % max(1, phase1_epochs // 3) == 0:
            print(f"      Phase1 {epoch+1}/{phase1_epochs}: loss={losses['ae']:.6f}")

    # Phase 2 (ramp K)
    if phase2_lw is not None and phase2_epochs > 0:
        warmup = max(1, int(phase2_epochs * 0.2))
        for epoch in range(phase2_epochs):
            if epoch < warmup:
                ramp = (epoch + 1) / warmup
                lw_ep = LossWeights(
                    tangent_bundle=phase2_lw.tangent_bundle,
                    diffeo=phase2_lw.diffeo,
                    curvature=phase2_lw.curvature * ramp,
                )
            else:
                lw_ep = phase2_lw
            losses = trainer.train_epoch(loader, {mc.name: lw_ep})
            if (epoch + 1) % max(1, phase2_epochs // 3) == 0:
                print(f"      Phase2 {epoch+1}/{phase2_epochs}: loss={losses['ae']:.6f}")

    ae = trainer.models["ae"]
    ae.eval()

    x = train_data.samples.to(DEVICE)
    with torch.no_grad():
        z = ae.encoder(x)
        x_hat = ae.decoder(z)
        recon = ((x_hat - x) ** 2).sum(-1).mean().item() / D

    return ae, recon


def evaluate_mfpt(pipeline, ae, sde, seed, n_steps):
    """Compute MFPT and W2."""
    torch.manual_seed(seed + 999)
    init_local = (torch.rand(N_TRAJ, 2, device=DEVICE) * 2 - 1) * 0.5
    init_ambient = sde.chart(init_local).to(DEVICE)

    torch.manual_seed(seed + 1234)
    dW = torch.randn(N_TRAJ, n_steps, 2, device=DEVICE)

    gt_traj, gt_alive, _ = simulate_ground_truth(
        init_local, sde, n_steps, DT, dW, BOUNDARY * 2,
    )

    with torch.no_grad():
        z0 = ae.encoder(init_ambient)
    _, x_traj = pipeline.simulate(z0, n_steps, DT, dW=dW,
                                    boundary=BOUNDARY * 2)

    gt_mfpt = compute_mfpt_ambient(gt_traj, RADII)
    lr_mfpt = compute_mfpt_ambient(x_traj, RADII)

    step_1 = int(round(1.0 / DT))
    both_alive = gt_alive[:, step_1] if step_1 < gt_alive.shape[1] else gt_alive[:, -1]
    w2 = compute_w2(x_traj[:, step_1], gt_traj[:, step_1], both_alive, both_alive)

    row = {"W2@1.0": w2}
    for r in RADII:
        gt_m = gt_mfpt[r]["mean"]
        lr_m = lr_mfpt[r]["mean"]
        row[f"MFPT_gt_r{r}"] = gt_m
        row[f"MFPT_lr_r{r}"] = lr_m
        if np.isfinite(gt_m) and gt_m > 0 and np.isfinite(lr_m):
            row[f"MFPT_err_r{r}"] = abs(lr_m - gt_m) / gt_m
        else:
            row[f"MFPT_err_r{r}"] = float('nan')
        row[f"exit_frac_r{r}"] = lr_mfpt[r]["exit_frac"]
    return row


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-seeds", type=int, default=10)
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--sde-epochs", type=int, default=300)
    parser.add_argument("--base-seed", type=int, default=42)
    parser.add_argument("--D", type=int, default=11)
    parser.add_argument("--N", type=int, default=50)
    parser.add_argument("--output", type=str, default="mfpt_k_ablation.csv")
    args = parser.parse_args()

    seeds = [args.base_seed + i * 1000 for i in range(args.n_seeds)]
    cond_names = list(CONDITIONS.keys())

    print(f"Device: {DEVICE}")
    print(f"D={args.D}, N={args.N}")
    print(f"Surfaces: {SURFACES}")
    print(f"Conditions: {cond_names}")
    print(f"Seeds ({len(seeds)}): {seeds}")
    print(f"Radii: {RADII}\n")

    t0 = time.time()
    all_rows = []

    for surface_name in SURFACES:
        for seed in seeds:
            print(f"\n{'='*60}")
            print(f"  {surface_name} | seed={seed}")
            print(f"{'='*60}")

            torch.manual_seed(seed)
            np.random.seed(seed)

            surface = FourierAugmentedSurface(surface_name, args.D)
            train_data = sample_from_highd_manifold(
                surface, local_drift_fn, local_diffusion_fn,
                [(-TRAIN_BOUND, TRAIN_BOUND), (-TRAIN_BOUND, TRAIN_BOUND)],
                n_samples=args.N, seed=seed, device=DEVICE,
            )
            x = train_data.samples.to(DEVICE)
            v = train_data.mu.to(DEVICE)
            Lambda = train_data.cov.to(DEVICE)
            sde = create_highd_lambdified_sde(surface, local_drift_fn, local_diffusion_fn)

            for cond_label, (p1_lw, p2_lw) in CONDITIONS.items():
                print(f"\n  --- {cond_label} ---")
                torch.manual_seed(seed)
                np.random.seed(seed)

                ae, recon = train_ae_two_phase(
                    surface, args.D, seed, args.N, args.epochs, p1_lw, p2_lw, train_data,
                )
                print(f"    recon={recon:.6f}")

                pipeline = train_pipeline(ae, x, v, Lambda, seed, args.N, args.sde_epochs)

                row = evaluate_mfpt(pipeline, ae, sde, seed, MFPT_N_STEPS)
                row.update({
                    "surface": surface_name,
                    "seed": seed,
                    "condition": cond_label,
                    "recon_per_dim": recon,
                })

                for r in RADII:
                    gt_m = row[f"MFPT_gt_r{r}"]
                    lr_m = row[f"MFPT_lr_r{r}"]
                    err = row[f"MFPT_err_r{r}"]
                    print(f"    r={r}: GT={gt_m:.3f} learned={lr_m:.3f} "
                          f"err={err:.1%} exit={row[f'exit_frac_r{r}']:.0%}")
                print(f"    W2@1.0={row['W2@1.0']:.4f}")

                all_rows.append(row)

            pd.DataFrame(all_rows).to_csv(args.output, index=False)

    elapsed = time.time() - t0
    print(f"\n\nTotal time: {elapsed / 60:.1f} min")

    df = pd.DataFrame(all_rows)
    df.to_csv(args.output, index=False)
    print(f"Saved {len(df)} rows to {args.output}")

    # ── Summary ───────────────────────────────────────────────────────
    print(f"\n\n{'='*80}")
    print("MFPT K-ABLATION SUMMARY")
    print(f"{'='*80}")

    for surface_name in SURFACES:
        print(f"\n  Surface: {surface_name}")
        sub = df[df["surface"] == surface_name]

        for r in RADII:
            col = f"MFPT_err_r{r}"
            print(f"\n    r={r}:")
            print(f"    {'condition':>12s}  {'MFPT_err':>14s}  {'MFPT_learned':>14s}  {'MFPT_GT':>14s}  {'W2@1.0':>14s}")
            for cond in cond_names:
                cs = sub[sub["condition"] == cond]
                err = cs[col].values
                lr = cs[f"MFPT_lr_r{r}"].values
                gt = cs[f"MFPT_gt_r{r}"].values
                w2 = cs["W2@1.0"].values
                print(f"    {cond:>12s}  {np.nanmean(err):>6.1%}±{np.nanstd(err):>.1%}  "
                      f"{np.nanmean(lr):>6.3f}±{np.nanstd(lr):>.3f}  "
                      f"{np.nanmean(gt):>6.3f}±{np.nanstd(gt):>.3f}  "
                      f"{np.nanmean(w2):>6.3f}±{np.nanstd(w2):>.3f}")

        # Paired t-test: T+F+K vs T+F
        print(f"\n    Paired t-test T+F+K vs T+F (negative = K helps):")
        for r in RADII:
            col = f"MFPT_err_r{r}"
            tf = sub[sub["condition"] == "T+F"].sort_values("seed")
            tfk = sub[sub["condition"] == "T+F+K"].sort_values("seed")
            if len(tf) != len(tfk) or len(tf) < 2:
                print(f"      r={r}: insufficient data")
                continue
            tf_v = tf[col].values
            tfk_v = tfk[col].values
            mask = np.isfinite(tf_v) & np.isfinite(tfk_v)
            if mask.sum() < 2:
                continue
            diff = tfk_v[mask] - tf_v[mask]
            _, p_val = stats.ttest_rel(tfk_v[mask], tf_v[mask])
            delta_pct = diff.mean() / tf_v[mask].mean() * 100
            sig = "**" if p_val < 0.01 else "*" if p_val < 0.05 else "+" if p_val < 0.1 else ""
            wins = (tfk_v[mask] < tf_v[mask]).sum()
            print(f"      r={r}: Δ={delta_pct:+.1f}%  p={p_val:.3f}{sig}  wins={wins}/{mask.sum()}")


if __name__ == "__main__":
    main()
