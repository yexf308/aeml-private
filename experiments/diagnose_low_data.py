"""
Diagnostic: Does K regularization help in the low-data regime?

With N_TRAIN=2000 and a [64,64] DiffusionNet (~4600 params), the network is
slightly overconstrained (6000 cov constraints > 4600 params). K adds redundant
constraints. But with N=20-50, the network is massively underconstrained
(60-150 constraints for 4600 params), and K might regularize effectively.

Sweep: N_TRAIN ∈ {20, 50, 200} × lambda_K ∈ {0, 0.1, 1.0}
Fixed: T+F AE (trained on full 2000 pts), shared drift_net, sinusoidal surface.

Usage:
    PYTHONUNBUFFERED=1 python -m experiments.diagnose_low_data
"""

import time
import torch
import numpy as np

from src.numeric.datagen import sample_from_manifold
from src.numeric.losses import LossWeights
from src.numeric.sde_nets import DriftNet, DiffusionNet
from src.numeric.sde_training import SDEPipelineTrainer
from src.numeric.training import MultiModelTrainer, TrainingConfig

from experiments.common import make_model_config
from experiments.data_driven_sde import (
    create_manifold_sde, evaluate_pipeline, lambdify_sde,
    TRAIN_BOUND, BATCH_SIZE, LR_AE, LR_SDE, SEED, DEVICE,
)
from experiments.trajectory_fidelity_study import compute_w2, compute_mmd


def main():
    surface_name = "sinusoidal"
    epochs_ae = 500
    epochs_sde = 300
    seed = SEED
    d = 2

    n_train_values = [20, 50, 200]
    lambda_K_values = [0.0, 0.1, 1.0]

    print(f"Device: {DEVICE}")
    print(f"Surface: {surface_name}")
    print(f"N_TRAIN sweep: {n_train_values}")
    print(f"lambda_K sweep: {lambda_K_values}")

    t0 = time.time()

    # Generate FULL dataset (2000 pts) for AE training
    print("\n--- Generating full dataset for AE training ---")
    torch.manual_seed(seed)
    np.random.seed(seed)
    manifold_sde = create_manifold_sde(surface_name)
    full_data = sample_from_manifold(
        manifold_sde,
        [(-TRAIN_BOUND, TRAIN_BOUND), (-TRAIN_BOUND, TRAIN_BOUND)],
        n_samples=2000, seed=seed, device=DEVICE,
    )

    # Train AE on full data (same quality AE for all runs)
    print(f"\n--- Stage 1: AE (T+F, {epochs_ae} epochs, N=2000) ---")
    torch.manual_seed(seed)
    np.random.seed(seed)
    lw = LossWeights(tangent_bundle=1.0, diffeo=1.0)
    trainer = MultiModelTrainer(TrainingConfig(
        epochs=epochs_ae, n_samples=2000, input_dim=3, hidden_dim=64,
        latent_dim=2, learning_rate=LR_AE, batch_size=BATCH_SIZE,
        test_size=0.03, print_interval=max(1, epochs_ae // 5), device=DEVICE,
    ))
    trainer.add_model(make_model_config("ae", lw, hidden_dims=[64, 64]))
    loader = trainer.create_data_loader(full_data)
    for epoch in range(epochs_ae):
        losses = trainer.train_epoch(loader)
        if (epoch + 1) % max(1, epochs_ae // 5) == 0:
            print(f"  Epoch {epoch+1}: loss={losses['ae']:.6f}")
    autoencoder = trainer.models["ae"]
    autoencoder.eval()

    # Lambdify SDE for evaluation
    sde = lambdify_sde(create_manifold_sde(surface_name))

    rows = []

    for n_train in n_train_values:
        print(f"\n{'='*70}")
        print(f"N_TRAIN = {n_train}  (params/constraints = {4600}/{n_train * 3})")
        print(f"{'='*70}")

        # Subsample training data for SDE stages
        torch.manual_seed(seed)
        idx = torch.randperm(2000)[:n_train]
        x_sub = full_data.samples[idx].to(DEVICE)
        v_sub = full_data.mu[idx].to(DEVICE)
        Lambda_sub = full_data.cov[idx].to(DEVICE)

        # Precompute derivatives on subsample
        dummy = SDEPipelineTrainer(
            autoencoder, DriftNet(d).to(DEVICE), DiffusionNet(d).to(DEVICE), device=DEVICE,
        )
        z_pre, dphi_pre, d2phi_pre = dummy.precompute_decoder_derivatives(x_sub)

        # Shared drift_net (trained on subsample)
        torch.manual_seed(seed + 100)
        drift_net = DriftNet(d).to(DEVICE)
        drift_pipeline = SDEPipelineTrainer(
            autoencoder, drift_net, DiffusionNet(d).to(DEVICE), device=DEVICE,
        )
        bs = min(BATCH_SIZE, n_train)
        drift_losses = drift_pipeline.train_stage2_precomputed(
            z_pre, dphi_pre, d2phi_pre, v_sub, Lambda_sub,
            epochs=epochs_sde, lr=LR_SDE,
            batch_size=bs, print_interval=max(1, epochs_sde // 5),
        )
        drift_net.eval()
        print(f"  Drift loss: {drift_losses[-1]:.6f}")

        for lk in lambda_K_values:
            print(f"\n  --- lambda_K={lk} ---")

            torch.manual_seed(seed + 200)
            diffusion_net = DiffusionNet(d).to(DEVICE)
            pipeline = SDEPipelineTrainer(
                autoencoder, drift_net, diffusion_net, device=DEVICE,
            )

            t_run = time.time()
            diff_losses = pipeline.train_stage3_precomputed(
                z_pre, dphi_pre, Lambda_sub,
                epochs=epochs_sde, lr=LR_SDE,
                batch_size=bs, print_interval=max(1, epochs_sde // 5),
                v=v_sub, d2phi=d2phi_pre, lambda_K=lk,
            )
            elapsed = time.time() - t_run

            # Evaluate trajectory fidelity
            print("  Evaluating...")
            eval_results = evaluate_pipeline(pipeline, autoencoder, sde, seed)

            rows.append({
                "N": n_train,
                "lambda_K": lk,
                "drift_loss": drift_losses[-1],
                "diff_loss": diff_losses[-1],
                **eval_results,
                "time": elapsed,
            })

    # Summary table
    print(f"\n\n{'='*90}")
    print("LOW-DATA REGIME: Does K help when the network is underconstrained?")
    print(f"{'='*90}")
    print()

    header = f"{'N':>5s}  {'lambda_K':>8s}  {'diff_loss':>10s}  {'MTE@0.1':>8s}  {'MTE@0.5':>8s}  {'MTE@1.0':>8s}  {'W2@1.0':>8s}  {'MMD@1.0':>8s}"
    print(header)
    print("-" * len(header))
    for r in rows:
        print(
            f"{r['N']:>5d}  "
            f"{r['lambda_K']:>8.2f}  "
            f"{r['diff_loss']:>10.6f}  "
            f"{r['MTE@0.1']:>8.4f}  "
            f"{r['MTE@0.5']:>8.4f}  "
            f"{r['MTE@1.0']:>8.4f}  "
            f"{r['W2@1.0']:>8.4f}  "
            f"{r['MMD@1.0']:>8.4f}"
        )

    # Per-N analysis
    print()
    for n in n_train_values:
        subset = [r for r in rows if r["N"] == n]
        mtes = {r["lambda_K"]: r["MTE@1.0"] for r in subset}
        best_lk = min(mtes, key=mtes.get)
        print(f"  N={n:>4d}: best lambda_K={best_lk:.2f} (MTE@1.0={mtes[best_lk]:.4f}), "
              f"lambda_K=0 MTE@1.0={mtes[0.0]:.4f}, "
              f"delta={((mtes[0.0] - mtes[best_lk]) / mtes[0.0] * 100):+.1f}%")

    print(f"\nTotal time: {time.time() - t0:.1f}s")
    print("Done.")


if __name__ == "__main__":
    main()
