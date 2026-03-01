"""
Diagnostic: K-weight sweep for T+F+K on sinusoidal surface.

Tests hypothesis that K weight=1.0 is too strong when combined with F.
Sweeps K weights [0.01, 0.05, 0.1, 0.3, 0.5, 1.0] at N=20 and N=200,
with T+F (no K) as baseline.

Usage:
    PYTHONUNBUFFERED=1 python -m experiments.diagnose_k_weight
"""

import time
import torch
import numpy as np
import pandas as pd

from src.numeric.datagen import sample_from_manifold
from src.numeric.datasets import DatasetBatch
from src.numeric.losses import LossWeights
from src.numeric.sde_nets import DriftNet, DiffusionNet
from src.numeric.sde_training import SDEPipelineTrainer
from src.numeric.training import MultiModelTrainer, TrainingConfig

from experiments.common import make_model_config
from experiments.data_driven_sde import (
    create_manifold_sde, evaluate_pipeline, lambdify_sde,
    TRAIN_BOUND, BATCH_SIZE, LR_AE, LR_SDE, SEED, DEVICE,
)

SURFACE = "sinusoidal"
AE_EPOCHS = 500
SDE_EPOCHS = 300
K_WEIGHTS = [0.0, 0.01, 0.05, 0.1, 0.3, 0.5, 1.0]
N_TRAINS = [20, 200]
N_TEST = 500
SEED_VAL = 42


def train_ae(train_data, n_train, lw, reg_name, seed, epochs):
    """Train autoencoder with given loss weights."""
    torch.manual_seed(seed)
    np.random.seed(seed)

    print(f"\n  AE training: {reg_name}, N={n_train}, epochs={epochs}")
    trainer = MultiModelTrainer(TrainingConfig(
        epochs=epochs,
        n_samples=n_train,
        input_dim=3,
        hidden_dim=64,
        latent_dim=2,
        learning_rate=LR_AE,
        batch_size=min(BATCH_SIZE, n_train),
        test_size=0.0,
        print_interval=max(1, epochs // 5),
        device=DEVICE,
    ))
    trainer.add_model(make_model_config("ae", lw, hidden_dims=[64, 64]))
    loader = trainer.create_data_loader(train_data)
    for epoch in range(epochs):
        losses = trainer.train_epoch(loader)
        if (epoch + 1) % max(1, epochs // 5) == 0:
            print(f"    Epoch {epoch+1}: loss={losses['ae']:.6f}")

    ae = trainer.models["ae"]
    ae.eval()
    return ae


def run_sde_pipeline(ae, x_sub, v_sub, Lambda_sub, n_train, seed):
    """Train drift and diffusion nets, return pipeline."""
    d = 2
    # Precompute decoder derivatives
    dummy = SDEPipelineTrainer(
        ae, DriftNet(d).to(DEVICE), DiffusionNet(d).to(DEVICE), device=DEVICE,
    )
    z_pre, dphi_pre, d2phi_pre = dummy.precompute_decoder_derivatives(
        x_sub.to(DEVICE),
    )

    # Drift net
    torch.manual_seed(seed + 100)
    drift_net = DriftNet(d).to(DEVICE)
    dp = SDEPipelineTrainer(
        ae, drift_net, DiffusionNet(d).to(DEVICE), device=DEVICE,
    )
    drift_losses = dp.train_stage2_precomputed(
        z_pre, dphi_pre, d2phi_pre, v_sub, Lambda_sub,
        epochs=SDE_EPOCHS, lr=LR_SDE,
        batch_size=min(BATCH_SIZE, n_train), print_interval=60,
    )
    drift_net.eval()

    # Diffusion net
    torch.manual_seed(seed + 200)
    diff_net = DiffusionNet(d).to(DEVICE)
    pipeline = SDEPipelineTrainer(ae, drift_net, diff_net, device=DEVICE)
    diff_losses = pipeline.train_stage3_precomputed(
        z_pre, dphi_pre, Lambda_sub,
        epochs=SDE_EPOCHS, lr=LR_SDE,
        batch_size=min(BATCH_SIZE, n_train), print_interval=60,
    )

    return pipeline, drift_losses[-1], diff_losses[-1]


def main():
    t0 = time.time()
    print(f"Device: {DEVICE}")
    print(f"Surface: {SURFACE}")
    print(f"K weights: {K_WEIGHTS}")
    print(f"N_train values: {N_TRAINS}")
    print(f"AE epochs: {AE_EPOCHS}, SDE epochs: {SDE_EPOCHS}")

    # Generate data once (large pool, subsample later)
    max_n = max(N_TRAINS)
    torch.manual_seed(SEED_VAL)
    np.random.seed(SEED_VAL)
    manifold_sde = create_manifold_sde(SURFACE)
    full_data = sample_from_manifold(
        manifold_sde,
        [(-TRAIN_BOUND, TRAIN_BOUND), (-TRAIN_BOUND, TRAIN_BOUND)],
        n_samples=max_n + N_TEST,
        seed=SEED_VAL,
        device=DEVICE,
    )

    # Split: first max_n for training pool, last N_TEST for held-out test
    test_x = full_data.samples[max_n:max_n + N_TEST].to(DEVICE)
    print(f"Test set: {test_x.shape[0]} points")

    # Lambdify SDE for evaluation (once)
    print("Lambdifying SDE for evaluation...")
    sde = lambdify_sde(create_manifold_sde(SURFACE))

    rows = []
    for n_train in N_TRAINS:
        # Subsample training data
        train_data = DatasetBatch(
            samples=full_data.samples[:n_train],
            local_samples=full_data.local_samples[:n_train],
            weights=full_data.weights[:n_train],
            mu=full_data.mu[:n_train],
            cov=full_data.cov[:n_train],
            p=full_data.p[:n_train],
            hessians=full_data.hessians[:n_train],
            local_cov=full_data.local_cov[:n_train] if full_data.local_cov is not None else None,
        )
        x_sub = train_data.samples.to(DEVICE)
        v_sub = train_data.mu.to(DEVICE)
        Lambda_sub = train_data.cov.to(DEVICE)

        for k_w in K_WEIGHTS:
            if k_w == 0.0:
                reg_name = "T+F"
                lw = LossWeights(tangent_bundle=1.0, diffeo=1.0)
            else:
                reg_name = f"T+F+K({k_w})"
                lw = LossWeights(tangent_bundle=1.0, diffeo=1.0, curvature=k_w)

            print(f"\n{'='*60}")
            print(f"N={n_train}, reg={reg_name}")
            print(f"{'='*60}")

            t_run = time.time()

            # Train AE
            ae = train_ae(train_data, n_train, lw, reg_name, SEED_VAL, AE_EPOCHS)

            # Reconstruction MSE on held-out test set
            with torch.no_grad():
                z_test = ae.encoder(test_x)
                x_hat = ae.decoder(z_test)
                recon_mse = ((x_hat - test_x) ** 2).sum(-1).mean().item()
            print(f"  Recon MSE (test): {recon_mse:.6f}")

            # SDE pipeline
            pipeline, drift_loss, diff_loss = run_sde_pipeline(
                ae, x_sub, v_sub, Lambda_sub, n_train, SEED_VAL,
            )

            # Evaluate trajectory fidelity
            print("  Evaluating trajectory fidelity...")
            eval_results = evaluate_pipeline(pipeline, ae, sde, SEED_VAL)

            row = {
                "N": n_train,
                "K_weight": k_w,
                "reg": reg_name,
                "recon_mse": recon_mse,
                "drift_loss": drift_loss,
                "diff_loss": diff_loss,
                **eval_results,
            }
            rows.append(row)
            print(f"  Run time: {time.time() - t_run:.1f}s")

    # Summary
    df = pd.DataFrame(rows)
    print(f"\n\n{'='*80}")
    print("K-WEIGHT SWEEP RESULTS (sinusoidal, learned projector fix)")
    print(f"{'='*80}")

    cols = ["N", "K_weight", "reg", "recon_mse", "MTE@1.0", "W2@1.0", "MMD@1.0"]
    print(df[cols].to_string(index=False, float_format="%.4f"))

    # Per N summary
    for n in N_TRAINS:
        sub = df[df["N"] == n]
        print(f"\n--- N={n} ---")
        print(sub[cols].to_string(index=False, float_format="%.4f"))

    csv_path = "diagnose_k_weight_results.csv"
    df.to_csv(csv_path, index=False)
    print(f"\nSaved {csv_path}")
    print(f"Total time: {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
