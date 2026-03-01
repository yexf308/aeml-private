"""
Diagnostic: Does covariance matching make the K identity redundant?

Hypothesis: If ambient covariance matching ||Dphi Sigma_z Dphi^T - Lambda||^2_F
already drives Sigma_z close to the true latent covariance, then the Ito
correction q computed from the learned Sigma_z will be nearly correct, and
the K residual ||(I-P)(v - q)||^2 will be ~0 regardless of lambda_K.

This script:
1. Trains a T+F AE on sinusoidal (500 epochs, seed=42)
2. Trains a shared drift_net (300 epochs)
3. Trains 4 diffusion_nets with lambda_K = 0, 0.01, 0.1, 1.0
4. For each, computes:
   a. Covariance loss: ||Dphi Sigma_z Dphi^T - Lambda||^2_F
   b. K residual with learned Sigma_z: ||(I-P)(v - q_learned)||^2
   c. K residual with true Sigma_z:    ||(I-P)(v - q_true)||^2
   d. Relative magnitude: K_residual / cov_loss

Usage:
    PYTHONUNBUFFERED=1 python -m experiments.diagnose_k_residual
"""

import copy
import time
import torch
import numpy as np

from src.numeric.datagen import sample_from_manifold
from src.numeric.losses import LossWeights
from src.numeric.sde_nets import DriftNet, DiffusionNet
from src.numeric.sde_training import SDEPipelineTrainer
from src.numeric.training import MultiModelTrainer, TrainingConfig
from src.numeric.geometry import curvature_drift_explicit_full, regularized_metric_inverse

from experiments.common import SURFACE_MAP, make_model_config
from experiments.data_driven_sde import (
    create_manifold_sde,
    TRAIN_BOUND, N_TRAIN, BATCH_SIZE, LR_AE, LR_SDE, SEED, DEVICE,
)


def compute_diagnostics(diffusion_net, z, dphi, d2phi, v, Lambda, label=""):
    """
    Compute covariance loss, K residual (learned cov), and K residual (true cov).

    All computations are done on the full dataset (no batching) with no_grad.

    Returns:
        dict with keys: cov_loss, K_residual_learned, K_residual_true, ratio
    """
    diffusion_net.eval()
    with torch.no_grad():
        B = z.shape[0]
        D = dphi.shape[1]
        d = z.shape[1]

        # --- Learned covariance ---
        sigma = diffusion_net(z)                     # (B, d, d)
        Sigma_z = sigma @ sigma.mT                   # (B, d, d)
        Sigma_z = 0.5 * (Sigma_z + Sigma_z.mT)

        # Covariance loss: ||Dphi Sigma_z Dphi^T - Lambda||^2_F
        Lambda_pred = dphi @ Sigma_z @ dphi.mT       # (B, D, D)
        cov_loss = ((Lambda_pred - Lambda) ** 2).sum((-1, -2)).mean().item()

        # --- Geometric quantities ---
        g = dphi.mT @ dphi                           # (B, d, d)
        ginv = regularized_metric_inverse(g)          # (B, d, d)
        pinv = ginv @ dphi.mT                         # (B, d, D)
        P_hat = dphi @ ginv @ dphi.mT                 # (B, D, D)
        P_hat = 0.5 * (P_hat + P_hat.mT)

        I_mat = torch.eye(D, device=z.device, dtype=z.dtype).unsqueeze(0)
        N_hat = I_mat - P_hat                         # (B, D, D)

        # --- K residual with LEARNED covariance ---
        q_learned = curvature_drift_explicit_full(d2phi, Sigma_z)  # (B, D), already halved
        normal_res_learned = (N_hat @ (v - q_learned).unsqueeze(-1)).squeeze(-1)
        K_residual_learned = (normal_res_learned ** 2).sum(-1).mean().item()

        # --- K residual with TRUE covariance ---
        # Pull back Lambda to latent space via the tangent projection
        Lambda_tan = P_hat @ Lambda @ P_hat
        Lambda_tan = 0.5 * (Lambda_tan + Lambda_tan.mT)
        Sigma_z_true = pinv @ Lambda_tan @ pinv.mT   # (B, d, d)
        Sigma_z_true = 0.5 * (Sigma_z_true + Sigma_z_true.mT)

        q_true = curvature_drift_explicit_full(d2phi, Sigma_z_true)  # (B, D)
        normal_res_true = (N_hat @ (v - q_true).unsqueeze(-1)).squeeze(-1)
        K_residual_true = (normal_res_true ** 2).sum(-1).mean().item()

        # --- Extra diagnostics ---
        # How close is learned Sigma_z to true Sigma_z?
        sigma_z_err = ((Sigma_z - Sigma_z_true) ** 2).sum((-1, -2)).mean().item()

        # Per-sample K residual stats (learned)
        per_sample_K = (normal_res_learned ** 2).sum(-1)  # (B,)
        K_max = per_sample_K.max().item()
        K_median = per_sample_K.median().item()

        # Normal component of v itself (for reference)
        normal_v = (N_hat @ v.unsqueeze(-1)).squeeze(-1)
        normal_v_norm = (normal_v ** 2).sum(-1).mean().item()

        # Normal component of q_learned
        normal_q = (N_hat @ q_learned.unsqueeze(-1)).squeeze(-1)
        normal_q_norm = (normal_q ** 2).sum(-1).mean().item()

    ratio = K_residual_learned / max(cov_loss, 1e-15)

    return {
        "cov_loss": cov_loss,
        "K_residual_learned": K_residual_learned,
        "K_residual_true": K_residual_true,
        "ratio": ratio,
        "sigma_z_err": sigma_z_err,
        "K_max": K_max,
        "K_median": K_median,
        "||N(v)||^2": normal_v_norm,
        "||N(q)||^2": normal_q_norm,
    }


def main():
    surface_name = "sinusoidal"
    epochs_ae = 500
    epochs_drift = 300
    epochs_diff = 300
    lambda_K_values = [0.0, 0.01, 0.1, 1.0]
    seed = SEED

    print(f"Device: {DEVICE}")
    print(f"Surface: {surface_name}")
    print(f"AE epochs: {epochs_ae}, Drift epochs: {epochs_drift}, Diff epochs: {epochs_diff}")
    print(f"lambda_K values: {lambda_K_values}")
    print(f"N_TRAIN={N_TRAIN}, BATCH_SIZE={BATCH_SIZE}, seed={seed}")

    t0 = time.time()

    # ====== Generate training data ======
    print("\n--- Generating training data ---")
    torch.manual_seed(seed)
    np.random.seed(seed)
    manifold_sde = create_manifold_sde(surface_name)
    train_data = sample_from_manifold(
        manifold_sde,
        [(-TRAIN_BOUND, TRAIN_BOUND), (-TRAIN_BOUND, TRAIN_BOUND)],
        n_samples=N_TRAIN, seed=seed, device=DEVICE,
    )
    x = train_data.samples.to(DEVICE)
    v = train_data.mu.to(DEVICE)
    Lambda = train_data.cov.to(DEVICE)
    print(f"  x: {x.shape}, v: {v.shape}, Lambda: {Lambda.shape}")

    # ====== Stage 1: Train T+F autoencoder ======
    print(f"\n--- Stage 1: Autoencoder (T+F, {epochs_ae} epochs) ---")
    torch.manual_seed(seed)
    np.random.seed(seed)
    lw = LossWeights(tangent_bundle=1.0, diffeo=1.0)
    trainer = MultiModelTrainer(TrainingConfig(
        epochs=epochs_ae,
        n_samples=N_TRAIN,
        input_dim=3,
        hidden_dim=64,
        latent_dim=2,
        learning_rate=LR_AE,
        batch_size=BATCH_SIZE,
        test_size=0.03,
        print_interval=max(1, epochs_ae // 5),
        device=DEVICE,
    ))
    trainer.add_model(make_model_config("ae", lw, hidden_dims=[64, 64]))
    data_loader = trainer.create_data_loader(train_data)
    for epoch in range(epochs_ae):
        losses = trainer.train_epoch(data_loader)
        if (epoch + 1) % max(1, epochs_ae // 5) == 0:
            print(f"  Epoch {epoch+1}: loss={losses['ae']:.6f}")
    autoencoder = trainer.models["ae"]
    autoencoder.eval()
    print(f"  Final AE loss: {losses['ae']:.6f}")

    # ====== Precompute decoder derivatives ======
    print("\n--- Precomputing decoder Jacobians/Hessians ---")
    t_pre = time.time()
    d = 2
    dummy = SDEPipelineTrainer(
        autoencoder, DriftNet(d).to(DEVICE), DiffusionNet(d).to(DEVICE), device=DEVICE,
    )
    z_pre, dphi_pre, d2phi_pre = dummy.precompute_decoder_derivatives(x)
    print(f"  z: {z_pre.shape}, dphi: {dphi_pre.shape}, d2phi: {d2phi_pre.shape}")
    print(f"  Precomputed in {time.time() - t_pre:.1f}s")

    # ====== Stage 2: Train shared drift_net ======
    print(f"\n--- Stage 2: Drift net ({epochs_drift} epochs) ---")
    torch.manual_seed(seed + 100)
    drift_net = DriftNet(d).to(DEVICE)
    drift_pipeline = SDEPipelineTrainer(
        autoencoder, drift_net, DiffusionNet(d).to(DEVICE), device=DEVICE,
    )
    drift_losses = drift_pipeline.train_stage2_precomputed(
        z_pre, dphi_pre, d2phi_pre, v, Lambda,
        epochs=epochs_drift, lr=LR_SDE,
        batch_size=BATCH_SIZE, print_interval=max(1, epochs_drift // 5),
    )
    drift_net.eval()
    print(f"  Final drift loss: {drift_losses[-1]:.6f}")

    # ====== Compute baseline K residual (before any diffusion training) ======
    # Use an untrained diffusion_net to see what the K residual looks like
    print("\n--- Baseline K residual (untrained diffusion_net) ---")
    torch.manual_seed(seed + 200)
    untrained_diff = DiffusionNet(d).to(DEVICE)
    baseline_diag = compute_diagnostics(
        untrained_diff, z_pre, dphi_pre, d2phi_pre, v, Lambda, label="untrained",
    )
    print(f"  cov_loss={baseline_diag['cov_loss']:.6f}")
    print(f"  K_residual_learned={baseline_diag['K_residual_learned']:.6f}")
    print(f"  K_residual_true={baseline_diag['K_residual_true']:.6f}")
    print(f"  ||N(v)||^2={baseline_diag['||N(v)||^2']:.6f}")

    # ====== Stage 3: Train diffusion_nets with varying lambda_K ======
    results = []
    for lk in lambda_K_values:
        print(f"\n{'='*60}")
        print(f"Stage 3: Diffusion net (lambda_K={lk}, {epochs_diff} epochs)")
        print(f"{'='*60}")

        # Same initialization for fair comparison
        torch.manual_seed(seed + 200)
        diffusion_net = DiffusionNet(d).to(DEVICE)
        pipeline = SDEPipelineTrainer(
            autoencoder, drift_net, diffusion_net, device=DEVICE,
        )

        t_run = time.time()
        diff_losses = pipeline.train_stage3_precomputed(
            z_pre, dphi_pre, Lambda,
            epochs=epochs_diff, lr=LR_SDE,
            batch_size=BATCH_SIZE, print_interval=max(1, epochs_diff // 5),
            v=v, d2phi=d2phi_pre, lambda_K=lk,
        )
        elapsed = time.time() - t_run
        print(f"  Training time: {elapsed:.1f}s")
        print(f"  Final training loss: {diff_losses[-1]:.6f}")

        # Compute diagnostics on full training set
        diag = compute_diagnostics(
            diffusion_net, z_pre, dphi_pre, d2phi_pre, v, Lambda,
            label=f"lambda_K={lk}",
        )
        diag["lambda_K"] = lk
        diag["train_loss"] = diff_losses[-1]
        diag["time"] = elapsed
        results.append(diag)

        print(f"  cov_loss           = {diag['cov_loss']:.8f}")
        print(f"  K_residual_learned = {diag['K_residual_learned']:.8f}")
        print(f"  K_residual_true    = {diag['K_residual_true']:.8f}")
        print(f"  ratio (K/cov)      = {diag['ratio']:.6f}")
        print(f"  sigma_z_err        = {diag['sigma_z_err']:.8f}")
        print(f"  K_max              = {diag['K_max']:.8f}")
        print(f"  K_median           = {diag['K_median']:.8f}")
        print(f"  ||N(v)||^2         = {diag['||N(v)||^2']:.8f}")
        print(f"  ||N(q)||^2         = {diag['||N(q)||^2']:.8f}")

    # ====== Summary table ======
    print(f"\n\n{'='*100}")
    print("DIAGNOSTIC SUMMARY: Does covariance matching make K identity redundant?")
    print(f"{'='*100}")
    print()

    # Header
    header = (
        f"{'lambda_K':>10s}  "
        f"{'cov_loss':>12s}  "
        f"{'K_res_learn':>12s}  "
        f"{'K_res_true':>12s}  "
        f"{'K/cov ratio':>12s}  "
        f"{'Sigma_z_err':>12s}  "
        f"{'K_max':>12s}  "
        f"{'K_median':>12s}  "
        f"{'||N(v)||^2':>12s}  "
        f"{'||N(q)||^2':>12s}"
    )
    print(header)
    print("-" * len(header))

    for r in results:
        print(
            f"{r['lambda_K']:>10.3f}  "
            f"{r['cov_loss']:>12.8f}  "
            f"{r['K_residual_learned']:>12.8f}  "
            f"{r['K_residual_true']:>12.8f}  "
            f"{r['ratio']:>12.6f}  "
            f"{r['sigma_z_err']:>12.8f}  "
            f"{r['K_max']:>12.8f}  "
            f"{r['K_median']:>12.8f}  "
            f"{r['||N(v)||^2']:>12.8f}  "
            f"{r['||N(q)||^2']:>12.8f}"
        )

    print()
    print("INTERPRETATION:")
    print("  - If K_res_learn is already ~0 (or ~= K_res_true) at lambda_K=0,")
    print("    then covariance matching alone satisfies the K identity.")
    print("  - If K_res_learn >> K_res_true, covariance matching leaves room")
    print("    for K penalty to help, but it doesn't change because Sigma_z")
    print("    is already close to truth.")
    print("  - K_res_true is the irreducible floor: the K residual you'd get")
    print("    even with the exact covariance, due to AE approximation error.")
    print()

    # Key comparison
    r0 = results[0]  # lambda_K = 0
    r3 = results[-1]  # lambda_K = 1.0
    print("KEY COMPARISON:")
    print(f"  lambda_K=0 -> K_residual_learned = {r0['K_residual_learned']:.8f}")
    print(f"  lambda_K=1 -> K_residual_learned = {r3['K_residual_learned']:.8f}")
    print(f"  True covariance floor:             {r0['K_residual_true']:.8f}")
    print(f"  ||N(v)||^2 (reference):            {r0['||N(v)||^2']:.8f}")
    print()

    if r0['K_residual_learned'] < 1e-4 and r0['K_residual_true'] < 1e-4:
        print("CONCLUSION: K residual is near zero even at lambda_K=0.")
        print("  -> Covariance matching makes the K penalty REDUNDANT.")
    elif abs(r0['K_residual_learned'] - r0['K_residual_true']) / max(r0['K_residual_true'], 1e-15) < 0.5:
        print("CONCLUSION: K residual (learned) is close to K residual (true cov).")
        print("  -> The K residual is dominated by AE approximation error,")
        print("     not by diffusion estimation error. K penalty can't help much.")
    elif r3['K_residual_learned'] < r0['K_residual_learned'] * 0.5:
        print("CONCLUSION: lambda_K=1 significantly reduces K residual vs lambda_K=0.")
        print("  -> K penalty is NOT redundant; it provides additional signal.")
    else:
        print("CONCLUSION: lambda_K has minimal effect on K residual.")
        print("  -> Likely redundant with covariance matching.")

    print(f"\nTotal time: {time.time() - t0:.1f}s")
    print("Done.")


if __name__ == "__main__":
    main()
