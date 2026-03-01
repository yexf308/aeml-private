"""
Diagnose K signal strength relative to covariance loss in diffusion net training.

Hypothesis 3: K penalty ||(I-P)(v-q)||^2 is orders of magnitude smaller than
covariance loss ||Dphi Sigma Dphi^T - Lambda||^2_F, so even lambda_K=1.0
doesn't meaningfully shift gradients.

Diagnostics:
  Part A: For a freshly initialized (untrained) diffusion net, and at checkpoints
          during training (epoch 50, 150, 300), compute:
            - cov_loss value
            - K_loss value (unweighted)
            - gradient norm from cov_loss alone
            - gradient norm from K_loss alone
  Part B: Train with very large lambda_K values (10, 100, 1000) and report
          MTE@1.0 and W2@1.0.

Usage:
    PYTHONUNBUFFERED=1 python -m experiments.diagnose_k_signal
"""

import copy
import time
import torch
import numpy as np

from src.numeric.datagen import sample_from_manifold
from src.numeric.losses import LossWeights
from src.numeric.sde_nets import DriftNet, DiffusionNet
from src.numeric.sde_training import SDEPipelineTrainer
from src.numeric.geometry import curvature_drift_explicit_full, regularized_metric_inverse
from src.numeric.training import MultiModelTrainer, TrainingConfig

from experiments.common import SURFACE_MAP, make_model_config
from experiments.data_driven_sde import (
    create_manifold_sde,
    evaluate_pipeline,
    DEVICE, TRAIN_BOUND, N_TRAIN, BATCH_SIZE, LR_AE, LR_SDE, SEED,
    N_TRAJ, T_MAX, DT, N_STEPS, BOUNDARY,
)
from experiments.trajectory_fidelity_study import lambdify_sde


def compute_grad_diagnostics(diffusion_net, z, dphi, d2phi, v, Lambda):
    """Compute cov_loss, K_loss, and their separate gradient norms.

    Returns dict with: cov_loss, K_loss, cov_grad_norm, K_grad_norm, ratio.
    """
    diffusion_net.train()

    # Forward pass (shared)
    sigma = diffusion_net(z)
    Sigma_z = sigma @ sigma.mT
    Sigma_z = 0.5 * (Sigma_z + Sigma_z.mT)

    # --- Covariance loss ---
    Lambda_pred = dphi @ Sigma_z @ dphi.mT
    cov_loss = ((Lambda_pred - Lambda) ** 2).sum((-1, -2)).mean()

    # --- K loss ---
    g = dphi.mT @ dphi
    ginv = regularized_metric_inverse(g)
    P_hat = dphi @ ginv @ dphi.mT
    P_hat = 0.5 * (P_hat + P_hat.mT)

    q = curvature_drift_explicit_full(d2phi, Sigma_z)
    D = dphi.shape[1]
    I_mat = torch.eye(D, device=z.device, dtype=z.dtype).unsqueeze(0)
    N_hat = I_mat - P_hat
    normal_res = (N_hat @ (v - q).unsqueeze(-1)).squeeze(-1)
    K_loss = (normal_res ** 2).sum(-1).mean()

    # --- Gradient norms ---
    # Create a temporary optimizer just for zero_grad convenience
    params = list(diffusion_net.parameters())

    # Cov gradient norm
    for p in params:
        if p.grad is not None:
            p.grad.zero_()
    cov_loss.backward(retain_graph=True)
    cov_grad_norm = sum(
        p.grad.norm().item() ** 2 for p in params if p.grad is not None
    ) ** 0.5

    # K gradient norm (reset grads first)
    for p in params:
        if p.grad is not None:
            p.grad.zero_()
    K_loss.backward()
    K_grad_norm = sum(
        p.grad.norm().item() ** 2 for p in params if p.grad is not None
    ) ** 0.5

    # Clean up
    for p in params:
        if p.grad is not None:
            p.grad.zero_()

    cov_val = cov_loss.item()
    K_val = K_loss.item()
    ratio = cov_val / max(K_val, 1e-30)
    grad_ratio = cov_grad_norm / max(K_grad_norm, 1e-30)

    return {
        "cov_loss": cov_val,
        "K_loss": K_val,
        "loss_ratio": ratio,
        "cov_grad_norm": cov_grad_norm,
        "K_grad_norm": K_grad_norm,
        "grad_ratio": grad_ratio,
    }


def train_autoencoder(train_data, epochs_ae, ae_loss_weights, ae_reg_name, seed):
    """Stage 1: Train autoencoder with given regularization."""
    torch.manual_seed(seed)
    np.random.seed(seed)

    print(f"\n--- Stage 1: Autoencoder (recon + {ae_reg_name}, {epochs_ae} epochs) ---")
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
    trainer.add_model(make_model_config("ae", ae_loss_weights, hidden_dims=[64, 64]))
    data_loader = trainer.create_data_loader(train_data)
    for epoch in range(epochs_ae):
        losses = trainer.train_epoch(data_loader)
        if (epoch + 1) % max(1, epochs_ae // 5) == 0:
            print(f"  Epoch {epoch+1}: loss={losses['ae']:.6f}")

    autoencoder = trainer.models["ae"]
    autoencoder.eval()
    return autoencoder


def main():
    t_start = time.time()
    surface_name = "sinusoidal"
    seed = SEED  # 42
    epochs_ae = 500
    epochs_drift = 300
    epochs_diff = 300
    d = 2

    print(f"Device: {DEVICE}")
    print(f"Surface: {surface_name}")
    print(f"Seed: {seed}")
    print(f"AE epochs: {epochs_ae}, Drift epochs: {epochs_drift}, Diff epochs: {epochs_diff}")

    # ====== Generate data (once) ======
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

    # ====== Stage 1: Train T+F AE ======
    ae_loss_weights = LossWeights(tangent_bundle=1.0, diffeo=1.0)
    autoencoder = train_autoencoder(train_data, epochs_ae, ae_loss_weights, "T+F", seed)

    # ====== Precompute decoder derivatives ======
    print("\n--- Precomputing decoder Jacobians/Hessians ---")
    t_pre = time.time()
    dummy = SDEPipelineTrainer(
        autoencoder, DriftNet(d).to(DEVICE), DiffusionNet(d).to(DEVICE), device=DEVICE,
    )
    z_all, dphi_all, d2phi_all = dummy.precompute_decoder_derivatives(x)
    print(f"  Precomputed in {time.time() - t_pre:.1f}s")

    # ====== Stage 2: Train shared drift_net ======
    print(f"\n--- Stage 2: Drift net ({epochs_drift} epochs) ---")
    torch.manual_seed(seed + 100)
    drift_net = DriftNet(d).to(DEVICE)
    drift_pipeline = SDEPipelineTrainer(
        autoencoder, drift_net, DiffusionNet(d).to(DEVICE), device=DEVICE,
    )
    drift_losses = drift_pipeline.train_stage2_precomputed(
        z_all, dphi_all, d2phi_all, v, Lambda,
        epochs=epochs_drift, lr=LR_SDE,
        batch_size=BATCH_SIZE, print_interval=max(1, epochs_drift // 5),
    )
    drift_net.eval()
    print(f"  Final drift loss: {drift_losses[-1]:.6f}")

    # ====================================================================
    # PART A: Gradient diagnostic at initialization and during training
    # ====================================================================
    print("\n" + "=" * 70)
    print("PART A: GRADIENT DIAGNOSTICS (cov_loss vs K_loss)")
    print("=" * 70)

    # Use a subsample for diagnostics (full dataset to get good statistics)
    n_diag = min(512, len(z_all))
    idx = torch.randperm(len(z_all))[:n_diag]
    z_diag = z_all[idx]
    dphi_diag = dphi_all[idx]
    d2phi_diag = d2phi_all[idx]
    v_diag = v[idx]
    Lambda_diag = Lambda[idx]

    # --- Freshly initialized diffusion net ---
    print("\n--- Freshly initialized diffusion net (epoch 0) ---")
    torch.manual_seed(seed + 200)
    diffusion_net_diag = DiffusionNet(d).to(DEVICE)
    diag_0 = compute_grad_diagnostics(
        diffusion_net_diag, z_diag, dphi_diag, d2phi_diag, v_diag, Lambda_diag,
    )
    print(f"  cov_loss:       {diag_0['cov_loss']:.6e}")
    print(f"  K_loss:         {diag_0['K_loss']:.6e}")
    print(f"  loss ratio:     {diag_0['loss_ratio']:.2f}x")
    print(f"  cov_grad_norm:  {diag_0['cov_grad_norm']:.6e}")
    print(f"  K_grad_norm:    {diag_0['K_grad_norm']:.6e}")
    print(f"  grad ratio:     {diag_0['grad_ratio']:.2f}x")

    # --- Train with lambda_K=0.1 and checkpoint at epoch 50, 150, 300 ---
    checkpoint_epochs = [50, 150, 300]
    print(f"\n--- Training diffusion net with lambda_K=0.1 (checkpointing at {checkpoint_epochs}) ---")
    torch.manual_seed(seed + 200)
    diffusion_net_train = DiffusionNet(d).to(DEVICE)
    optimizer = torch.optim.Adam(diffusion_net_train.parameters(), lr=LR_SDE)

    from torch.utils.data import DataLoader, TensorDataset
    dataset = TensorDataset(z_all, dphi_all, d2phi_all, v, Lambda)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    lambda_K_train = 0.1
    next_ckpt_idx = 0

    from src.numeric.sde_losses import ambient_diffusion_loss

    for epoch in range(epochs_diff):
        epoch_loss = 0.0
        n_batches = 0
        for z_b, dphi_b, d2phi_b, v_b, Lambda_b in loader:
            loss = ambient_diffusion_loss(
                diffusion_net_train, z_b, Lambda_b, dphi=dphi_b,
                v=v_b, d2phi=d2phi_b, lambda_K=lambda_K_train,
            )
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            n_batches += 1
        avg_loss = epoch_loss / max(n_batches, 1)

        if (epoch + 1) % 60 == 0:
            print(f"  Epoch {epoch+1}: loss={avg_loss:.6e}")

        # Checkpoint diagnostics
        if next_ckpt_idx < len(checkpoint_epochs) and (epoch + 1) == checkpoint_epochs[next_ckpt_idx]:
            print(f"\n  --- Checkpoint at epoch {epoch+1} (lambda_K=0.1 training) ---")
            diag_ckpt = compute_grad_diagnostics(
                diffusion_net_train, z_diag, dphi_diag, d2phi_diag, v_diag, Lambda_diag,
            )
            print(f"    total_train_loss: {avg_loss:.6e}")
            print(f"    cov_loss:         {diag_ckpt['cov_loss']:.6e}")
            print(f"    K_loss:           {diag_ckpt['K_loss']:.6e}")
            print(f"    loss ratio:       {diag_ckpt['loss_ratio']:.2f}x")
            print(f"    cov_grad_norm:    {diag_ckpt['cov_grad_norm']:.6e}")
            print(f"    K_grad_norm:      {diag_ckpt['K_grad_norm']:.6e}")
            print(f"    grad ratio:       {diag_ckpt['grad_ratio']:.2f}x")
            next_ckpt_idx += 1

    # ====================================================================
    # PART B: Extreme lambda_K sweep — does it move the needle?
    # ====================================================================
    print("\n" + "=" * 70)
    print("PART B: EXTREME LAMBDA_K SWEEP")
    print("=" * 70)

    lambda_K_values = [0.0, 0.1, 1.0, 10.0, 100.0, 1000.0]

    # Lambdify SDE for evaluation (once)
    print("--- Lambdifying SDE for evaluation ---")
    sde = lambdify_sde(create_manifold_sde(surface_name))

    results = []
    for lk in lambda_K_values:
        print(f"\n{'='*50}")
        print(f"lambda_K = {lk}")
        print(f"{'='*50}")

        torch.manual_seed(seed + 200)
        diffusion_net = DiffusionNet(d).to(DEVICE)
        pipeline = SDEPipelineTrainer(
            autoencoder, drift_net, diffusion_net, device=DEVICE,
        )
        t_run = time.time()

        diff_losses = pipeline.train_stage3_precomputed(
            z_all, dphi_all, Lambda,
            epochs=epochs_diff, lr=LR_SDE,
            batch_size=BATCH_SIZE, print_interval=max(1, epochs_diff // 5),
            v=v, d2phi=d2phi_all, lambda_K=lk,
        )

        # Evaluate
        print("  Evaluating trajectory fidelity...")
        eval_results = evaluate_pipeline(pipeline, autoencoder, sde, seed)

        # Also compute final cov_loss and K_loss separately for reporting
        final_diag = compute_grad_diagnostics(
            diffusion_net, z_diag, dphi_diag, d2phi_diag, v_diag, Lambda_diag,
        )

        results.append({
            "lambda_K": lk,
            "final_train_loss": diff_losses[-1],
            "final_cov_loss": final_diag["cov_loss"],
            "final_K_loss": final_diag["K_loss"],
            "MTE@1.0": eval_results["MTE@1.0"],
            "W2@1.0": eval_results["W2@1.0"],
            "MMD@1.0": eval_results["MMD@1.0"],
            "time": time.time() - t_run,
        })
        print(f"  Run time: {time.time() - t_run:.1f}s")

    # ====================================================================
    # Summary
    # ====================================================================
    print("\n" + "=" * 70)
    print("SUMMARY: EXTREME LAMBDA_K SWEEP")
    print("=" * 70)
    header = f"{'lambda_K':>10} {'train_loss':>12} {'cov_loss':>12} {'K_loss':>12} {'MTE@1.0':>10} {'W2@1.0':>10} {'MMD@1.0':>10}"
    print(header)
    print("-" * len(header))
    for r in results:
        print(
            f"{r['lambda_K']:>10.1f} "
            f"{r['final_train_loss']:>12.6e} "
            f"{r['final_cov_loss']:>12.6e} "
            f"{r['final_K_loss']:>12.6e} "
            f"{r['MTE@1.0']:>10.4f} "
            f"{r['W2@1.0']:>10.4f} "
            f"{r['MMD@1.0']:>10.4f}"
        )

    print(f"\nTotal time: {time.time() - t_start:.1f}s")
    print("\nDone.")


if __name__ == "__main__":
    main()
