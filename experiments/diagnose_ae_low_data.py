"""
Diagnostic: Does K help manifold construction (AE training) in low-data regime?

Train AEs with T, T+F, T+K, T+F+K on N=20, 50, 200 points.
Evaluate: reconstruction error on held-out test set + downstream SDE trajectory quality.

Usage:
    PYTHONUNBUFFERED=1 python -m experiments.diagnose_ae_low_data
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


REG_CONFIGS = {
    "T":     LossWeights(tangent_bundle=1.0),
    "T+K":   LossWeights(tangent_bundle=1.0, curvature=1.0),
    "T+F":   LossWeights(tangent_bundle=1.0, diffeo=1.0),
    "T+F+K": LossWeights(tangent_bundle=1.0, diffeo=1.0, curvature=1.0),
}


def train_ae(train_data, n_train, epochs, lw, name, seed):
    """Train AE on given data."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    bs = min(BATCH_SIZE, n_train)
    trainer = MultiModelTrainer(TrainingConfig(
        epochs=epochs, n_samples=n_train, input_dim=3, hidden_dim=64,
        latent_dim=2, learning_rate=LR_AE, batch_size=bs,
        test_size=0.0, print_interval=max(1, epochs // 5), device=DEVICE,
    ))
    trainer.add_model(make_model_config(name, lw, hidden_dims=[64, 64]))
    loader = trainer.create_data_loader(train_data)
    for epoch in range(epochs):
        losses = trainer.train_epoch(loader)
        if (epoch + 1) % max(1, epochs // 5) == 0:
            print(f"    Epoch {epoch+1}: loss={losses[name]:.6f}")
    ae = trainer.models[name]
    ae.eval()
    return ae, losses[name]


def eval_recon(ae, test_x):
    """Compute reconstruction MSE on test set."""
    with torch.no_grad():
        z = ae.encoder(test_x)
        x_hat = ae.decoder(z)
        mse = ((x_hat - test_x) ** 2).sum(-1).mean().item()
    return mse


def eval_geometry(ae, test_x):
    """Evaluate geometric quality: metric condition number, tangent accuracy."""
    from src.numeric.geometry import regularized_metric_inverse
    with torch.no_grad():
        z = ae.encoder(test_x)
        dphi = ae.decoder.jacobian_network(z)  # (B, 3, 2)
        g = dphi.mT @ dphi  # (B, 2, 2)

        # Condition number of metric tensor
        eigvals = torch.linalg.eigvalsh(g)  # (B, 2)
        cond = (eigvals[:, -1] / eigvals[:, 0].clamp(min=1e-10))
        mean_cond = cond.mean().item()
        max_cond = cond.max().item()

        # Projector quality: P = dphi ginv dphi^T should be rank-2 projector
        ginv = regularized_metric_inverse(g)
        P = dphi @ ginv @ dphi.mT  # (B, 3, 3)
        P = 0.5 * (P + P.mT)
        # Idempotency error: ||P^2 - P||_F
        P2_err = ((P @ P - P) ** 2).sum((-1, -2)).mean().item()

    return {
        "mean_cond": mean_cond,
        "max_cond": max_cond,
        "proj_err": P2_err,
    }


def main():
    surface_name = "sinusoidal"
    epochs_ae = 500
    epochs_sde = 300
    seed = SEED
    d = 2

    n_train_values = [20, 50, 200]

    print(f"Device: {DEVICE}")
    print(f"Surface: {surface_name}")
    print(f"N_TRAIN sweep: {n_train_values}")
    print(f"AE regs: {list(REG_CONFIGS.keys())}")

    t0 = time.time()

    # Generate full dataset — use for subsampling and test set
    print("\n--- Generating data ---")
    torch.manual_seed(seed)
    np.random.seed(seed)
    manifold_sde = create_manifold_sde(surface_name)
    full_data = sample_from_manifold(
        manifold_sde,
        [(-TRAIN_BOUND, TRAIN_BOUND), (-TRAIN_BOUND, TRAIN_BOUND)],
        n_samples=2000, seed=seed, device=DEVICE,
    )
    x_full = full_data.samples.to(DEVICE)
    v_full = full_data.mu.to(DEVICE)
    Lambda_full = full_data.cov.to(DEVICE)

    # Held-out test set (last 500 points)
    test_x = x_full[1500:].to(DEVICE)

    sde = lambdify_sde(create_manifold_sde(surface_name))

    rows = []

    for n_train in n_train_values:
        print(f"\n{'='*70}")
        print(f"N_TRAIN = {n_train}")
        print(f"{'='*70}")

        # Subsample
        torch.manual_seed(seed)
        idx = torch.randperm(1500)[:n_train]  # don't overlap with test
        x_sub = x_full[idx]
        v_sub = v_full[idx]
        Lambda_sub = Lambda_full[idx]

        # Create sub-DatasetBatch by indexing all fields
        from src.numeric.datasets import DatasetBatch
        sub_data = DatasetBatch(
            samples=full_data.samples[idx],
            local_samples=full_data.local_samples[idx],
            mu=full_data.mu[idx],
            cov=full_data.cov[idx],
            p=full_data.p[idx],
            weights=full_data.weights[idx],
            hessians=full_data.hessians[idx],
            local_cov=full_data.local_cov[idx] if full_data.local_cov is not None else None,
        )

        for reg_name, lw in REG_CONFIGS.items():
            print(f"\n  --- AE reg: {reg_name} ---")
            t_run = time.time()

            # Stage 1: Train AE on small dataset
            ae, ae_loss = train_ae(sub_data, n_train, epochs_ae, lw, "ae", seed)

            # Evaluate AE quality
            recon_mse = eval_recon(ae, test_x)
            geom = eval_geometry(ae, test_x)
            print(f"    Recon MSE (test): {recon_mse:.6f}")
            print(f"    Metric cond (mean/max): {geom['mean_cond']:.2f} / {geom['max_cond']:.2f}")
            print(f"    Projector err: {geom['proj_err']:.8f}")

            # Stage 2+3: SDE training on same small dataset
            dummy = SDEPipelineTrainer(
                ae, DriftNet(d).to(DEVICE), DiffusionNet(d).to(DEVICE), device=DEVICE,
            )
            z_pre, dphi_pre, d2phi_pre = dummy.precompute_decoder_derivatives(x_sub.to(DEVICE))

            torch.manual_seed(seed + 100)
            drift_net = DriftNet(d).to(DEVICE)
            bs = min(BATCH_SIZE, n_train)
            drift_pipeline = SDEPipelineTrainer(
                ae, drift_net, DiffusionNet(d).to(DEVICE), device=DEVICE,
            )
            drift_losses = drift_pipeline.train_stage2_precomputed(
                z_pre, dphi_pre, d2phi_pre, v_sub.to(DEVICE), Lambda_sub.to(DEVICE),
                epochs=epochs_sde, lr=LR_SDE,
                batch_size=bs, print_interval=max(1, epochs_sde // 5),
            )
            drift_net.eval()

            torch.manual_seed(seed + 200)
            diffusion_net = DiffusionNet(d).to(DEVICE)
            pipeline = SDEPipelineTrainer(
                ae, drift_net, diffusion_net, device=DEVICE,
            )
            diff_losses = pipeline.train_stage3_precomputed(
                z_pre, dphi_pre, Lambda_sub.to(DEVICE),
                epochs=epochs_sde, lr=LR_SDE,
                batch_size=bs, print_interval=max(1, epochs_sde // 5),
            )

            print("    Evaluating trajectories...")
            eval_results = evaluate_pipeline(pipeline, ae, sde, seed)

            rows.append({
                "N": n_train,
                "ae_reg": reg_name,
                "ae_loss": ae_loss,
                "recon_mse": recon_mse,
                "mean_cond": geom["mean_cond"],
                "max_cond": geom["max_cond"],
                "proj_err": geom["proj_err"],
                "drift_loss": drift_losses[-1],
                "diff_loss": diff_losses[-1],
                **eval_results,
                "time": time.time() - t_run,
            })

    # Summary table
    print(f"\n\n{'='*120}")
    print("AE LOW-DATA REGIME: Does K help manifold construction?")
    print(f"{'='*120}")

    header = (
        f"{'N':>4s}  {'reg':>5s}  {'recon_mse':>10s}  {'mean_cond':>10s}  "
        f"{'proj_err':>10s}  {'MTE@0.1':>8s}  {'MTE@0.5':>8s}  "
        f"{'MTE@1.0':>8s}  {'W2@1.0':>8s}  {'MMD@1.0':>8s}"
    )
    print(header)
    print("-" * len(header))
    for r in rows:
        print(
            f"{r['N']:>4d}  {r['ae_reg']:>5s}  {r['recon_mse']:>10.6f}  "
            f"{r['mean_cond']:>10.2f}  {r['proj_err']:>10.8f}  "
            f"{r['MTE@0.1']:>8.4f}  {r['MTE@0.5']:>8.4f}  "
            f"{r['MTE@1.0']:>8.4f}  {r['W2@1.0']:>8.4f}  {r['MMD@1.0']:>8.4f}"
        )

    # Per-N best reg
    print()
    for n in n_train_values:
        subset = [r for r in rows if r["N"] == n]
        print(f"  N={n}:")
        for metric in ["recon_mse", "MTE@1.0", "W2@1.0"]:
            vals = {r["ae_reg"]: r[metric] for r in subset}
            best = min(vals, key=vals.get)
            print(f"    Best {metric}: {best} = {vals[best]:.4f}  "
                  f"(T={vals.get('T', float('nan')):.4f}, T+K={vals.get('T+K', float('nan')):.4f}, "
                  f"T+F={vals.get('T+F', float('nan')):.4f}, T+F+K={vals.get('T+F+K', float('nan')):.4f})")

    print(f"\nTotal time: {time.time() - t0:.1f}s")
    print("Done.")


if __name__ == "__main__":
    main()
