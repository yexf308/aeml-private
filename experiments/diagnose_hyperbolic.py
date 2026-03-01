"""
Diagnostic: Why does K hurt on hyperbolic_paraboloid but help on paraboloid?

Compare curvature signal strength and AE geometry quality across surfaces.

Usage:
    PYTHONUNBUFFERED=1 python -m experiments.diagnose_hyperbolic
"""

import torch
import numpy as np

from src.numeric.datagen import sample_from_manifold
from src.numeric.losses import LossWeights
from src.numeric.geometry import regularized_metric_inverse, curvature_drift_explicit_full
from src.numeric.training import MultiModelTrainer, TrainingConfig

from experiments.common import make_model_config
from experiments.data_driven_sde import (
    create_manifold_sde, TRAIN_BOUND, BATCH_SIZE, LR_AE, SEED, DEVICE,
)


def analyze_surface(surface_name, train_data, ae):
    """Analyze curvature signal strength for a trained AE."""
    x = train_data.samples.to(DEVICE)
    v = train_data.mu.to(DEVICE)
    Lambda = train_data.cov.to(DEVICE)
    p_true = train_data.p.to(DEVICE)

    with torch.no_grad():
        z = ae.encoder(x)
        dphi = ae.decoder.jacobian_network(z)
        d2phi = ae.decoder.hessian_network(z)

        # Learned geometry
        g = dphi.mT @ dphi
        ginv = regularized_metric_inverse(g)
        pinv = ginv @ dphi.mT
        P_hat = dphi @ pinv
        P_hat = 0.5 * (P_hat + P_hat.mT)

        D = dphi.shape[1]
        I_mat = torch.eye(D, device=DEVICE).unsqueeze(0)
        N_hat = I_mat - P_hat
        N_true = I_mat - p_true

        # Pull back covariance to latent
        Lambda_tan = P_hat @ Lambda @ P_hat
        Lambda_tan = 0.5 * (Lambda_tan + Lambda_tan.mT)
        Sigma_z = pinv @ Lambda_tan @ pinv.mT
        Sigma_z = 0.5 * (Sigma_z + Sigma_z.mT)

        # Ito correction
        q = curvature_drift_explicit_full(d2phi, Sigma_z)  # (B, D), halved

        # Normal components
        Nv = (N_hat @ v.unsqueeze(-1)).squeeze(-1)       # (I-P_hat)v
        Nq = (N_hat @ q.unsqueeze(-1)).squeeze(-1)       # (I-P_hat)q
        Nv_true = (N_true @ v.unsqueeze(-1)).squeeze(-1)  # (I-P_true)v

        # Signal magnitudes
        norm_v = v.norm(dim=-1)                     # ||v||
        norm_Nv = Nv.norm(dim=-1)                   # ||(I-P_hat)v||
        norm_Nq = Nq.norm(dim=-1)                   # ||(I-P_hat)q||
        norm_Nv_true = Nv_true.norm(dim=-1)         # ||(I-P_true)v||

        # Curvature target = (I-P_hat)v, model = (I-P_hat)q
        # K loss = ||(I-P_hat)(q - v)||^2
        K_residual = ((Nq - Nv) ** 2).sum(-1)

        # Reconstruction quality
        x_hat = ae.decoder(z)
        recon = ((x_hat - x) ** 2).sum(-1)

        # Projector alignment
        proj_err = ((P_hat - p_true) ** 2).sum((-1, -2))

        # Hessian norms per sample
        hess_norm = (d2phi ** 2).sum((-1, -2, -3))  # ||d2phi||_F^2

        # Eigenvalues of metric for condition number
        eigvals = torch.linalg.eigvalsh(g)
        cond = eigvals[:, -1] / eigvals[:, 0].clamp(min=1e-10)

    return {
        "||v||": norm_v.mean().item(),
        "||(I-P_hat)v||": norm_Nv.mean().item(),
        "||(I-P_hat)q||": norm_Nq.mean().item(),
        "||(I-P_true)v||": norm_Nv_true.mean().item(),
        "K_residual": K_residual.mean().item(),
        "K_residual_std": K_residual.std().item(),
        "signal_ratio": norm_Nv.mean().item() / (norm_v.mean().item() + 1e-10),
        "recon_mse": recon.mean().item(),
        "proj_err": proj_err.mean().item(),
        "||d2phi||_F": hess_norm.mean().sqrt().item(),
        "metric_cond": cond.mean().item(),
        "metric_cond_max": cond.max().item(),
    }


def main():
    surfaces = ["sinusoidal", "paraboloid", "hyperbolic_paraboloid"]
    epochs_ae = 500
    seed = SEED

    print(f"Device: {DEVICE}")
    print(f"Comparing curvature signal across surfaces\n")

    results = {}

    for surface_name in surfaces:
        print(f"\n{'='*60}")
        print(f"Surface: {surface_name}")
        print(f"{'='*60}")

        # Generate data
        torch.manual_seed(seed)
        np.random.seed(seed)
        manifold_sde = create_manifold_sde(surface_name)
        train_data = sample_from_manifold(
            manifold_sde,
            [(-TRAIN_BOUND, TRAIN_BOUND), (-TRAIN_BOUND, TRAIN_BOUND)],
            n_samples=2000, seed=seed, device=DEVICE,
        )

        # Analyze raw data: true normal drift magnitude
        x = train_data.samples.to(DEVICE)
        v = train_data.mu.to(DEVICE)
        p_true = train_data.p.to(DEVICE)
        D = x.shape[1]
        I_mat = torch.eye(D, device=DEVICE).unsqueeze(0)
        N_true = I_mat - p_true
        with torch.no_grad():
            Nv_true = (N_true @ v.unsqueeze(-1)).squeeze(-1)
            Tv_true = (p_true @ v.unsqueeze(-1)).squeeze(-1)

        print(f"\n  Raw data statistics:")
        print(f"    ||v|| mean:           {v.norm(dim=-1).mean():.6f}")
        print(f"    ||(I-P_true)v|| mean: {Nv_true.norm(dim=-1).mean():.6f}")
        print(f"    ||P_true v|| mean:    {Tv_true.norm(dim=-1).mean():.6f}")
        print(f"    Normal/Total ratio:   {Nv_true.norm(dim=-1).mean() / v.norm(dim=-1).mean():.4f}")
        print(f"    ||(I-P_true)v|| std:  {Nv_true.norm(dim=-1).std():.6f}")

        # Train T+F AE
        for reg_name, lw in [("T+F", LossWeights(tangent_bundle=1.0, diffeo=1.0)),
                              ("T+F+K", LossWeights(tangent_bundle=1.0, diffeo=1.0, curvature=0.1))]:
            print(f"\n  --- AE: {reg_name} ---")
            torch.manual_seed(seed)
            np.random.seed(seed)
            trainer = MultiModelTrainer(TrainingConfig(
                epochs=epochs_ae, n_samples=2000, input_dim=3, hidden_dim=64,
                latent_dim=2, learning_rate=LR_AE, batch_size=BATCH_SIZE,
                test_size=0.03, print_interval=max(1, epochs_ae // 5), device=DEVICE,
            ))
            trainer.add_model(make_model_config("ae", lw, hidden_dims=[64, 64]))
            loader = trainer.create_data_loader(train_data)
            for epoch in range(epochs_ae):
                losses = trainer.train_epoch(loader)
                if (epoch + 1) % max(1, epochs_ae // 5) == 0:
                    print(f"    Epoch {epoch+1}: loss={losses['ae']:.6f}")
            ae = trainer.models["ae"]
            ae.eval()

            stats = analyze_surface(surface_name, train_data, ae)
            results[(surface_name, reg_name)] = stats

            for k, val in stats.items():
                print(f"    {k}: {val:.6f}")

    # Summary comparison
    print(f"\n\n{'='*100}")
    print("CROSS-SURFACE COMPARISON")
    print(f"{'='*100}")

    metrics = ["||v||", "||(I-P_hat)v||", "||(I-P_hat)q||", "signal_ratio",
               "K_residual", "recon_mse", "proj_err", "||d2phi||_F", "metric_cond"]

    header = f"{'surface':>25s}  {'reg':>6s}  " + "  ".join(f"{m:>14s}" for m in metrics)
    print(header)
    print("-" * len(header))

    for surface_name in surfaces:
        for reg_name in ["T+F", "T+F+K"]:
            key = (surface_name, reg_name)
            if key in results:
                vals = "  ".join(f"{results[key][m]:>14.6f}" for m in metrics)
                print(f"{surface_name:>25s}  {reg_name:>6s}  {vals}")

    print("\nDone.")


if __name__ == "__main__":
    main()
