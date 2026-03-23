"""
Test drift smoothness regularization across ambient dimensions D.

Key question: does metric-weighted smoothness outperform Euclidean at high D
where chart anisotropy is larger?

Design:
  D ∈ {3, 11, 201}
  N = 20 (sparse regime)
  Conditions: baseline, smooth-euc, smooth-metric
  Surface: paraboloid
  5 seeds

For D=3: uses standard ManifoldSDE pipeline
For D>3: uses FourierAugmentedSurface (highd_manifolds)

Usage:
    # Smoke test
    python -u -m experiments.smoothness_across_D --n-seeds 1 --epochs 50 --sde-epochs 50 --d-values 3 11

    # Full run
    python -u -m experiments.smoothness_across_D --n-seeds 5
"""
import argparse
import copy
import time

import numpy as np
import pandas as pd
import torch
from scipy import stats

from src.numeric.geometry import (
    curvature_drift_explicit_full,
    regularized_metric_inverse,
)
from src.numeric.losses import LossWeights
from src.numeric.sde_nets import DriftNet, DiffusionNet
from src.numeric.sde_training import SDEPipelineTrainer
from src.numeric.training import MultiModelTrainer, TrainingConfig

from experiments.common import make_model_config
from experiments.data_driven_sde import (
    DEVICE, TRAIN_BOUND, BOUNDARY, LR_AE, LR_SDE, BATCH_SIZE,
    evaluate_pipeline,
)

SURFACE = "paraboloid"
N_TRAIN = 20
LAMBDA_SMOOTH = 0.5
AUG_SIGMA = 0.1


def hidden_dims_for_D(D):
    if D <= 11:
        return [64, 64]
    elif D <= 51:
        return [128, 128]
    else:
        return [256, 256]


def local_drift_fn(uv):
    """Rotation drift: (-v, u)."""
    return torch.stack([-uv[:, 1], uv[:, 0]], dim=-1)


def local_diffusion_fn(uv):
    """State-dependent diffusion."""
    u, v = uv[:, 0], uv[:, 1]
    B = uv.shape[0]
    sigma = torch.zeros(B, 2, 2, device=uv.device)
    sigma[:, 0, 0] = 1 + u ** 2 / 4
    sigma[:, 0, 1] = u + v
    sigma[:, 1, 1] = 1 + v ** 2 / 4
    return sigma


def get_data_and_sde(D, seed):
    """Get training data and lambdified SDE for given D."""
    if D == 3:
        # Use standard SymPy pipeline for D=3
        from src.numeric.datagen import sample_from_manifold
        from experiments.data_driven_sde import create_manifold_sde
        from experiments.trajectory_fidelity_study import lambdify_sde

        manifold_sde = create_manifold_sde(SURFACE)
        train_data = sample_from_manifold(
            manifold_sde,
            [(-TRAIN_BOUND, TRAIN_BOUND), (-TRAIN_BOUND, TRAIN_BOUND)],
            n_samples=N_TRAIN, seed=seed, device=DEVICE,
        )
        sde = lambdify_sde(create_manifold_sde(SURFACE))
        return train_data, sde
    else:
        # Use FourierAugmented pipeline for D>3
        from src.numeric.highd_manifolds import (
            FourierAugmentedSurface,
            sample_from_highd_manifold,
            create_highd_lambdified_sde,
        )

        surface = FourierAugmentedSurface(SURFACE, D)
        train_data = sample_from_highd_manifold(
            surface, local_drift_fn, local_diffusion_fn,
            [(-TRAIN_BOUND, TRAIN_BOUND), (-TRAIN_BOUND, TRAIN_BOUND)],
            n_samples=N_TRAIN, seed=seed, device=DEVICE,
        )
        sde = create_highd_lambdified_sde(surface, local_drift_fn, local_diffusion_fn)
        return train_data, sde


def train_ae(train_data, D, seed, epochs_ae):
    """Train autoencoder (T+F only, no K) for given D."""
    hdims = hidden_dims_for_D(D)
    lw = LossWeights(tangent_bundle=1.0, diffeo=1.0)

    torch.manual_seed(seed)
    np.random.seed(seed)

    trainer = MultiModelTrainer(TrainingConfig(
        epochs=epochs_ae, n_samples=N_TRAIN, input_dim=D,
        hidden_dim=hdims[0], latent_dim=2, learning_rate=LR_AE,
        batch_size=min(N_TRAIN, BATCH_SIZE), test_size=0.03,
        print_interval=max(1, epochs_ae // 5), device=DEVICE,
    ))
    mc = make_model_config("ae", lw, extrinsic_dim=D, hidden_dims=hdims)
    trainer.add_model(mc)
    loader = trainer.create_data_loader(train_data)

    for epoch in range(epochs_ae):
        losses = trainer.train_epoch(loader)
        if (epoch + 1) % max(1, epochs_ae // 5) == 0:
            print(f"    AE Epoch {epoch+1}/{epochs_ae}: loss={losses['ae']:.6f}")

    ae = trainer.models["ae"]
    ae.eval()
    return ae, losses["ae"]


def compute_drift_error_at_grid(autoencoder, pipeline, sde, device):
    """E_mu of the learned drift_net at a held-out uniform grid."""
    s = torch.linspace(-0.9, 0.9, 10, device=device)
    grid = torch.stack(torch.meshgrid(s, s, indexing='ij'), dim=-1).reshape(-1, 2)

    b_true = sde.ambient_drift(grid).detach()

    with torch.no_grad():
        x_grid = sde.chart(grid)
        z = autoencoder.encoder(x_grid)

    dphi = torch.func.vmap(torch.func.jacrev(autoencoder.decoder))(z)

    # ambient_covariance needs autograd for surface Jacobian (highd case)
    Lambda_true = sde.ambient_covariance(grid).detach()

    with torch.no_grad():
        d2phi = autoencoder.decoder.hessian_network(z)
        g = dphi.mT @ dphi
        ginv = regularized_metric_inverse(g)
        pinv = ginv @ dphi.mT
        P_hat = dphi @ pinv
        P_hat = 0.5 * (P_hat + P_hat.mT)

        Lambda_tan = P_hat @ Lambda_true @ P_hat
        Lambda_tan = 0.5 * (Lambda_tan + Lambda_tan.mT)
        Sigma_z = pinv @ Lambda_tan @ pinv.mT
        Sigma_z = 0.5 * (Sigma_z + Sigma_z.mT)
        q = curvature_drift_explicit_full(d2phi, Sigma_z)

        b_z = pipeline.drift_net(z)
        dphi_bz = (dphi @ b_z.unsqueeze(-1)).squeeze(-1)
        b_pred = dphi_bz + q

    D = b_true.shape[-1]
    err = ((b_pred - b_true) ** 2).sum(-1) / D
    return err.median().item()


def run_fork(ae, train_data, sde, seed, lambda_smooth, use_metric, epochs_sde):
    """Run Stage 2+3 with shared frozen AE."""
    x = train_data.samples.to(DEVICE)
    v = train_data.mu.to(DEVICE)
    Lambda = train_data.cov.to(DEVICE)
    batch_size = min(N_TRAIN, BATCH_SIZE)

    d = 2
    torch.manual_seed(seed + 100)
    drift_net = DriftNet(d).to(DEVICE)
    diffusion_net = DiffusionNet(d).to(DEVICE)
    pipeline = SDEPipelineTrainer(ae, drift_net, diffusion_net, device=DEVICE)

    drift_losses = pipeline.train_stage2(
        x, v, Lambda, epochs=epochs_sde, lr=LR_SDE,
        batch_size=batch_size, print_interval=0,
        lambda_smooth=lambda_smooth, aug_sigma=AUG_SIGMA,
        use_metric=use_metric,
    )

    diff_losses = pipeline.train_stage3(
        x, Lambda, epochs=epochs_sde, lr=LR_SDE,
        batch_size=batch_size, print_interval=0,
    )

    results = evaluate_pipeline(pipeline, ae, sde, seed)
    results["E_mu_held"] = compute_drift_error_at_grid(ae, pipeline, sde, DEVICE)
    results["drift_loss_final"] = drift_losses[-1]
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-seeds", type=int, default=5)
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--sde-epochs", type=int, default=300)
    parser.add_argument("--lambda-smooth", type=float, default=LAMBDA_SMOOTH)
    parser.add_argument("--d-values", type=int, nargs="+", default=[3, 11, 201])
    args = parser.parse_args()

    seeds = [42 + i * 1000 for i in range(args.n_seeds)]
    lam = args.lambda_smooth

    # Conditions: (name, lambda_smooth, use_metric)
    conditions = [
        ("baseline",     0.0, True),
        ("smooth-euc",   lam, False),
        ("smooth-metric", lam, True),
    ]

    rows = []
    t0 = time.time()

    for D in args.d_values:
        print(f"\n{'#'*70}")
        print(f"  D = {D}")
        print(f"{'#'*70}")

        for seed in seeds:
            train_data, sde = get_data_and_sde(D, seed)

            # Train AE once per (D, seed)
            ae, ae_loss = train_ae(train_data, D, seed, args.epochs)

            for cond_name, lam_s, use_met in conditions:
                print(f"\n  D={D} seed={seed} {cond_name} (lambda={lam_s}, metric={use_met})")
                t_fork = time.time()

                results = run_fork(
                    ae, train_data, sde, seed, lam_s, use_met, args.sde_epochs,
                )
                elapsed = time.time() - t_fork

                row = {
                    "D": D, "seed": seed, "condition": cond_name,
                    "ae_loss": ae_loss, **results,
                }
                rows.append(row)
                print(f"    E_mu_held={results['E_mu_held']:.4f}  "
                      f"W2={results['W2@1.0']:.4f}  ({elapsed:.0f}s)")

    df = pd.DataFrame(rows)
    csv_path = "smoothness_across_D.csv"
    df.to_csv(csv_path, index=False)
    print(f"\nSaved {len(df)} rows to {csv_path}")

    # Summary per D
    metrics = ["E_mu_held", "W2@1.0", "sW2@1.0", "MTE@1.0"]

    def paired_compare(name, df_a, df_b):
        print(f"\n  {name}:")
        a = df_a.sort_values("seed")
        b = df_b.sort_values("seed")
        for m in metrics:
            if m not in a.columns:
                continue
            av, bv = a[m].values, b[m].values
            if len(av) != len(bv) or len(av) < 2:
                continue
            mask = np.isfinite(av) & np.isfinite(bv)
            if mask.sum() < 2:
                continue
            ac, bc = av[mask], bv[mask]
            delta = (bc.mean() - ac.mean()) / ac.mean() * 100
            n_help = int((bc < ac).sum())
            _, p = stats.ttest_rel(bc, ac)
            sig = "**" if p < 0.01 else ("*" if p < 0.05 else ("+" if p < 0.1 else ""))
            print(f"    {m:<12} {delta:>+7.1f}%  {n_help}/{len(ac)} wins  p={p:.4f} {sig}")

    for D in args.d_values:
        print(f"\n{'='*70}")
        print(f"  D = {D}")
        print(f"{'='*70}")

        df_d = df[df["D"] == D]
        baseline = df_d[df_d["condition"] == "baseline"]
        s_euc = df_d[df_d["condition"] == "smooth-euc"]
        s_met = df_d[df_d["condition"] == "smooth-metric"]

        paired_compare("smooth-euc vs baseline", baseline, s_euc)
        paired_compare("smooth-metric vs baseline", baseline, s_met)
        paired_compare("smooth-metric vs smooth-euc (does metric help?)", s_euc, s_met)

    print(f"\nTotal time: {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
