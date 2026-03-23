"""
Quick test: is the T+F latent drift field harder to learn?

Hypothesis: T+F chart creates a more complex (nonlinear) latent drift b_z(z),
requiring more drift_net capacity. We test by:
  1. Measuring b_z_target complexity (variance, gradient norm)
  2. Training drift_net with increasing width and checking residual
  3. Comparing the latent drift field smoothness between conditions

Usage:
    python -m experiments.drift_complexity_test --epochs 500 --sde-epochs 300
    python -m experiments.drift_complexity_test --epochs 50 --sde-epochs 50  # smoke
"""

import argparse
import torch
import numpy as np

from src.numeric.losses import LossWeights
from src.numeric.sde_nets import DriftNet, DiffusionNet
from src.numeric.sde_training import SDEPipelineTrainer
from src.numeric.geometry import regularized_metric_inverse

from src.numeric.highd_manifolds import (
    FourierAugmentedSurface,
    sample_from_highd_manifold,
    create_highd_lambdified_sde,
)

from experiments.data_driven_sde import (
    TRAIN_BOUND, BATCH_SIZE, LR_AE, LR_SDE, DEVICE,
)
from experiments.highd_N_D_sweep import (
    hidden_dims_for_D, local_drift_fn, local_diffusion_fn,
)
from experiments.highd_baseline_comparison import _train_ae

NO_PENALTY_LW = LossWeights()
TF_LW = LossWeights(tangent_bundle=1.0, diffeo=1.0)
LAMBDA_SMOOTH = 0.5


def analyze_drift_target(z, b_z_target, g, label):
    """Analyze properties of the encoder-pullback drift target."""
    N, d = z.shape

    # Basic stats
    b_norm = (b_z_target ** 2).sum(-1).sqrt()  # (N,)
    print(f"\n  [{label}] Drift target stats:")
    print(f"    ||b_z||:  mean={b_norm.mean():.4f}, std={b_norm.std():.4f}, "
          f"max={b_norm.max():.4f}")

    # Metric-weighted norm: b^T g b
    b_g_norm = torch.bmm(
        b_z_target.unsqueeze(1),
        torch.bmm(g, b_z_target.unsqueeze(-1))
    ).squeeze().sqrt()
    print(f"    ||b_z||_g: mean={b_g_norm.mean():.4f}, std={b_g_norm.std():.4f}")

    # Drift field variability (higher = more complex)
    b_mean = b_z_target.mean(0)
    b_var = ((b_z_target - b_mean) ** 2).sum(-1).mean()
    print(f"    Variance around mean: {b_var:.4f}")

    # Coefficient of variation
    cv = b_norm.std() / b_norm.mean()
    print(f"    CV(||b_z||): {cv:.4f}")

    # Nearest-neighbor drift difference (local Lipschitz proxy)
    from torch.cdist import cdist if hasattr(torch, 'cdist') else None
    dists = torch.cdist(z, z)  # (N, N)
    dists.fill_diagonal_(float('inf'))
    nn_idx = dists.argmin(dim=1)  # (N,)
    nn_dist = dists.gather(1, nn_idx.unsqueeze(1)).squeeze()  # (N,)
    b_diff = ((b_z_target - b_z_target[nn_idx]) ** 2).sum(-1).sqrt()
    lipschitz_local = b_diff / nn_dist.clamp(min=1e-8)
    print(f"    Local Lipschitz: mean={lipschitz_local.mean():.4f}, "
          f"max={lipschitz_local.max():.4f}")

    # Metric tensor stats
    g_det = torch.linalg.det(g)
    g_cond = torch.linalg.cond(g)
    print(f"    Metric det:  mean={g_det.mean():.4f}, std={g_det.std():.4f}")
    print(f"    Metric cond: mean={g_cond.mean():.4f}, max={g_cond.max():.4f}")

    return {
        "b_norm_mean": b_norm.mean().item(),
        "b_norm_std": b_norm.std().item(),
        "b_var": b_var.item(),
        "cv": cv.item(),
        "lipschitz_mean": lipschitz_local.mean().item(),
        "lipschitz_max": lipschitz_local.max().item(),
        "g_det_mean": g_det.mean().item(),
        "g_cond_mean": g_cond.mean().item(),
    }


def train_drift_varying_capacity(z, b_z_target, g, seed, epochs, label):
    """Train drift_net with different hidden sizes and report final loss."""
    d = z.shape[1]
    batch_size = min(len(z), BATCH_SIZE)
    print(f"\n  [{label}] Drift capacity sweep:")

    for hidden in [32, 64, 128, 256]:
        torch.manual_seed(seed + 100)
        drift_net = DriftNet(d, hidden_dim=hidden).to(DEVICE)
        # Simple dummy pipeline just for training
        dummy_ae = type('obj', (object,), {'encoder': lambda self, x: x, 'decoder': lambda self, x: x})()
        from torch.utils.data import TensorDataset, DataLoader

        optimizer = torch.optim.Adam(drift_net.parameters(), lr=LR_SDE)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=50,
        )
        dataset = TensorDataset(z, b_z_target, g)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        best_loss = float('inf')
        for epoch in range(epochs):
            epoch_loss = 0.0
            n_b = 0
            for z_b, target_b, g_b in loader:
                pred = drift_net(z_b)
                residual = pred - target_b
                # Metric-weighted MSE
                loss = (residual.unsqueeze(1) @ g_b @ residual.unsqueeze(-1)).mean()
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(drift_net.parameters(), max_norm=1.0)
                optimizer.step()
                epoch_loss += loss.item()
                n_b += 1
            avg = epoch_loss / max(n_b, 1)
            scheduler.step(avg)
            if avg < best_loss:
                best_loss = avg

        # Eval final
        drift_net.eval()
        with torch.no_grad():
            pred = drift_net(z)
            res = pred - b_z_target
            final_loss = (res.unsqueeze(1) @ g @ res.unsqueeze(-1)).mean().item()
            # Also unweighted
            final_mse = (res ** 2).sum(-1).mean().item()

        print(f"    hidden={hidden:>3d}: metric_loss={final_loss:.6f}, "
              f"mse={final_mse:.6f}, best_train={best_loss:.6f}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--sde-epochs", type=int, default=300)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--D", type=int, default=11)
    parser.add_argument("--N", type=int, default=50)
    parser.add_argument("--surface", type=str, default="paraboloid")
    args = parser.parse_args()

    print(f"Device: {DEVICE}")
    print(f"D={args.D}, N={args.N}, surface={args.surface}, seed={args.seed}")

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    surface = FourierAugmentedSurface(args.surface, args.D)
    train_data = sample_from_highd_manifold(
        surface, local_drift_fn, local_diffusion_fn,
        [(-TRAIN_BOUND, TRAIN_BOUND), (-TRAIN_BOUND, TRAIN_BOUND)],
        n_samples=args.N, seed=args.seed, device=DEVICE,
    )
    x = train_data.samples.to(DEVICE)
    v = train_data.mu.to(DEVICE)
    Lambda = train_data.cov.to(DEVICE)
    sde = create_highd_lambdified_sde(surface, local_drift_fn, local_diffusion_fn)

    for cond_label, lw in [("no_penalty", NO_PENALTY_LW), ("T+F", TF_LW)]:
        print(f"\n{'='*60}")
        print(f"  Condition: {cond_label}")
        print(f"{'='*60}")

        torch.manual_seed(args.seed)
        np.random.seed(args.seed)

        ae, recon = _train_ae(surface, args.D, args.seed, args.N, args.epochs, lw, train_data)
        print(f"  recon_per_dim={recon:.6f}")

        # Precompute enc-pull targets
        d = 2
        dummy = SDEPipelineTrainer(
            ae, DriftNet(d).to(DEVICE), DiffusionNet(d).to(DEVICE), device=DEVICE,
        )
        z_pre, dphi_pre, _ = dummy.precompute_decoder_derivatives(x)
        b_z_target, g = dummy.precompute_enc_pull_target(x, v, Lambda, dphi_pre)

        # Diagnostic 1: drift target complexity
        analyze_drift_target(z_pre, b_z_target, g, cond_label)

        # Diagnostic 2: capacity sweep
        train_drift_varying_capacity(
            z_pre, b_z_target, g, args.seed, args.sde_epochs, cond_label,
        )


if __name__ == "__main__":
    main()
