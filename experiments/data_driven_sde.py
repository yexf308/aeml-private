"""
Data-driven latent SDE pipeline experiment.

3-stage pipeline:
  Stage 1: Train autoencoder with recon + T + K (existing MultiModelTrainer).
  Stage 2: Freeze AE, train drift_net with tangential drift matching.
  Stage 3: Freeze AE, train diffusion_net with tangent-projected cov + K reg.

Then simulate trajectories with the learned pipeline (Euler-Maruyama in latent space)
and compare against ground truth.

Usage:
    python -m experiments.data_driven_sde --surface sinusoidal --epochs 500
    python -m experiments.data_driven_sde --surface all --epochs 500
"""

import argparse
import math
import torch
import numpy as np
import sympy as sp

from src.numeric.autoencoders import AutoEncoder
from src.numeric.datagen import sample_from_manifold
from src.numeric.losses import LossWeights
from src.numeric.sde_nets import DriftNet, DiffusionNet
from src.numeric.sde_training import SDEPipelineTrainer
from src.numeric.training import MultiModelTrainer, TrainingConfig
from src.symbolic.manifold_sdes import ManifoldSDE
from src.symbolic.riemannian import RiemannianManifold
from src.symbolic.surfaces import surface

from experiments.common import SURFACE_MAP, make_model_config
from experiments.trajectory_fidelity_study import (
    lambdify_sde,
    simulate_ground_truth,
    compute_mte_at_step,
)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Default hyperparameters
TRAIN_BOUND = 1.0
N_TRAIN = 2000
BATCH_SIZE = 32
LR_AE = 0.005
LR_SDE = 1e-3
BOUNDARY = 3.0
N_TRAJ = 200
T_MAX = 1.0
DT = 0.01
N_STEPS = int(T_MAX / DT)
SEED = 42

SURFACES = ["sinusoidal", "paraboloid", "hyperbolic_paraboloid"]


def create_manifold_sde(surface_name: str) -> ManifoldSDE:
    """Create manifold SDE with non-trivial dynamics."""
    u, v = sp.symbols("u v", real=True)
    local_coord, chart = surface(SURFACE_MAP[surface_name], u, v)
    manifold = RiemannianManifold(local_coord, chart)
    local_drift = sp.Matrix([-v, u])
    local_diffusion = sp.Matrix([[1 + u**2 / 4, u + v], [0, 1 + v**2 / 4]])
    return ManifoldSDE(manifold, local_drift=local_drift, local_diffusion=local_diffusion)


def run_pipeline(surface_name: str, epochs_ae: int, epochs_sde: int, seed: int = SEED):
    """Run the full 3-stage pipeline for one surface."""
    print(f"\n{'='*60}")
    print(f"Surface: {surface_name}")
    print(f"{'='*60}")

    torch.manual_seed(seed)
    np.random.seed(seed)

    # Sample training data
    manifold_sde = create_manifold_sde(surface_name)
    train_data = sample_from_manifold(
        manifold_sde,
        [(-TRAIN_BOUND, TRAIN_BOUND), (-TRAIN_BOUND, TRAIN_BOUND)],
        n_samples=N_TRAIN,
        seed=seed,
        device=DEVICE,
    )

    # ====== Stage 1: Autoencoder with T + K ======
    print("\n--- Stage 1: Autoencoder (recon + T + K) ---")
    loss_weights = LossWeights(tangent_bundle=1.0, curvature=1.0)
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
    trainer.add_model(make_model_config("ae", loss_weights))
    data_loader = trainer.create_data_loader(train_data)
    for epoch in range(epochs_ae):
        losses = trainer.train_epoch(data_loader)
        if (epoch + 1) % max(1, epochs_ae // 5) == 0:
            print(f"  Epoch {epoch+1}: loss={losses['ae']:.6f}")

    autoencoder = trainer.models["ae"]
    autoencoder.eval()

    # ====== Stages 2 & 3: SDE nets ======
    d = 2
    drift_net = DriftNet(d).to(DEVICE)
    diffusion_net = DiffusionNet(d).to(DEVICE)

    pipeline = SDEPipelineTrainer(autoencoder, drift_net, diffusion_net, device=DEVICE)

    # Extract (x, v, Lambda) from training data
    x = train_data.samples.to(DEVICE)
    v = train_data.mu.to(DEVICE)
    Lambda = train_data.cov.to(DEVICE)

    print(f"\n--- Stage 2: Drift net ({epochs_sde} epochs) ---")
    pipeline.train_stage2(x, v, Lambda, epochs=epochs_sde, lr=LR_SDE,
                          batch_size=BATCH_SIZE, print_interval=max(1, epochs_sde // 5))

    print(f"\n--- Stage 3: Diffusion net ({epochs_sde} epochs) ---")
    pipeline.train_stage3(x, v, Lambda, epochs=epochs_sde, lr=LR_SDE,
                          batch_size=BATCH_SIZE, lambda_K=0.1,
                          print_interval=max(1, epochs_sde // 5))

    # ====== Evaluation: simulate and compare ======
    print("\n--- Evaluating trajectory fidelity ---")
    sde = lambdify_sde(create_manifold_sde(surface_name))

    torch.manual_seed(seed + 999)
    init_local = (torch.rand(N_TRAJ, 2, device=DEVICE) * 2 - 1) * TRAIN_BOUND
    init_ambient = sde.chart(init_local).to(DEVICE)

    torch.manual_seed(seed + 1234)
    dW = torch.randn(N_TRAJ, N_STEPS, 2, device=DEVICE)

    # Ground truth
    gt_traj, gt_alive = simulate_ground_truth(init_local, sde, N_STEPS, DT, dW, BOUNDARY)

    # Learned pipeline
    with torch.no_grad():
        z0 = autoencoder.encoder(init_ambient)
    _, learned_traj = pipeline.simulate(z0, N_STEPS, DT, dW=dW)

    # Compute MTE at snapshots
    for t_val in [0.1, 0.5, 1.0]:
        step = int(round(t_val / DT))
        # For learned pipeline, all trajectories are "alive" (no boundary check in latent)
        alive = gt_alive[:, step]
        mte = compute_mte_at_step(learned_traj, gt_traj, alive, step)
        print(f"  MTE@{t_val}: {mte:.4f}")

    return {
        "surface": surface_name,
        "autoencoder": autoencoder,
        "drift_net": drift_net,
        "diffusion_net": diffusion_net,
        "pipeline": pipeline,
    }


def main():
    parser = argparse.ArgumentParser(description="Data-driven latent SDE pipeline")
    parser.add_argument("--surface", type=str, default="sinusoidal",
                        help="Surface name or 'all'")
    parser.add_argument("--epochs", type=int, default=500,
                        help="Autoencoder training epochs")
    parser.add_argument("--sde-epochs", type=int, default=300,
                        help="SDE net training epochs (stages 2 & 3)")
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args()

    print(f"Device: {DEVICE}")
    surfaces = SURFACES if args.surface == "all" else [args.surface]

    for surf in surfaces:
        run_pipeline(surf, epochs_ae=args.epochs, epochs_sde=args.sde_epochs, seed=args.seed)

    print("\nDone.")


if __name__ == "__main__":
    main()
