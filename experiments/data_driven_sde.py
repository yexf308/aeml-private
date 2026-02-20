"""
Data-driven latent SDE pipeline experiment.

3-stage pipeline:
  Stage 1: Train autoencoder with recon + regularization.
  Stage 2: Freeze AE, train drift_net with tangential drift matching.
  Stage 3: Freeze AE, train diffusion_net with ambient covariance matching.

Then simulate trajectories with the learned pipeline (Euler-Maruyama in latent space)
and compare against ground truth.

Usage:
    python -m experiments.data_driven_sde --surface sinusoidal --epochs 500
    python -m experiments.data_driven_sde --surface all --epochs 500
    python -m experiments.data_driven_sde --ablation --surface sinusoidal
"""

import argparse
import time
import torch
import numpy as np
import pandas as pd
import sympy as sp

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
    compute_w2,
    compute_mmd,
)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Default hyperparameters
TRAIN_BOUND = 1.0
N_TRAIN = 20
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

# Regularization ablation configs (Stage 1 AE regularization)
# K weight 0.1 (not 1.0) — discovered via sweep that K=1.0 overwhelms F,
# while K=0.1-0.5 provides useful curvature regularization.
REG_CONFIGS = {
    "T":     LossWeights(tangent_bundle=1.0),
    "T+K":   LossWeights(tangent_bundle=1.0, curvature=0.1),
    "T+F":   LossWeights(tangent_bundle=1.0, diffeo=1.0),
    "T+F+K": LossWeights(tangent_bundle=1.0, diffeo=1.0, curvature=0.1),
}


def create_manifold_sde(surface_name: str) -> ManifoldSDE:
    """Create manifold SDE with non-trivial dynamics."""
    u, v = sp.symbols("u v", real=True)
    local_coord, chart = surface(SURFACE_MAP[surface_name], u, v)
    manifold = RiemannianManifold(local_coord, chart)
    local_drift = sp.Matrix([-v, u])
    local_diffusion = sp.Matrix([[1 + u**2 / 4, u + v], [0, 1 + v**2 / 4]])
    return ManifoldSDE(manifold, local_drift=local_drift, local_diffusion=local_diffusion)


def train_autoencoder(train_data, epochs_ae, ae_loss_weights, ae_reg_name, seed):
    """Stage 1: Train autoencoder with given regularization."""
    torch.manual_seed(seed)
    np.random.seed(seed)

    print(f"\n--- Stage 1: Autoencoder (recon + {ae_reg_name}) ---")
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

    ae_final_loss = losses["ae"]
    autoencoder = trainer.models["ae"]
    autoencoder.eval()
    return autoencoder, ae_final_loss


def evaluate_pipeline(pipeline, autoencoder, sde, seed):
    """Evaluate trajectory fidelity of the learned pipeline."""
    torch.manual_seed(seed + 999)
    init_local = (torch.rand(N_TRAJ, 2, device=DEVICE) * 2 - 1) * TRAIN_BOUND
    init_ambient = sde.chart(init_local).to(DEVICE)

    torch.manual_seed(seed + 1234)
    dW = torch.randn(N_TRAJ, N_STEPS, 2, device=DEVICE)

    gt_traj, gt_alive = simulate_ground_truth(init_local, sde, N_STEPS, DT, dW, BOUNDARY)

    with torch.no_grad():
        z0 = autoencoder.encoder(init_ambient)
    _, learned_traj = pipeline.simulate(z0, N_STEPS, DT, dW=dW)

    # Learned trajectories have no boundary — mark all alive
    B = learned_traj.shape[0]
    learned_alive = torch.ones(B, N_STEPS + 1, dtype=torch.bool, device=DEVICE)

    results = {}

    # Pointwise: MTE at multiple horizons
    for t_val in [0.1, 0.5, 1.0]:
        step = int(round(t_val / DT))
        both_alive = gt_alive[:, step] & learned_alive[:, step]
        mte = compute_mte_at_step(learned_traj, gt_traj, both_alive, step)
        results[f"MTE@{t_val}"] = mte
        print(f"  MTE@{t_val}: {mte:.4f}")

    # Distributional: W2 and MMD at T=1.0
    step_final = int(round(1.0 / DT))
    w2 = compute_w2(
        learned_traj[:, step_final], gt_traj[:, step_final],
        learned_alive[:, step_final], gt_alive[:, step_final],
    )
    mmd = compute_mmd(
        learned_traj[:, step_final], gt_traj[:, step_final],
        learned_alive[:, step_final], gt_alive[:, step_final],
    )
    results["W2@1.0"] = w2
    results["MMD@1.0"] = mmd
    print(f"  W2@1.0:  {w2:.4f}")
    print(f"  MMD@1.0: {mmd:.4f}")

    # Survival: fraction of GT trajectories still in bounds at T=1.0
    survival = gt_alive[:, step_final].float().mean().item()
    results["survival@1.0"] = survival
    print(f"  Survival@1.0: {survival:.1%}")

    return results


def run_pipeline(
    surface_name: str,
    epochs_ae: int,
    epochs_sde: int,
    seed: int = SEED,
    ae_loss_weights: LossWeights = None,
    ae_reg_name: str = "T+F+K",
):
    """Run the full 3-stage pipeline for one surface."""
    if ae_loss_weights is None:
        ae_loss_weights = LossWeights(tangent_bundle=1.0, diffeo=1.0, curvature=1.0)

    print(f"\n{'='*60}")
    print(f"Surface: {surface_name} | AE reg: {ae_reg_name}")
    print(f"{'='*60}")

    torch.manual_seed(seed)
    np.random.seed(seed)

    manifold_sde = create_manifold_sde(surface_name)
    train_data = sample_from_manifold(
        manifold_sde,
        [(-TRAIN_BOUND, TRAIN_BOUND), (-TRAIN_BOUND, TRAIN_BOUND)],
        n_samples=N_TRAIN,
        seed=seed,
        device=DEVICE,
    )

    autoencoder, ae_final_loss = train_autoencoder(
        train_data, epochs_ae, ae_loss_weights, ae_reg_name, seed,
    )

    d = 2
    drift_net = DriftNet(d).to(DEVICE)
    diffusion_net = DiffusionNet(d).to(DEVICE)
    pipeline = SDEPipelineTrainer(autoencoder, drift_net, diffusion_net, device=DEVICE)

    x = train_data.samples.to(DEVICE)
    v = train_data.mu.to(DEVICE)
    Lambda = train_data.cov.to(DEVICE)

    print(f"\n--- Stage 2: Drift net ({epochs_sde} epochs) ---")
    drift_losses = pipeline.train_stage2(
        x, v, Lambda, epochs=epochs_sde, lr=LR_SDE,
        batch_size=BATCH_SIZE, print_interval=max(1, epochs_sde // 5),
    )

    print(f"\n--- Stage 3: Diffusion net ({epochs_sde} epochs) ---")
    diff_losses = pipeline.train_stage3(
        x, Lambda, epochs=epochs_sde, lr=LR_SDE,
        batch_size=BATCH_SIZE,
        print_interval=max(1, epochs_sde // 5),
    )

    print("\n--- Evaluating trajectory fidelity ---")
    sde = lambdify_sde(create_manifold_sde(surface_name))
    mte_results = evaluate_pipeline(pipeline, autoencoder, sde, seed)

    return {
        "surface": surface_name,
        "ae_reg": ae_reg_name,
        "ae_loss": ae_final_loss,
        "drift_loss": drift_losses[-1],
        "diff_loss": diff_losses[-1],
        **mte_results,
    }


def run_ablation(surface_name: str, epochs_ae: int, epochs_sde: int, seed: int = SEED):
    """AE regularization ablation: share data, share drift_net per AE group."""
    print(f"\nAE regularization ablation on {surface_name}")
    print(f"AE regs: {list(REG_CONFIGS.keys())}")

    t0 = time.time()

    # ====== Shared across ALL runs: data + evaluation setup ======
    print("\n--- Generating training data (once) ---")
    torch.manual_seed(seed)
    np.random.seed(seed)
    manifold_sde = create_manifold_sde(surface_name)
    train_data = sample_from_manifold(
        manifold_sde,
        [(-TRAIN_BOUND, TRAIN_BOUND), (-TRAIN_BOUND, TRAIN_BOUND)],
        n_samples=N_TRAIN,
        seed=seed,
        device=DEVICE,
    )
    x = train_data.samples.to(DEVICE)
    v = train_data.mu.to(DEVICE)
    Lambda = train_data.cov.to(DEVICE)

    print("--- Lambdifying SDE for evaluation (once) ---")
    sde = lambdify_sde(create_manifold_sde(surface_name))

    rows = []
    for reg_name, lw in REG_CONFIGS.items():
        print(f"\n{'='*60}")
        print(f"AE reg group: {reg_name}")
        print(f"{'='*60}")

        autoencoder, ae_final_loss = train_autoencoder(
            train_data, epochs_ae, lw, reg_name, seed,
        )

        # Precompute decoder derivatives once for this AE
        print("  Precomputing decoder Jacobians/Hessians...")
        t_pre = time.time()
        d = 2
        dummy_pipeline = SDEPipelineTrainer(
            autoencoder, DriftNet(d).to(DEVICE), DiffusionNet(d).to(DEVICE), device=DEVICE,
        )
        z_pre, dphi_pre, d2phi_pre = dummy_pipeline.precompute_decoder_derivatives(x)
        print(f"  Precomputed in {time.time() - t_pre:.1f}s")

        # Stage 2: Train drift_net (one per AE group)
        torch.manual_seed(seed + 100)
        drift_net = DriftNet(d).to(DEVICE)
        diffusion_net_tmp = DiffusionNet(d).to(DEVICE)
        drift_pipeline = SDEPipelineTrainer(
            autoencoder, drift_net, diffusion_net_tmp, device=DEVICE,
        )
        print(f"\n  Stage 2: Drift net ({epochs_sde} epochs)")
        drift_losses = drift_pipeline.train_stage2_precomputed(
            z_pre, dphi_pre, d2phi_pre, v, Lambda,
            epochs=epochs_sde, lr=LR_SDE,
            batch_size=BATCH_SIZE, print_interval=max(1, epochs_sde // 5),
        )
        drift_net.eval()

        # Stage 3: Train diffusion_net
        torch.manual_seed(seed + 200)
        diffusion_net = DiffusionNet(d).to(DEVICE)
        pipeline = SDEPipelineTrainer(
            autoencoder, drift_net, diffusion_net, device=DEVICE,
        )
        t_run = time.time()

        print(f"\n  Stage 3: Diffusion net ({epochs_sde} epochs)")
        diff_losses = pipeline.train_stage3_precomputed(
            z_pre, dphi_pre, Lambda,
            epochs=epochs_sde, lr=LR_SDE,
            batch_size=BATCH_SIZE, print_interval=max(1, epochs_sde // 5),
        )

        print("  Evaluating trajectory fidelity...")
        mte_results = evaluate_pipeline(pipeline, autoencoder, sde, seed)

        rows.append({
            "surface": surface_name,
            "ae_reg": reg_name,
            "ae_loss": ae_final_loss,
            "drift_loss": drift_losses[-1],
            "diff_loss": diff_losses[-1],
            **mte_results,
        })
        print(f"  Run time: {time.time() - t_run:.1f}s")

    df = pd.DataFrame(rows)
    print(f"\n\n{'='*70}")
    print("ABLATION RESULTS")
    print(f"{'='*70}")
    print(df.to_string(index=False))
    print(f"\nTotal ablation time: {time.time() - t0:.1f}s")

    csv_path = f"sde_ablation_{surface_name}.csv"
    df.to_csv(csv_path, index=False)
    print(f"Saved {csv_path}")
    return df


def run_ablation_K(surface_name: str, epochs_ae: int, epochs_sde: int, seed: int = SEED):
    """Lambda_K ablation: fixed T+F AE, shared drift_net, sweep lambda_K in Stage 3."""
    lambda_K_values = [0.0, 0.01, 0.1, 1.0]
    print(f"\nLambda_K ablation on {surface_name} (AE=T+F)")
    print(f"lambda_K values: {lambda_K_values}")

    t0 = time.time()

    # Shared data
    print("\n--- Generating training data (once) ---")
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

    print("--- Lambdifying SDE for evaluation (once) ---")
    sde = lambdify_sde(create_manifold_sde(surface_name))

    # Single T+F AE
    lw = LossWeights(tangent_bundle=1.0, diffeo=1.0)
    autoencoder, ae_final_loss = train_autoencoder(train_data, epochs_ae, lw, "T+F", seed)

    # Precompute once
    print("  Precomputing decoder Jacobians/Hessians...")
    d = 2
    dummy = SDEPipelineTrainer(
        autoencoder, DriftNet(d).to(DEVICE), DiffusionNet(d).to(DEVICE), device=DEVICE,
    )
    z_pre, dphi_pre, d2phi_pre = dummy.precompute_decoder_derivatives(x)

    # Shared drift_net
    torch.manual_seed(seed + 100)
    drift_net = DriftNet(d).to(DEVICE)
    drift_pipeline = SDEPipelineTrainer(
        autoencoder, drift_net, DiffusionNet(d).to(DEVICE), device=DEVICE,
    )
    print(f"\n  Stage 2: Drift net ({epochs_sde} epochs) — shared")
    drift_losses = drift_pipeline.train_stage2_precomputed(
        z_pre, dphi_pre, d2phi_pre, v, Lambda,
        epochs=epochs_sde, lr=LR_SDE,
        batch_size=BATCH_SIZE, print_interval=max(1, epochs_sde // 5),
    )
    drift_net.eval()

    # Sweep lambda_K in Stage 3
    rows = []
    for lk in lambda_K_values:
        print(f"\n{'='*60}")
        print(f"lambda_K = {lk}")
        print(f"{'='*60}")

        torch.manual_seed(seed + 200)
        diffusion_net = DiffusionNet(d).to(DEVICE)
        pipeline = SDEPipelineTrainer(
            autoencoder, drift_net, diffusion_net, device=DEVICE,
        )
        t_run = time.time()

        print(f"  Stage 3: Diffusion net ({epochs_sde} epochs, lambda_K={lk})")
        diff_losses = pipeline.train_stage3_precomputed(
            z_pre, dphi_pre, Lambda,
            epochs=epochs_sde, lr=LR_SDE,
            batch_size=BATCH_SIZE, print_interval=max(1, epochs_sde // 5),
            v=v, d2phi=d2phi_pre, lambda_K=lk,
        )

        print("  Evaluating trajectory fidelity...")
        eval_results = evaluate_pipeline(pipeline, autoencoder, sde, seed)

        rows.append({
            "surface": surface_name,
            "ae_reg": "T+F",
            "lambda_K": lk,
            "ae_loss": ae_final_loss,
            "drift_loss": drift_losses[-1],
            "diff_loss": diff_losses[-1],
            **eval_results,
        })
        print(f"  Run time: {time.time() - t_run:.1f}s")

    df = pd.DataFrame(rows)
    print(f"\n\n{'='*70}")
    print("LAMBDA_K ABLATION RESULTS")
    print(f"{'='*70}")
    print(df.to_string(index=False))
    print(f"\nTotal time: {time.time() - t0:.1f}s")

    csv_path = f"sde_ablation_K_{surface_name}.csv"
    df.to_csv(csv_path, index=False)
    print(f"Saved {csv_path}")
    return df


def main():
    parser = argparse.ArgumentParser(description="Data-driven latent SDE pipeline")
    parser.add_argument("--surface", type=str, default="sinusoidal",
                        help="Surface name or 'all'")
    parser.add_argument("--epochs", type=int, default=500,
                        help="Autoencoder training epochs")
    parser.add_argument("--sde-epochs", type=int, default=300,
                        help="SDE net training epochs (stages 2 & 3)")
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--ablation", action="store_true",
                        help="Run AE regularization ablation")
    parser.add_argument("--ablation-K", action="store_true", dest="ablation_K",
                        help="Run lambda_K ablation (T+F AE, sweep lambda_K in Stage 3)")
    args = parser.parse_args()

    print(f"Device: {DEVICE}")

    if args.ablation_K:
        surfaces = SURFACES if args.surface == "all" else [args.surface]
        for surf in surfaces:
            run_ablation_K(surf, epochs_ae=args.epochs, epochs_sde=args.sde_epochs, seed=args.seed)
    elif args.ablation:
        surfaces = SURFACES if args.surface == "all" else [args.surface]
        for surf in surfaces:
            run_ablation(surf, epochs_ae=args.epochs, epochs_sde=args.sde_epochs, seed=args.seed)
    else:
        surfaces = SURFACES if args.surface == "all" else [args.surface]
        for surf in surfaces:
            run_pipeline(surf, epochs_ae=args.epochs, epochs_sde=args.sde_epochs, seed=args.seed)

    print("\nDone.")


if __name__ == "__main__":
    main()
