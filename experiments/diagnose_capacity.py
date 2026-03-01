"""
Diagnostic: Does a larger AE decoder let T+F+K outperform T+F?

Hypothesis: With a [64,64] decoder, K and F compete for capacity.
A larger decoder should have room for both.

Sweep:
  - N_train: 20, 200
  - Decoder hidden_dims: [64,64], [128,128], [128,128,128], [256,256]
  - AE reg: T+F vs T+F+K (curvature weight=1.0)

Surface: sinusoidal, seed=42.
AE: 500 epochs, SDE: 300 epochs.
Metrics: recon_mse (500 held-out), MTE@1.0, W2@1.0, MMD@1.0.

Usage:
    PYTHONUNBUFFERED=1 python -m experiments.diagnose_capacity
"""

import time
import torch
import numpy as np
import pandas as pd

from src.numeric.datagen import sample_from_manifold
from src.numeric.losses import LossWeights
from src.numeric.sde_nets import DriftNet, DiffusionNet
from src.numeric.sde_training import SDEPipelineTrainer
from src.numeric.training import MultiModelTrainer, TrainingConfig

from experiments.common import make_model_config
from experiments.data_driven_sde import (
    DEVICE, TRAIN_BOUND, BATCH_SIZE, LR_AE, LR_SDE, SEED,
    create_manifold_sde, evaluate_pipeline,
)
from experiments.trajectory_fidelity_study import lambdify_sde


SURFACE = "sinusoidal"
EPOCHS_AE = 500
EPOCHS_SDE = 300
N_TEST = 500

N_TRAIN_VALUES = [20, 200]
HIDDEN_DIMS_LIST = [
    [64, 64],
    [128, 128],
    [128, 128, 128],
    [256, 256],
]
REG_CONFIGS = {
    "T+F":   LossWeights(tangent_bundle=1.0, diffeo=1.0),
    "T+F+K": LossWeights(tangent_bundle=1.0, diffeo=1.0, curvature=1.0),
}


def train_ae(train_data, hidden_dims, lw, reg_name, n_train, seed):
    """Train an AE with specified decoder size and regularization."""
    torch.manual_seed(seed)
    np.random.seed(seed)

    bs = min(BATCH_SIZE, n_train)
    print(f"    AE training: hidden={hidden_dims}, reg={reg_name}, "
          f"N={n_train}, bs={bs}")

    trainer = MultiModelTrainer(TrainingConfig(
        epochs=EPOCHS_AE,
        n_samples=n_train,
        input_dim=3,
        hidden_dim=hidden_dims[0],
        latent_dim=2,
        learning_rate=LR_AE,
        batch_size=bs,
        test_size=0.0,
        print_interval=max(1, EPOCHS_AE // 5),
        device=DEVICE,
    ))
    trainer.add_model(make_model_config("ae", lw, hidden_dims=hidden_dims))
    loader = trainer.create_data_loader(train_data)
    for epoch in range(EPOCHS_AE):
        losses = trainer.train_epoch(loader)
        if (epoch + 1) % max(1, EPOCHS_AE // 5) == 0:
            print(f"      Epoch {epoch+1}: loss={losses['ae']:.6f}")

    ae = trainer.models["ae"]
    ae.eval()
    return ae, losses["ae"]


def compute_recon_mse(ae, test_x):
    """Reconstruction MSE on held-out test points."""
    with torch.no_grad():
        z = ae.encoder(test_x)
        x_hat = ae.decoder(z)
        mse = ((x_hat - test_x) ** 2).sum(-1).mean().item()
    return mse


def run_experiment():
    seed = SEED
    t0 = time.time()

    print(f"Device: {DEVICE}")
    print(f"Surface: {SURFACE}")
    print(f"AE epochs: {EPOCHS_AE}, SDE epochs: {EPOCHS_SDE}")
    print(f"N_train values: {N_TRAIN_VALUES}")
    print(f"Decoder hidden_dims: {HIDDEN_DIMS_LIST}")
    print(f"Regs: {list(REG_CONFIGS.keys())}")
    print()

    # ================================================================
    # Lambdify SDE once
    # ================================================================
    print("--- Lambdifying SDE for evaluation (once) ---")
    sde = lambdify_sde(create_manifold_sde(SURFACE))

    # Generate test data (500 held-out points, fixed seed)
    print("--- Generating test data (500 pts) ---")
    torch.manual_seed(seed + 9999)
    np.random.seed(seed + 9999)
    manifold_sde_test = create_manifold_sde(SURFACE)
    test_data = sample_from_manifold(
        manifold_sde_test,
        [(-TRAIN_BOUND, TRAIN_BOUND), (-TRAIN_BOUND, TRAIN_BOUND)],
        n_samples=N_TEST, seed=seed + 9999, device=DEVICE,
    )
    test_x = test_data.samples.to(DEVICE)

    rows = []

    for n_train in N_TRAIN_VALUES:
        print(f"\n{'='*70}")
        print(f"N_TRAIN = {n_train}")
        print(f"{'='*70}")

        # Generate training data
        torch.manual_seed(seed)
        np.random.seed(seed)
        manifold_sde = create_manifold_sde(SURFACE)
        train_data = sample_from_manifold(
            manifold_sde,
            [(-TRAIN_BOUND, TRAIN_BOUND), (-TRAIN_BOUND, TRAIN_BOUND)],
            n_samples=n_train, seed=seed, device=DEVICE,
        )
        x = train_data.samples.to(DEVICE)
        v = train_data.mu.to(DEVICE)
        Lambda = train_data.cov.to(DEVICE)

        for hidden_dims in HIDDEN_DIMS_LIST:
            for reg_name, lw in REG_CONFIGS.items():
                print(f"\n  --- N={n_train}, hidden={hidden_dims}, reg={reg_name} ---")
                t_run = time.time()

                # Stage 1: Train AE
                ae, ae_loss = train_ae(
                    train_data, hidden_dims, lw, reg_name, n_train, seed,
                )

                # Reconstruction MSE on held-out data
                recon_mse = compute_recon_mse(ae, test_x)
                print(f"    recon_mse (test): {recon_mse:.6f}")

                # Precompute decoder derivatives
                d = 2
                dummy = SDEPipelineTrainer(
                    ae, DriftNet(d).to(DEVICE), DiffusionNet(d).to(DEVICE),
                    device=DEVICE,
                )
                z_pre, dphi_pre, d2phi_pre = dummy.precompute_decoder_derivatives(
                    x.to(DEVICE)
                )

                # Stage 2: Train drift net
                torch.manual_seed(seed + 100)
                drift_net = DriftNet(d).to(DEVICE)
                dp = SDEPipelineTrainer(
                    ae, drift_net, DiffusionNet(d).to(DEVICE), device=DEVICE,
                )
                bs_sde = min(BATCH_SIZE, n_train)
                drift_losses = dp.train_stage2_precomputed(
                    z_pre, dphi_pre, d2phi_pre, v, Lambda,
                    epochs=EPOCHS_SDE, lr=LR_SDE,
                    batch_size=bs_sde, print_interval=60,
                )
                drift_net.eval()

                # Stage 3: Train diffusion net
                torch.manual_seed(seed + 200)
                diff_net = DiffusionNet(d).to(DEVICE)
                pipeline = SDEPipelineTrainer(
                    ae, drift_net, diff_net, device=DEVICE,
                )
                diff_losses = pipeline.train_stage3_precomputed(
                    z_pre, dphi_pre, Lambda,
                    epochs=EPOCHS_SDE, lr=LR_SDE,
                    batch_size=bs_sde, print_interval=60,
                )

                # Evaluate
                print("    Evaluating trajectory fidelity...")
                eval_results = evaluate_pipeline(pipeline, ae, sde, seed)

                n_ae_params = sum(p.numel() for p in ae.parameters())
                rows.append({
                    "N_train": n_train,
                    "hidden_dims": str(hidden_dims),
                    "reg": reg_name,
                    "ae_params": n_ae_params,
                    "ae_loss": ae_loss,
                    "recon_mse": recon_mse,
                    "drift_loss": drift_losses[-1],
                    "diff_loss": diff_losses[-1],
                    "MTE@1.0": eval_results["MTE@1.0"],
                    "W2@1.0": eval_results["W2@1.0"],
                    "MMD@1.0": eval_results["MMD@1.0"],
                })
                elapsed = time.time() - t_run
                print(f"    Done in {elapsed:.1f}s")

    # ================================================================
    # Summary table
    # ================================================================
    df = pd.DataFrame(rows)

    print(f"\n\n{'='*90}")
    print("FULL RESULTS TABLE")
    print(f"{'='*90}")
    show_cols = [
        "N_train", "hidden_dims", "reg", "ae_params",
        "recon_mse", "MTE@1.0", "W2@1.0", "MMD@1.0",
    ]
    print(df[show_cols].to_string(index=False))

    # ================================================================
    # T+F vs T+F+K comparison
    # ================================================================
    print(f"\n\n{'='*90}")
    print("T+F vs T+F+K COMPARISON (delta = T+F+K - T+F; negative => K helps)")
    print(f"{'='*90}")

    delta_rows = []
    for (n_train, hdims), group in df.groupby(["N_train", "hidden_dims"]):
        tf_row = group[group["reg"] == "T+F"].iloc[0]
        tfk_row = group[group["reg"] == "T+F+K"].iloc[0]
        d_recon = tfk_row["recon_mse"] - tf_row["recon_mse"]
        d_mte = tfk_row["MTE@1.0"] - tf_row["MTE@1.0"]
        d_w2 = tfk_row["W2@1.0"] - tf_row["W2@1.0"]
        d_mmd = tfk_row["MMD@1.0"] - tf_row["MMD@1.0"]
        k_helps = (d_mte < 0) or (d_w2 < 0)
        delta_rows.append({
            "N_train": n_train,
            "hidden_dims": hdims,
            "ae_params": tf_row["ae_params"],
            "TF_recon": tf_row["recon_mse"],
            "TFK_recon": tfk_row["recon_mse"],
            "d_recon": d_recon,
            "TF_MTE": tf_row["MTE@1.0"],
            "TFK_MTE": tfk_row["MTE@1.0"],
            "d_MTE": d_mte,
            "TF_W2": tf_row["W2@1.0"],
            "TFK_W2": tfk_row["W2@1.0"],
            "d_W2": d_w2,
            "TF_MMD": tf_row["MMD@1.0"],
            "TFK_MMD": tfk_row["MMD@1.0"],
            "d_MMD": d_mmd,
            "K_helps": k_helps,
        })

    df_delta = pd.DataFrame(delta_rows)
    print(df_delta.to_string(index=False))

    # ================================================================
    # Concise verdict
    # ================================================================
    print(f"\n\n{'='*90}")
    print("CONCISE VERDICT")
    print(f"{'='*90}")
    for _, r in df_delta.iterrows():
        verdict = "K HELPS" if r["K_helps"] else "K HURTS or NEUTRAL"
        print(f"  N={int(r['N_train']):>3d}  hidden={r['hidden_dims']:<20s}  "
              f"d_MTE={r['d_MTE']:+.4f}  d_W2={r['d_W2']:+.4f}  "
              f"d_MMD={r['d_MMD']:+.4f}  => {verdict}")

    csv_path = "/network/rit/lab/Yelab/aeml/diagnose_capacity_results.csv"
    df.to_csv(csv_path, index=False)
    print(f"\nSaved results to {csv_path}")
    print(f"Total time: {time.time() - t0:.1f}s")


if __name__ == "__main__":
    run_experiment()
