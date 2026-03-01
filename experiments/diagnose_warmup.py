"""
Diagnostic: Does warm-up with T+F before switching to T+F+K help?

Strategy: Train AE with T+F for W epochs, then T+F+K for remaining epochs.
Sweep: W ∈ {0, 100, 250, 400, 500} (where 0=T+F+K from scratch, 500=pure T+F)
Test at N=20 and N=200.

Usage:
    PYTHONUNBUFFERED=1 python -m experiments.diagnose_warmup
"""

import time
import torch
import numpy as np

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


def train_ae_warmup(train_data, n_train, total_epochs, warmup_epochs, seed):
    """Train AE: T+F for warmup_epochs, then T+F+K for remaining."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    bs = min(BATCH_SIZE, n_train)

    lw_tf = LossWeights(tangent_bundle=1.0, diffeo=1.0)
    lw_tfk = LossWeights(tangent_bundle=1.0, diffeo=1.0, curvature=1.0)

    # Phase 1: T+F
    if warmup_epochs > 0:
        trainer = MultiModelTrainer(TrainingConfig(
            epochs=warmup_epochs, n_samples=n_train, input_dim=3, hidden_dim=64,
            latent_dim=2, learning_rate=LR_AE, batch_size=bs,
            test_size=0.0, print_interval=max(1, total_epochs // 5), device=DEVICE,
        ))
        trainer.add_model(make_model_config("ae", lw_tf, hidden_dims=[64, 64]))
        loader = trainer.create_data_loader(train_data)
        for epoch in range(warmup_epochs):
            losses = trainer.train_epoch(loader)
            if (epoch + 1) % max(1, total_epochs // 5) == 0:
                print(f"    [T+F] Epoch {epoch+1}: loss={losses['ae']:.6f}")
        ae = trainer.models["ae"]
    else:
        ae = None

    # Phase 2: T+F+K (continue from same model)
    remaining = total_epochs - warmup_epochs
    if remaining > 0:
        trainer2 = MultiModelTrainer(TrainingConfig(
            epochs=remaining, n_samples=n_train, input_dim=3, hidden_dim=64,
            latent_dim=2, learning_rate=LR_AE, batch_size=bs,
            test_size=0.0, print_interval=max(1, total_epochs // 5), device=DEVICE,
        ))
        trainer2.add_model(make_model_config("ae", lw_tfk, hidden_dims=[64, 64]))
        # Copy weights from phase 1
        if ae is not None:
            trainer2.models["ae"].load_state_dict(ae.state_dict())
        loader2 = trainer2.create_data_loader(train_data)
        for epoch in range(remaining):
            losses = trainer2.train_epoch(loader2)
            if (warmup_epochs + epoch + 1) % max(1, total_epochs // 5) == 0:
                print(f"    [T+F+K] Epoch {warmup_epochs + epoch+1}: loss={losses['ae']:.6f}")
        ae = trainer2.models["ae"]
        final_loss = losses["ae"]
    else:
        final_loss = losses["ae"]

    ae.eval()
    return ae, final_loss


def eval_recon(ae, test_x):
    with torch.no_grad():
        z = ae.encoder(test_x)
        x_hat = ae.decoder(z)
        return ((x_hat - test_x) ** 2).sum(-1).mean().item()


def eval_geometry(ae, test_x):
    from src.numeric.geometry import regularized_metric_inverse
    with torch.no_grad():
        z = ae.encoder(test_x)
        dphi = ae.decoder.jacobian_network(z)
        g = dphi.mT @ dphi
        eigvals = torch.linalg.eigvalsh(g)
        cond = (eigvals[:, -1] / eigvals[:, 0].clamp(min=1e-10))
        return {"mean_cond": cond.mean().item(), "max_cond": cond.max().item()}


def main():
    surface_name = "sinusoidal"
    total_epochs = 500
    epochs_sde = 300
    seed = SEED
    d = 2

    n_train_values = [20, 200]
    warmup_values = [0, 100, 250, 400, 500]  # 0=pure T+F+K, 500=pure T+F

    print(f"Device: {DEVICE}")
    print(f"Surface: {surface_name}")
    print(f"N_TRAIN: {n_train_values}")
    print(f"Warmup epochs (T+F before T+F+K): {warmup_values}")

    t0 = time.time()

    # Generate data
    torch.manual_seed(seed)
    np.random.seed(seed)
    manifold_sde = create_manifold_sde(surface_name)
    full_data = sample_from_manifold(
        manifold_sde,
        [(-TRAIN_BOUND, TRAIN_BOUND), (-TRAIN_BOUND, TRAIN_BOUND)],
        n_samples=2000, seed=seed, device=DEVICE,
    )
    x_full = full_data.samples.to(DEVICE)
    test_x = x_full[1500:]
    sde = lambdify_sde(create_manifold_sde(surface_name))

    rows = []

    for n_train in n_train_values:
        print(f"\n{'='*70}")
        print(f"N_TRAIN = {n_train}")
        print(f"{'='*70}")

        torch.manual_seed(seed)
        idx = torch.randperm(1500)[:n_train]
        x_sub = x_full[idx]
        v_sub = full_data.mu[idx].to(DEVICE)
        Lambda_sub = full_data.cov[idx].to(DEVICE)

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

        for warmup in warmup_values:
            label = f"W={warmup}" if warmup < total_epochs else "pure T+F"
            if warmup == 0:
                label = "pure T+F+K"
            print(f"\n  --- {label} (T+F for {warmup}, T+F+K for {total_epochs - warmup}) ---")

            ae, ae_loss = train_ae_warmup(sub_data, n_train, total_epochs, warmup, seed)
            recon = eval_recon(ae, test_x)
            geom = eval_geometry(ae, test_x)
            print(f"    Recon MSE: {recon:.6f}, Cond: {geom['mean_cond']:.2f}")

            # SDE pipeline
            dummy = SDEPipelineTrainer(
                ae, DriftNet(d).to(DEVICE), DiffusionNet(d).to(DEVICE), device=DEVICE,
            )
            z_pre, dphi_pre, d2phi_pre = dummy.precompute_decoder_derivatives(x_sub.to(DEVICE))

            torch.manual_seed(seed + 100)
            drift_net = DriftNet(d).to(DEVICE)
            bs = min(BATCH_SIZE, n_train)
            dp = SDEPipelineTrainer(ae, drift_net, DiffusionNet(d).to(DEVICE), device=DEVICE)
            drift_losses = dp.train_stage2_precomputed(
                z_pre, dphi_pre, d2phi_pre, v_sub, Lambda_sub,
                epochs=epochs_sde, lr=LR_SDE, batch_size=bs,
                print_interval=max(1, epochs_sde // 5),
            )
            drift_net.eval()

            torch.manual_seed(seed + 200)
            diff_net = DiffusionNet(d).to(DEVICE)
            pipeline = SDEPipelineTrainer(ae, drift_net, diff_net, device=DEVICE)
            diff_losses = pipeline.train_stage3_precomputed(
                z_pre, dphi_pre, Lambda_sub,
                epochs=epochs_sde, lr=LR_SDE, batch_size=bs,
                print_interval=max(1, epochs_sde // 5),
            )

            print("    Evaluating...")
            eval_results = evaluate_pipeline(pipeline, ae, sde, seed)

            rows.append({
                "N": n_train,
                "warmup": warmup,
                "label": label,
                "ae_loss": ae_loss,
                "recon_mse": recon,
                "mean_cond": geom["mean_cond"],
                "drift_loss": drift_losses[-1],
                "diff_loss": diff_losses[-1],
                **eval_results,
            })

    # Summary
    print(f"\n\n{'='*100}")
    print("WARMUP RESULTS: T+F warmup before T+F+K")
    print(f"{'='*100}")

    header = f"{'N':>4s}  {'warmup':>6s}  {'label':>12s}  {'recon':>10s}  {'cond':>6s}  {'MTE@0.1':>8s}  {'MTE@0.5':>8s}  {'MTE@1.0':>8s}  {'W2@1.0':>8s}  {'MMD@1.0':>8s}"
    print(header)
    print("-" * len(header))
    for r in rows:
        print(
            f"{r['N']:>4d}  {r['warmup']:>6d}  {r['label']:>12s}  "
            f"{r['recon_mse']:>10.6f}  {r['mean_cond']:>6.2f}  "
            f"{r['MTE@0.1']:>8.4f}  {r['MTE@0.5']:>8.4f}  {r['MTE@1.0']:>8.4f}  "
            f"{r['W2@1.0']:>8.4f}  {r['MMD@1.0']:>8.4f}"
        )

    print(f"\nTotal time: {time.time() - t0:.1f}s")
    print("Done.")


if __name__ == "__main__":
    main()
