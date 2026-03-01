"""
Diagnostic: Why does a geometrically better AE (T+F+K) produce worse SDE trajectories
on hyperbolic_paraboloid?

Hypothesis: T+F+K creates a latent space where the drift function is more complex,
making it harder for the small DriftNet [64,64] to learn.

For each surface × AE config (T+F, T+F+K):
1. Train AE, precompute derivatives
2. Analyze the latent drift target: b_z_target = ginv @ dphi^T @ (v - q)
3. Measure: smoothness (Lipschitz estimate), complexity (gradient norms),
   dynamic range, and how well the drift net fits
4. Train drift net and diffusion net, evaluate MTE
5. Also test with LARGER drift net [128, 128, 128] to see if capacity fixes it

Usage:
    PYTHONUNBUFFERED=1 python -m experiments.diagnose_drift_landscape
"""

import time
import torch
import numpy as np

from src.numeric.datagen import sample_from_manifold
from src.numeric.losses import LossWeights
from src.numeric.sde_nets import DriftNet, DiffusionNet
from src.numeric.sde_training import SDEPipelineTrainer
from src.numeric.training import MultiModelTrainer, TrainingConfig
from src.numeric.geometry import regularized_metric_inverse, curvature_drift_explicit_full

from experiments.common import make_model_config
from experiments.data_driven_sde import (
    create_manifold_sde, evaluate_pipeline, lambdify_sde,
    TRAIN_BOUND, N_TRAIN, BATCH_SIZE, LR_AE, LR_SDE, SEED, DEVICE,
)


def train_ae(train_data, epochs, lw, name, seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    trainer = MultiModelTrainer(TrainingConfig(
        epochs=epochs, n_samples=N_TRAIN, input_dim=3, hidden_dim=64,
        latent_dim=2, learning_rate=LR_AE, batch_size=BATCH_SIZE,
        test_size=0.03, print_interval=max(1, epochs // 5), device=DEVICE,
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


def compute_latent_drift_target(ae, x, v, Lambda):
    """Compute the target latent drift b_z = ginv @ dphi^T @ (v - q)."""
    with torch.no_grad():
        z = ae.encoder(x).detach()
        dphi = ae.decoder.jacobian_network(z)
        d2phi = ae.decoder.hessian_network(z)

        g = dphi.mT @ dphi
        ginv = regularized_metric_inverse(g)
        pinv = ginv @ dphi.mT
        P_hat = dphi @ pinv
        P_hat = 0.5 * (P_hat + P_hat.mT)

        # Pull back covariance
        Lambda_tan = P_hat @ Lambda @ P_hat
        Lambda_tan = 0.5 * (Lambda_tan + Lambda_tan.mT)
        Sigma_z = pinv @ Lambda_tan @ pinv.mT
        Sigma_z = 0.5 * (Sigma_z + Sigma_z.mT)

        # Ito correction
        q = curvature_drift_explicit_full(d2phi, Sigma_z)

        # Target latent drift: b_z = ginv @ dphi^T @ (v - q)
        ambient_target = v - q  # (B, D)
        b_z_target = (pinv @ ambient_target.unsqueeze(-1)).squeeze(-1)  # (B, d)

    return z, b_z_target, dphi, d2phi, Sigma_z, q


def analyze_drift_complexity(z, b_z_target):
    """Measure drift function complexity in latent space."""
    B = z.shape[0]

    # Dynamic range
    b_norm = b_z_target.norm(dim=-1)
    b_mean = b_z_target.mean(dim=0)
    b_std = b_z_target.std(dim=0)

    # Lipschitz estimate: max ||b(z_i) - b(z_j)|| / ||z_i - z_j||
    # Use random pairs to estimate
    n_pairs = min(5000, B * (B - 1) // 2)
    idx1 = torch.randint(0, B, (n_pairs,))
    idx2 = torch.randint(0, B, (n_pairs,))
    mask = idx1 != idx2
    idx1, idx2 = idx1[mask], idx2[mask]

    dz = (z[idx1] - z[idx2]).norm(dim=-1).clamp(min=1e-8)
    db = (b_z_target[idx1] - b_z_target[idx2]).norm(dim=-1)
    lipschitz_ratios = db / dz

    # Nearest-neighbor smoothness: for each point, how different is its drift
    # from its nearest neighbor's drift?
    with torch.no_grad():
        dists = torch.cdist(z, z)
        dists.fill_diagonal_(float('inf'))
        nn_idx = dists.argmin(dim=1)
        nn_dz = dists[torch.arange(B), nn_idx]
        nn_db = (b_z_target - b_z_target[nn_idx]).norm(dim=-1)
        nn_ratio = nn_db / nn_dz.clamp(min=1e-8)

    # Latent space coverage
    z_range = z.max(dim=0).values - z.min(dim=0).values

    return {
        "||b_z|| mean": b_norm.mean().item(),
        "||b_z|| std": b_norm.std().item(),
        "||b_z|| max": b_norm.max().item(),
        "b_z range (dim0)": (b_z_target[:, 0].max() - b_z_target[:, 0].min()).item(),
        "b_z range (dim1)": (b_z_target[:, 1].max() - b_z_target[:, 1].min()).item(),
        "Lipschitz est (mean)": lipschitz_ratios.mean().item(),
        "Lipschitz est (95th)": lipschitz_ratios.quantile(0.95).item(),
        "Lipschitz est (max)": lipschitz_ratios.max().item(),
        "NN smoothness (mean)": nn_ratio.mean().item(),
        "NN smoothness (95th)": nn_ratio.quantile(0.95).item(),
        "z range (dim0)": z_range[0].item(),
        "z range (dim1)": z_range[1].item(),
    }


def run_sde_pipeline(ae, z_pre, dphi_pre, d2phi_pre, v, Lambda, sde, seed,
                     epochs_sde=300, drift_hidden=None, label=""):
    """Train drift+diffusion and evaluate."""
    d = 2
    if drift_hidden is None:
        drift_hidden = [64, 64]

    torch.manual_seed(seed + 100)
    drift_net = DriftNet(d, hidden_dims=drift_hidden).to(DEVICE)
    dp = SDEPipelineTrainer(ae, drift_net, DiffusionNet(d).to(DEVICE), device=DEVICE)
    drift_losses = dp.train_stage2_precomputed(
        z_pre, dphi_pre, d2phi_pre, v, Lambda,
        epochs=epochs_sde, lr=LR_SDE, batch_size=BATCH_SIZE,
        print_interval=max(1, epochs_sde // 5),
    )
    drift_net.eval()

    # Evaluate drift fit: compute residual on training data
    with torch.no_grad():
        b_pred = drift_net(z_pre)
        # Compute target for comparison
        g = dphi_pre.mT @ dphi_pre
        ginv = regularized_metric_inverse(g)
        pinv = ginv @ dphi_pre.mT
        P_hat = dphi_pre @ pinv
        P_hat = 0.5 * (P_hat + P_hat.mT)
        Lambda_tan = P_hat @ Lambda @ P_hat
        Lambda_tan = 0.5 * (Lambda_tan + Lambda_tan.mT)
        Sigma_z = pinv @ Lambda_tan @ pinv.mT
        Sigma_z = 0.5 * (Sigma_z + Sigma_z.mT)
        q = curvature_drift_explicit_full(d2phi_pre, Sigma_z)
        b_target = (pinv @ (v - q).unsqueeze(-1)).squeeze(-1)
        drift_residual = ((b_pred - b_target) ** 2).sum(-1).mean().item()

    torch.manual_seed(seed + 200)
    diff_net = DiffusionNet(d).to(DEVICE)
    pipeline = SDEPipelineTrainer(ae, drift_net, diff_net, device=DEVICE)
    diff_losses = pipeline.train_stage3_precomputed(
        z_pre, dphi_pre, Lambda,
        epochs=epochs_sde, lr=LR_SDE, batch_size=BATCH_SIZE,
        print_interval=max(1, epochs_sde // 5),
    )

    print(f"    {label} Evaluating...")
    eval_results = evaluate_pipeline(pipeline, ae, sde, seed)

    return {
        "drift_loss": drift_losses[-1],
        "drift_residual": drift_residual,
        "diff_loss": diff_losses[-1],
        **eval_results,
    }


def main():
    surfaces = ["sinusoidal", "paraboloid", "hyperbolic_paraboloid"]
    epochs_ae = 500
    epochs_sde = 300
    seed = SEED

    print(f"Device: {DEVICE}")
    print(f"Investigating drift landscape across surfaces\n")

    t0 = time.time()
    all_rows = []

    for surface_name in surfaces:
        print(f"\n{'='*70}")
        print(f"Surface: {surface_name}")
        print(f"{'='*70}")

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
        sde = lambdify_sde(create_manifold_sde(surface_name))

        for reg_name, lw in [("T+F", LossWeights(tangent_bundle=1.0, diffeo=1.0)),
                              ("T+F+K", LossWeights(tangent_bundle=1.0, diffeo=1.0, curvature=0.1))]:
            print(f"\n  --- AE: {reg_name} ---")
            ae, ae_loss = train_ae(train_data, epochs_ae, lw, "ae", seed)

            # Compute latent drift target
            z, b_z_target, dphi, d2phi, Sigma_z, q = compute_latent_drift_target(ae, x, v, Lambda)

            # Analyze complexity
            complexity = analyze_drift_complexity(z, b_z_target)
            print(f"\n    Drift complexity:")
            for k, val in complexity.items():
                print(f"      {k}: {val:.6f}")

            # Precompute for training
            dummy = SDEPipelineTrainer(ae, DriftNet(2).to(DEVICE), DiffusionNet(2).to(DEVICE), device=DEVICE)
            z_pre, dphi_pre, d2phi_pre = dummy.precompute_decoder_derivatives(x)

            # Standard drift net [64, 64]
            print(f"\n    --- DriftNet [64, 64] ---")
            results_small = run_sde_pipeline(
                ae, z_pre, dphi_pre, d2phi_pre, v, Lambda, sde, seed,
                drift_hidden=[64, 64], label="[64,64]",
            )
            print(f"      drift_residual: {results_small['drift_residual']:.6f}")
            print(f"      MTE@1.0: {results_small['MTE@1.0']:.4f}")

            # Larger drift net [128, 128, 128]
            print(f"\n    --- DriftNet [128, 128, 128] ---")
            results_large = run_sde_pipeline(
                ae, z_pre, dphi_pre, d2phi_pre, v, Lambda, sde, seed,
                drift_hidden=[128, 128, 128], label="[128,128,128]",
            )
            print(f"      drift_residual: {results_large['drift_residual']:.6f}")
            print(f"      MTE@1.0: {results_large['MTE@1.0']:.4f}")

            all_rows.append({
                "surface": surface_name,
                "ae_reg": reg_name,
                "ae_loss": ae_loss,
                **{f"cplx_{k}": val for k, val in complexity.items()},
                "small_drift_loss": results_small["drift_loss"],
                "small_drift_resid": results_small["drift_residual"],
                "small_MTE": results_small["MTE@1.0"],
                "small_W2": results_small["W2@1.0"],
                "large_drift_loss": results_large["drift_loss"],
                "large_drift_resid": results_large["drift_residual"],
                "large_MTE": results_large["MTE@1.0"],
                "large_W2": results_large["W2@1.0"],
            })

    # Summary
    print(f"\n\n{'='*120}")
    print("DRIFT LANDSCAPE ANALYSIS")
    print(f"{'='*120}")

    header = (
        f"{'surface':>25s}  {'reg':>6s}  "
        f"{'Lip(95th)':>10s}  {'NN_smooth':>10s}  {'||b_z||':>8s}  "
        f"{'sm_resid':>10s}  {'sm_MTE':>8s}  {'sm_W2':>8s}  "
        f"{'lg_resid':>10s}  {'lg_MTE':>8s}  {'lg_W2':>8s}"
    )
    print(header)
    print("-" * len(header))
    for r in all_rows:
        print(
            f"{r['surface']:>25s}  {r['ae_reg']:>6s}  "
            f"{r['cplx_Lipschitz est (95th)']:>10.4f}  "
            f"{r['cplx_NN smoothness (mean)']:>10.4f}  "
            f"{r['cplx_||b_z|| mean']:>8.4f}  "
            f"{r['small_drift_resid']:>10.6f}  "
            f"{r['small_MTE']:>8.4f}  "
            f"{r['small_W2']:>8.4f}  "
            f"{r['large_drift_resid']:>10.6f}  "
            f"{r['large_MTE']:>8.4f}  "
            f"{r['large_W2']:>8.4f}"
        )

    # Does larger drift net fix the problem?
    print(f"\n\nKEY QUESTION: Does a larger drift net fix the T+F+K gap on hyperbolic_paraboloid?")
    for surface_name in surfaces:
        tf = [r for r in all_rows if r["surface"] == surface_name and r["ae_reg"] == "T+F"][0]
        tfk = [r for r in all_rows if r["surface"] == surface_name and r["ae_reg"] == "T+F+K"][0]
        print(f"\n  {surface_name}:")
        print(f"    Small DriftNet:  T+F MTE={tf['small_MTE']:.4f}  T+F+K MTE={tfk['small_MTE']:.4f}  delta={tfk['small_MTE'] - tf['small_MTE']:+.4f}")
        print(f"    Large DriftNet:  T+F MTE={tf['large_MTE']:.4f}  T+F+K MTE={tfk['large_MTE']:.4f}  delta={tfk['large_MTE'] - tf['large_MTE']:+.4f}")
        print(f"    Lipschitz(95th): T+F={tf['cplx_Lipschitz est (95th)']:.4f}  T+F+K={tfk['cplx_Lipschitz est (95th)']:.4f}")

    print(f"\nTotal time: {time.time() - t0:.1f}s")
    print("Done.")


if __name__ == "__main__":
    main()
