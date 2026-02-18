"""
Diagnostic: Error spatial distribution along trajectory paths for T+F+D vs T+F+Kf.

Key question: Does T+F+D have worse geometry specifically where trajectories visit,
despite having better AVERAGE geometry?

Approach:
1. Train both configs on sinusoidal (exact study hyperparams)
2. Simulate ground-truth trajectories to get trajectory positions
3. Evaluate Ito correction error at trajectory positions vs uniform grid
4. Check error spatial correlation with trajectory density
"""

import math
import torch
import numpy as np
import sympy as sp

from src.numeric.autoencoders import AutoEncoder
from src.numeric.datagen import sample_from_manifold
from src.numeric.geometry import (
    ambient_quadratic_variation_drift,
    regularized_metric_inverse,
    transform_covariance,
)
from src.numeric.losses import LossWeights
from src.numeric.training import MultiModelTrainer, TrainingConfig
from src.symbolic.manifold_sdes import ManifoldSDE
from src.symbolic.riemannian import RiemannianManifold
from src.symbolic.surfaces import surface

from experiments.common import SURFACE_MAP, make_model_config

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {DEVICE}")

SEED = 42
EPOCHS = 500
N_TRAIN = 2000
BATCH_SIZE = 32
LR = 0.005
TRAIN_BOUND = 1.0
BOUNDARY = 3.0

CONFIGS = {
    "T+F+Kf": LossWeights(tangent_bundle=1.0, diffeo=1.0, curvature_full=0.1),
    "T+F+D":  LossWeights(tangent_bundle=1.0, diffeo=1.0, drift=0.1),
}


def create_manifold_sde(surface_name):
    u, v = sp.symbols("u v", real=True)
    local_coord, chart = surface(SURFACE_MAP[surface_name], u, v)
    manifold = RiemannianManifold(local_coord, chart)
    local_drift = sp.Matrix([-v, u])
    local_diffusion = sp.Matrix([[1 + u**2 / 4, u + v], [0, 1 + v**2 / 4]])
    return ManifoldSDE(manifold, local_drift=local_drift, local_diffusion=local_diffusion)


def lambdify_sde(manifold_sde):
    """Minimal lambdify for ground-truth simulation."""
    from experiments.trajectory_fidelity_study import lambdify_sde as _lambdify
    return _lambdify(manifold_sde)


def simulate_gt_and_collect_positions(sde, n_traj, n_steps, dt, device):
    """Simulate ground-truth trajectories, return local coords at each step."""
    torch.manual_seed(SEED + 999)
    initial_local = (torch.rand(n_traj, 2, device=device) * 2 - 1) * TRAIN_BOUND
    sqrt_dt = math.sqrt(dt)

    torch.manual_seed(SEED + 1234)
    dW = torch.randn(n_traj, n_steps, 2, device=device)

    coords = initial_local.clone()
    alive = torch.ones(n_traj, dtype=torch.bool, device=device)
    all_positions = [coords.clone()]

    for step in range(n_steps):
        drift = sde.local_drift(coords)
        diffusion = sde.local_diffusion(coords)
        noise = dW[:, step, :]
        coords_new = coords + drift * dt + torch.bmm(
            diffusion, noise.unsqueeze(-1)
        ).squeeze(-1) * sqrt_dt
        out = (coords_new.abs() > BOUNDARY).any(dim=-1)
        alive = alive & ~out
        coords = torch.where(alive.unsqueeze(-1), coords_new, coords)
        all_positions.append(coords.clone())

    # Flatten: collect all visited positions (alive only at each step)
    return initial_local, dW, all_positions


def evaluate_at_positions(model, manifold_sde, positions, device):
    """Evaluate geometric errors at given (u,v) positions.

    Returns per-point Ito correction error and projection error.
    """
    model.eval()
    # Sample manifold data at these positions
    data = sample_from_manifold(
        manifold_sde,
        None,  # We'll override with specific positions
        n_samples=len(positions),
        seed=0,
        device=device,
    )
    # Actually, sample_from_manifold samples uniformly. We need to evaluate at specific points.
    # Let's compute everything manually.

    from src.symbolic.surfaces import surface as surface_fn
    u_sym, v_sym = sp.symbols("u v", real=True)
    local_coord, chart = surface_fn(SURFACE_MAP["sinusoidal"], u_sym, v_sym)
    manifold = RiemannianManifold(local_coord, chart)

    # Lambdify chart, Hessians, metric for evaluation
    chart_fns = [sp.lambdify((u_sym, v_sym), chart[i], "numpy") for i in range(3)]

    # Get Hessians symbolically
    hess_exprs = []
    for r in range(3):
        row = []
        for j in range(2):
            col = []
            for k in range(2):
                col.append(sp.diff(chart[r], local_coord[j], local_coord[k]))
            row.append(col)
        hess_exprs.append(row)
    hess_fns = [[[sp.lambdify((u_sym, v_sym), hess_exprs[r][j][k], "numpy")
                   for k in range(2)] for j in range(2)] for r in range(3)]

    # Compute ambient positions, Hessians, drift, covariance at given local coords
    uv = positions.detach().cpu().numpy()
    u_np, v_np = uv[:, 0], uv[:, 1]
    B = len(positions)

    # Chart
    x_true = np.stack([f(u_np, v_np) for f in chart_fns], axis=-1)
    x_true = torch.tensor(x_true, dtype=torch.float32, device=device)

    # True Hessians (B, 3, 2, 2)
    hessians = np.zeros((B, 3, 2, 2))
    for r in range(3):
        for j in range(2):
            for k in range(2):
                hessians[:, r, j, k] = hess_fns[r][j][k](u_np, v_np)
    hessians = torch.tensor(hessians, dtype=torch.float32, device=device)

    # True local covariance: sigma @ sigma^T where sigma is local diffusion
    sde_obj = create_manifold_sde("sinusoidal")
    sde_lamb = lambdify_sde(sde_obj)
    sigma_local = sde_lamb.local_diffusion(positions)  # (B, 2, 2)
    local_cov_true = torch.bmm(sigma_local, sigma_local.mT)  # (B, 2, 2)

    # True Ito correction
    q_true = 0.5 * ambient_quadratic_variation_drift(local_cov_true, hessians)

    # Ambient drift and covariance
    mu = sde_lamb.ambient_drift(positions)        # (B, 3)
    cov = sde_lamb.ambient_covariance(positions)  # (B, 3, 3)

    # Now evaluate model
    with torch.no_grad():
        z = model.encoder(x_true)
        dphi = model.jacobian_decoder(z)
        d2phi = model.hessian_decoder(z)
        g = torch.bmm(dphi.mT, dphi)
        g_inv = regularized_metric_inverse(g)
        p_hat = torch.bmm(dphi, torch.bmm(g_inv, dphi.mT))

        # Simulation-style pullback
        dphi_T = dphi.mT
        sigma_sim = torch.bmm(
            g_inv,
            torch.bmm(dphi_T, torch.bmm(cov, torch.bmm(dphi, g_inv)))
        )

        # Model Ito correction (simulation formula)
        q_model_raw = ambient_quadratic_variation_drift(sigma_sim, d2phi)
        q_model = 0.5 * q_model_raw

        # Per-point errors
        ito_err = torch.norm(q_model - q_true, dim=-1)  # (B,)

        # True projection (from chart Jacobian)
        # dphi_true = chart Jacobian at (u,v)
        jac_exprs = []
        for r in range(3):
            row = []
            for j in range(2):
                row.append(sp.diff(chart[r], local_coord[j]))
            jac_exprs.append(row)
        jac_fns = [[sp.lambdify((u_sym, v_sym), jac_exprs[r][j], "numpy")
                     for j in range(2)] for r in range(3)]
        dphi_true_np = np.zeros((B, 3, 2))
        for r in range(3):
            for j in range(2):
                dphi_true_np[:, r, j] = jac_fns[r][j](u_np, v_np)
        dphi_true = torch.tensor(dphi_true_np, dtype=torch.float32, device=device)
        g_true = torch.bmm(dphi_true.mT, dphi_true)
        g_true_inv = regularized_metric_inverse(g_true)
        p_true = torch.bmm(dphi_true, torch.bmm(g_true_inv, dphi_true.mT))

        proj_err = torch.norm(p_hat - p_true, dim=(-2, -1))  # (B,)

        # z-drift error (what simulation actually uses)
        residual_true = mu - 0.5 * q_true
        residual_model = mu - 0.5 * q_model_raw  # sim uses 0.5*raw
        # Wait, sim computes: q = ambient_quadratic_variation_drift(Sigma, hessian)
        # then: residual = b - 0.5 * q
        # So residual_model = mu - 0.5 * q_model_raw
        mu_z_true = torch.bmm(g_inv, torch.bmm(dphi_T, residual_true.unsqueeze(-1))).squeeze(-1)
        mu_z_model = torch.bmm(g_inv, torch.bmm(dphi_T, residual_model.unsqueeze(-1))).squeeze(-1)
        z_drift_err = torch.norm(mu_z_model - mu_z_true, dim=-1)  # (B,)

    return {
        "ito_err": ito_err.cpu(),
        "proj_err": proj_err.cpu(),
        "z_drift_err": z_drift_err.cpu(),
        "positions": positions.cpu(),
    }


def main():
    surface_name = "sinusoidal"
    print(f"\n{'='*70}")
    print(f"Error Spatial Distribution: {surface_name}")
    print(f"{'='*70}")

    manifold_sde = create_manifold_sde(surface_name)
    sde = lambdify_sde(manifold_sde)

    # Sample training data
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    train_data = sample_from_manifold(
        manifold_sde,
        [(-TRAIN_BOUND, TRAIN_BOUND), (-TRAIN_BOUND, TRAIN_BOUND)],
        n_samples=N_TRAIN,
        seed=SEED,
        device=DEVICE,
    )

    # Simulate ground-truth trajectories to get visited positions
    print("\nSimulating ground-truth trajectories...")
    n_traj = 200
    dt = 0.01
    T_max = 1.0  # Focus on t=1.0 where MTE is measured
    n_steps = int(T_max / dt)
    initial_local, dW, all_positions = simulate_gt_and_collect_positions(
        sde, n_traj, n_steps, dt, DEVICE
    )

    # Collect trajectory positions at several time snapshots
    snapshot_steps = [10, 25, 50, 100]  # t=0.1, 0.25, 0.5, 1.0
    traj_positions = {}
    for step in snapshot_steps:
        traj_positions[step * dt] = all_positions[step]

    # Also create uniform test grid for comparison
    torch.manual_seed(SEED + 500)
    n_uniform = 500
    uniform_positions = (torch.rand(n_uniform, 2, device=DEVICE) * 2 - 1) * TRAIN_BOUND

    # Train both configs and evaluate
    models = {}
    for config_name, loss_weights in CONFIGS.items():
        print(f"\n--- Training: {config_name} ---")
        trainer = MultiModelTrainer(TrainingConfig(
            epochs=EPOCHS,
            n_samples=N_TRAIN,
            input_dim=3,
            hidden_dim=64,
            latent_dim=2,
            learning_rate=LR,
            batch_size=BATCH_SIZE,
            test_size=0.03,
            print_interval=max(1, EPOCHS // 5),
            device=DEVICE,
        ))
        trainer.add_model(make_model_config(config_name, loss_weights))

        data_loader = trainer.create_data_loader(train_data)
        for epoch in range(EPOCHS):
            losses = trainer.train_epoch(data_loader)
            if (epoch + 1) % 100 == 0:
                print(f"  Epoch {epoch+1}: {losses[config_name]:.6f}")

        models[config_name] = trainer.models[config_name]

    # Evaluate at uniform positions
    print(f"\n{'='*70}")
    print("UNIFORM GRID EVALUATION")
    print(f"{'='*70}")
    for config_name, model in models.items():
        results = evaluate_at_positions(model, manifold_sde, uniform_positions, DEVICE)
        print(f"\n  {config_name} (uniform, n={n_uniform}):")
        print(f"    Ito err:     mean={results['ito_err'].mean():.6f}, std={results['ito_err'].std():.6f}, "
              f"max={results['ito_err'].max():.6f}, p95={results['ito_err'].quantile(0.95):.6f}")
        print(f"    Proj err:    mean={results['proj_err'].mean():.6f}, std={results['proj_err'].std():.6f}, "
              f"max={results['proj_err'].max():.6f}, p95={results['proj_err'].quantile(0.95):.6f}")
        print(f"    z-drift err: mean={results['z_drift_err'].mean():.6f}, std={results['z_drift_err'].std():.6f}, "
              f"max={results['z_drift_err'].max():.6f}, p95={results['z_drift_err'].quantile(0.95):.6f}")

    # Evaluate along trajectory positions
    print(f"\n{'='*70}")
    print("TRAJECTORY PATH EVALUATION")
    print(f"{'='*70}")
    for t, positions in traj_positions.items():
        print(f"\n  --- Time t={t:.2f} ({len(positions)} points) ---")
        for config_name, model in models.items():
            results = evaluate_at_positions(model, manifold_sde, positions, DEVICE)
            print(f"    {config_name}:")
            print(f"      Ito err:     mean={results['ito_err'].mean():.6f}, std={results['ito_err'].std():.6f}, "
                  f"max={results['ito_err'].max():.6f}, p95={results['ito_err'].quantile(0.95):.6f}")
            print(f"      Proj err:    mean={results['proj_err'].mean():.6f}, std={results['proj_err'].std():.6f}, "
                  f"max={results['proj_err'].max():.6f}, p95={results['proj_err'].quantile(0.95):.6f}")
            print(f"      z-drift err: mean={results['z_drift_err'].mean():.6f}, std={results['z_drift_err'].std():.6f}, "
                  f"max={results['z_drift_err'].max():.6f}, p95={results['z_drift_err'].quantile(0.95):.6f}")

    # Spatial analysis: bin positions by radius and compare errors
    print(f"\n{'='*70}")
    print("SPATIAL BINNING BY RADIUS")
    print(f"{'='*70}")
    traj_at_1 = traj_positions[1.0]
    radius = torch.norm(traj_at_1, dim=-1).cpu()
    bins = [(0, 0.5), (0.5, 1.0), (1.0, 1.5), (1.5, 2.0), (2.0, 3.0)]

    for config_name, model in models.items():
        results = evaluate_at_positions(model, manifold_sde, traj_at_1, DEVICE)
        print(f"\n  {config_name} at t=1.0:")
        for lo, hi in bins:
            mask = (radius >= lo) & (radius < hi)
            n = mask.sum().item()
            if n == 0:
                continue
            ito_bin = results['ito_err'][mask]
            zd_bin = results['z_drift_err'][mask]
            print(f"    r=[{lo:.1f},{hi:.1f}): n={n:3d}, "
                  f"ito_err={ito_bin.mean():.6f}, z_drift_err={zd_bin.mean():.6f}")

    print("\n\nDONE.")


if __name__ == "__main__":
    main()
