"""
Trajectory Fidelity Study for AEML

Validates that learned geometry faithfully reproduces manifold dynamics
through actual SDE trajectory simulation.

Compares three simulation modes:
1. Ground truth: SDE in true local coords, mapped via true chart
2. Learned latent: TRUE SDE dynamics, mapped to ambient via learned autoencoder
3. End-to-end: Invert ambient SDE through learned chart, step in z-space

Metrics:
- Path-wise (T<=1.0): MTE (mean trajectory error), RPD (relative path divergence)
- Distributional (T<=5.0): W2 (Wasserstein-2), MMD (max mean discrepancy)

Usage:
    python -m experiments.trajectory_fidelity_study --surface paraboloid
    python -m experiments.trajectory_fidelity_study --surface all --epochs 500
"""

import argparse
import math
import torch
import numpy as np
import pandas as pd
import sympy as sp
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
from typing import Dict, Callable, Tuple, List
from dataclasses import dataclass

from src.numeric.autoencoders import AutoEncoder
from src.numeric.datagen import sample_from_manifold
from src.numeric.geometry import ambient_quadratic_variation_drift, regularized_metric_inverse
from src.numeric.training import MultiModelTrainer, TrainingConfig
from src.symbolic.manifold_sdes import ManifoldSDE
from src.symbolic.riemannian import RiemannianManifold
from src.symbolic.surfaces import surface

from experiments.common import SURFACE_MAP, PENALTY_CONFIGS, make_model_config

try:
    import ot
    HAS_POT = True
except ImportError:
    HAS_POT = False
    print("Warning: POT not installed. W2 metrics will be skipped. Install with: pip install POT")


# Surfaces used in this study
STUDY_SURFACES = ["paraboloid", "hyperbolic_paraboloid", "sinusoidal"]

# Time grids for metrics
PATH_TIMES = [0.1, 0.2, 0.5, 1.0]
DIST_TIMES = [1.0, 2.0, 3.0, 4.0, 5.0]


def sample_ring_initial(
    n: int,
    inner: float,
    outer: float,
    device: str,
    seed: int,
) -> torch.Tensor:
    """Sample 2D points uniformly from [-outer, outer]^2 \\ [-inner, inner]^2.

    Args:
        n: Number of points to sample.
        inner: Half-side of the inner square to exclude.
        outer: Half-side of the outer square to sample within.
        device: Torch device.
        seed: Random seed.

    Returns:
        (n, 2) tensor of local coordinates in the ring region.
    """
    torch.manual_seed(seed)
    oversample = 8
    pts = (torch.rand(n * oversample, 2, device=device) * 2 - 1) * outer
    in_inner = (pts.abs() <= inner).all(dim=-1)
    ring_pts = pts[~in_inner]
    if len(ring_pts) >= n:
        return ring_pts[:n]
    # If not enough, return what we have (shouldn't happen with 8x oversample)
    return ring_pts


# ============================================================================
# Lambdification utilities
# ============================================================================

@dataclass
class LambdifiedSDE:
    """Batched torch-callable SDE coefficient functions."""
    local_drift: Callable           # (B, 2) tensor -> (B, 2) tensor
    local_diffusion: Callable       # (B, 2) tensor -> (B, 2, 2) tensor
    chart: Callable                 # (B, 2) tensor -> (B, 3) tensor
    ambient_drift: Callable         # (B, 2) tensor -> (B, 3) tensor
    ambient_covariance: Callable    # (B, 2) tensor -> (B, 3, 3) tensor


def lambdify_sde(manifold_sde: ManifoldSDE) -> LambdifiedSDE:
    """Convert symbolic SDE coefficients to batched torch-callable functions.

    Each returned function accepts a (B, 2) tensor of local coordinates
    and returns the corresponding batched output tensor.
    """
    coords = manifold_sde.manifold.local_coordinates
    d = manifold_sde.intrinsic_dim
    D = manifold_sde.extrinsic_dim

    # Lambdify each scalar component with numpy
    drift_fns = [sp.lambdify(coords, manifold_sde.local_drift[i], "numpy")
                 for i in range(d)]
    diff_fns = [[sp.lambdify(coords, manifold_sde.local_diffusion[i, j], "numpy")
                 for j in range(d)] for i in range(d)]
    chart_fns = [sp.lambdify(coords, manifold_sde.manifold.chart[i], "numpy")
                 for i in range(D)]
    amb_drift_fns = [sp.lambdify(coords, manifold_sde.ambient_drift[i], "numpy")
                     for i in range(D)]
    amb_cov_fns = [[sp.lambdify(coords, manifold_sde.ambient_covariance[i, j], "numpy")
                    for j in range(D)] for i in range(D)]

    def local_drift_batch(uv: torch.Tensor) -> torch.Tensor:
        u = uv[:, 0].detach().cpu().numpy()
        v = uv[:, 1].detach().cpu().numpy()
        result = np.stack([f(u, v) for f in drift_fns], axis=-1)
        return torch.tensor(result, dtype=uv.dtype, device=uv.device)

    def local_diffusion_batch(uv: torch.Tensor) -> torch.Tensor:
        u = uv[:, 0].detach().cpu().numpy()
        v = uv[:, 1].detach().cpu().numpy()
        B = len(u)
        result = np.zeros((B, d, d))
        for i in range(d):
            for j in range(d):
                result[:, i, j] = diff_fns[i][j](u, v)
        return torch.tensor(result, dtype=uv.dtype, device=uv.device)

    def chart_batch(uv: torch.Tensor) -> torch.Tensor:
        u = uv[:, 0].detach().cpu().numpy()
        v = uv[:, 1].detach().cpu().numpy()
        result = np.stack([f(u, v) for f in chart_fns], axis=-1)
        return torch.tensor(result, dtype=uv.dtype, device=uv.device)

    def ambient_drift_batch(uv: torch.Tensor) -> torch.Tensor:
        u = uv[:, 0].detach().cpu().numpy()
        v = uv[:, 1].detach().cpu().numpy()
        result = np.stack([f(u, v) for f in amb_drift_fns], axis=-1)
        return torch.tensor(result, dtype=uv.dtype, device=uv.device)

    def ambient_covariance_batch(uv: torch.Tensor) -> torch.Tensor:
        u = uv[:, 0].detach().cpu().numpy()
        v = uv[:, 1].detach().cpu().numpy()
        B = len(u)
        result = np.zeros((B, D, D))
        for i in range(D):
            for j in range(D):
                result[:, i, j] = amb_cov_fns[i][j](u, v)
        return torch.tensor(result, dtype=uv.dtype, device=uv.device)

    return LambdifiedSDE(
        local_drift=local_drift_batch,
        local_diffusion=local_diffusion_batch,
        chart=chart_batch,
        ambient_drift=ambient_drift_batch,
        ambient_covariance=ambient_covariance_batch,
    )


# ============================================================================
# Trajectory simulation
# ============================================================================

def simulate_ground_truth(
    initial_local: torch.Tensor,
    sde: LambdifiedSDE,
    n_steps: int,
    dt: float,
    dW: torch.Tensor,
    boundary: float,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Simulate ground-truth trajectories in local coords, map to ambient.

    Args:
        initial_local: (B, 2) starting local coordinates
        sde: lambdified SDE coefficients
        n_steps: number of Euler-Maruyama steps
        dt: step size
        dW: (B, n_steps, 2) pre-generated Brownian increments
        boundary: hard boundary in local coordinates

    Returns:
        ambient_traj: (B, n_steps+1, D) trajectories in ambient space
        alive: (B, n_steps+1) boolean mask (True = trajectory still active)
    """
    B = initial_local.shape[0]
    device = initial_local.device
    sqrt_dt = math.sqrt(dt)

    alive = torch.ones(B, n_steps + 1, dtype=torch.bool, device=device)

    coords = initial_local.clone()
    x0 = sde.chart(coords)
    D = x0.shape[-1]
    ambient_traj = torch.zeros(B, n_steps + 1, D, device=device)
    ambient_traj[:, 0] = x0

    for step in range(n_steps):
        drift = sde.local_drift(coords)
        diffusion = sde.local_diffusion(coords)
        noise = dW[:, step, :]

        coords_new = coords + drift * dt + torch.bmm(
            diffusion, noise.unsqueeze(-1)
        ).squeeze(-1) * sqrt_dt

        out = (coords_new.abs() > boundary).any(dim=-1)
        alive[:, step + 1] = alive[:, step] & ~out
        coords = torch.where(alive[:, step + 1].unsqueeze(-1), coords_new, coords)
        ambient_traj[:, step + 1] = sde.chart(coords)

    return ambient_traj, alive


def simulate_learned_latent(
    model: AutoEncoder,
    initial_local: torch.Tensor,
    sde: LambdifiedSDE,
    n_steps: int,
    dt: float,
    dW: torch.Tensor,
    boundary: float,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Simulate TRUE SDE in TRUE local coords, map to ambient via learned model.

    The SDE dynamics are identical to ground truth (same local drift,
    diffusion, and Brownian increments in the true chart).  The only
    difference is the mapping to ambient space: ground truth uses the
    true chart phi, whereas this function maps through the learned
    autoencoder (decoder . encoder . phi).  MTE therefore isolates the
    autoencoder's reconstruction fidelity along SDE trajectories.

    Args:
        model: Trained autoencoder.
        initial_local: (B, 2) starting TRUE local coordinates.
        sde: Lambdified SDE (true chart coefficients).
        n_steps: Number of Euler-Maruyama steps.
        dt: Step size.
        dW: (B, n_steps, 2) pre-generated Brownian increments.
        boundary: Hard boundary in local coordinates.

    Returns:
        ambient_traj: (B, n_steps+1, 3)
        alive: (B, n_steps+1) bool mask
    """
    model.eval()
    B = initial_local.shape[0]
    device = initial_local.device
    sqrt_dt = math.sqrt(dt)

    alive = torch.ones(B, n_steps + 1, dtype=torch.bool, device=device)

    coords = initial_local.clone()

    # Map initial position through learned model: phi -> encoder -> decoder
    with torch.no_grad():
        x_true = sde.chart(coords)
        x0 = model.decoder(model.encoder(x_true))
        D = x0.shape[-1]
    ambient_traj = torch.zeros(B, n_steps + 1, D, device=device)
    ambient_traj[:, 0] = x0

    for step in range(n_steps):
        # TRUE SDE coefficients at TRUE local coordinates
        drift = sde.local_drift(coords)
        diffusion = sde.local_diffusion(coords)
        noise = dW[:, step, :]

        coords_new = coords + drift * dt + torch.bmm(
            diffusion, noise.unsqueeze(-1)
        ).squeeze(-1) * sqrt_dt

        out = (coords_new.abs() > boundary).any(dim=-1)
        alive[:, step + 1] = alive[:, step] & ~out
        coords = torch.where(alive[:, step + 1].unsqueeze(-1), coords_new, coords)

        # Map to ambient via learned model (decoder . encoder . true_chart)
        with torch.no_grad():
            x_true = sde.chart(coords)
            ambient_traj[:, step + 1] = model.decoder(model.encoder(x_true))

    return ambient_traj, alive


def simulate_end_to_end(
    model: AutoEncoder,
    initial_ambient: torch.Tensor,
    sde: LambdifiedSDE,
    n_steps: int,
    dt: float,
    dW: torch.Tensor,
    boundary: float,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """End-to-end ATLAS pipeline simulation: invert ambient SDE through learned chart.

    At each step:
    1. Compute learned geometry (Jacobian, Hessian, metric) at z
    2. Decode z to ambient x, extract true local coords via Monge projection
    3. Evaluate TRUE ambient SDE coefficients b(u,v), Lambda(u,v)
    4. Pull back covariance: Sigma = g_inv @ dphi^T @ Lambda @ dphi @ g_inv
    5. Ito correction via LEARNED Hessian: q = tr(Sigma * H)
    6. Invert drift: mu = g_inv @ dphi^T @ (b - 0.5*q)
    7. Diffusion: sigma = cholesky(Sigma + eps*I)
    8. Euler-Maruyama step in z-space

    This tests whether the learned Hessian accuracy (improved by K penalty)
    translates to better trajectory quality through more accurate Ito correction.

    Args:
        model: Trained autoencoder.
        initial_ambient: (B, 3) starting ambient positions.
        sde: Lambdified SDE with ambient_drift and ambient_covariance.
        n_steps: Number of Euler-Maruyama steps.
        dt: Step size.
        dW: (B, n_steps, 2) pre-generated Brownian increments.
        boundary: Hard boundary in latent coordinates.

    Returns:
        ambient_traj: (B, n_steps+1, D) trajectories in ambient space.
        alive: (B, n_steps+1) boolean mask.
    """
    model.eval()
    B = initial_ambient.shape[0]
    device = initial_ambient.device
    sqrt_dt = math.sqrt(dt)

    D = initial_ambient.shape[-1]
    ambient_traj = torch.zeros(B, n_steps + 1, D, device=device)
    alive = torch.ones(B, n_steps + 1, dtype=torch.bool, device=device)

    # Encode initial ambient positions to latent space
    with torch.no_grad():
        z = model.encoder(initial_ambient)
    ambient_traj[:, 0] = initial_ambient

    for step in range(n_steps):
        # 1. Learned geometry at current z
        dphi = model.decoder_jacobian(z).detach()       # (B, 3, 2)
        hessian = model.decoder_hessian(z).detach()      # (B, 3, 2, 2)
        g = torch.bmm(dphi.mT, dphi)                    # (B, 2, 2)
        g_inv = regularized_metric_inverse(g)            # (B, 2, 2)

        # 2. Decode to ambient, extract true local coords (Monge patch)
        with torch.no_grad():
            x = model.decoder(z)
        true_local = x[:, :2].detach()                   # (B, 2)

        # 3. Evaluate TRUE ambient SDE at true local coords
        b = sde.ambient_drift(true_local)                # (B, 3)
        Lambda = sde.ambient_covariance(true_local)      # (B, 3, 3)

        # 4. Pull back covariance: Sigma = g_inv @ dphi^T @ Lambda @ dphi @ g_inv
        dphi_T = dphi.mT                                 # (B, 2, 3)
        Sigma = torch.bmm(
            g_inv,
            torch.bmm(dphi_T, torch.bmm(Lambda, torch.bmm(dphi, g_inv)))
        )                                                # (B, 2, 2)

        # 5. Ito correction via LEARNED Hessian
        q = ambient_quadratic_variation_drift(Sigma, hessian)  # (B, 3)

        # 6. Invert drift: mu = g_inv @ dphi^T @ (b - 0.5*q)
        residual = b - 0.5 * q                           # (B, 3)
        mu = torch.bmm(
            g_inv, torch.bmm(dphi_T, residual.unsqueeze(-1))
        ).squeeze(-1)                                    # (B, 2)

        # 7. Diffusion via Cholesky of regularized Sigma
        Sigma_reg = Sigma + 1e-6 * torch.eye(2, device=device, dtype=Sigma.dtype)
        try:
            sigma = torch.linalg.cholesky(Sigma_reg)     # (B, 2, 2)
        except torch.linalg.LinAlgError:
            # Fallback: eigenvalue clamping for non-PD matrices
            evals, evecs = torch.linalg.eigh(Sigma_reg)
            evals = evals.clamp(min=1e-6)
            sigma = torch.bmm(evecs, torch.diag_embed(torch.sqrt(evals)))

        # 8. Euler-Maruyama step in z-space
        noise = dW[:, step, :]                           # (B, 2)
        z_new = z.detach() + mu * dt + torch.bmm(
            sigma, noise.unsqueeze(-1)
        ).squeeze(-1) * sqrt_dt

        # 9. Boundary / NaN check
        out = (z_new.abs() > boundary).any(dim=-1) | torch.isnan(z_new).any(dim=-1) | torch.isinf(z_new).any(dim=-1)
        alive[:, step + 1] = alive[:, step] & ~out
        z = torch.where(alive[:, step + 1].unsqueeze(-1), z_new, z)

        # 10. Map to ambient for output
        with torch.no_grad():
            ambient_traj[:, step + 1] = model.decoder(z)

    return ambient_traj, alive


# ============================================================================
# Metrics
# ============================================================================

def compute_mte_at_step(
    traj_learned: torch.Tensor,
    traj_true: torch.Tensor,
    alive: torch.Tensor,
    step: int,
) -> float:
    """Mean trajectory error at a single time step.

    Args:
        traj_learned: (B, T, 3)
        traj_true: (B, T, 3)
        alive: (B,) bool mask for this step (intersection of GT and learned alive)
        step: time step index

    Returns:
        Scalar MTE value.
    """
    dist = torch.norm(traj_learned[:, step] - traj_true[:, step], dim=-1)  # (B,)
    count = alive.float().sum()
    if count.item() < 1:
        return float("nan")
    return (dist * alive.float()).sum().item() / count.item()


def compute_rpd_at_step(
    traj_learned: torch.Tensor,
    traj_true: torch.Tensor,
    alive_both: torch.Tensor,
    alive_true: torch.Tensor,
    step: int,
    eps: float = 1e-8,
) -> float:
    """Relative path divergence at a single time step.

    RPD(t) = MTE(t) / mean(||x_true(t) - x_true(0)||)
    """
    mte = compute_mte_at_step(traj_learned, traj_true, alive_both, step)
    if math.isnan(mte):
        return float("nan")
    disp = torch.norm(traj_true[:, step] - traj_true[:, 0], dim=-1)
    count_true = alive_true.float().sum()
    if count_true.item() < 1:
        return float("nan")
    mean_disp = (disp * alive_true.float()).sum().item() / count_true.item()
    return mte / (mean_disp + eps)


def compute_w2(
    pos_learned: torch.Tensor,
    pos_true: torch.Tensor,
    alive_learned: torch.Tensor,
    alive_true: torch.Tensor,
) -> float:
    """Wasserstein-2 distance between two alive point clouds."""
    if not HAS_POT:
        return float("nan")

    x = pos_learned[alive_learned].detach().cpu().numpy()
    y = pos_true[alive_true].detach().cpu().numpy()

    if len(x) < 2 or len(y) < 2:
        return float("nan")

    a = np.ones(len(x)) / len(x)
    b = np.ones(len(y)) / len(y)
    M = ot.dist(x, y, metric="sqeuclidean")
    w2_sq = ot.emd2(a, b, M)
    return float(np.sqrt(max(w2_sq, 0.0)))


def compute_mmd(
    pos_learned: torch.Tensor,
    pos_true: torch.Tensor,
    alive_learned: torch.Tensor,
    alive_true: torch.Tensor,
) -> float:
    """MMD with Gaussian kernel and median bandwidth heuristic."""
    x = pos_learned[alive_learned].detach().cpu().float()
    y = pos_true[alive_true].detach().cpu().float()

    if len(x) < 2 or len(y) < 2:
        return float("nan")

    # Median heuristic for bandwidth
    all_pts = torch.cat([x, y], dim=0)
    dists = torch.cdist(all_pts, all_pts)
    mask = dists > 0
    if mask.sum() == 0:
        return 0.0
    sigma2 = dists[mask].median().item() ** 2

    def k(a, b):
        return torch.exp(-torch.cdist(a, b) ** 2 / (2 * sigma2))

    mmd2 = k(x, x).mean() + k(y, y).mean() - 2 * k(x, y).mean()
    return float(torch.sqrt(mmd2.clamp(min=0)).item())


def compute_metrics_at_snapshots(
    traj_learned: torch.Tensor,
    traj_true: torch.Tensor,
    alive_learned: torch.Tensor,
    alive_true: torch.Tensor,
    dt: float,
    path_times: List[float],
    dist_times: List[float],
) -> List[Dict]:
    """Compute all metrics at specified snapshot times.

    Returns a list of dicts, one per snapshot time.
    """
    all_times = sorted(set(path_times + dist_times))
    n_steps = traj_learned.shape[1] - 1
    rows = []

    for t in all_times:
        step = min(int(round(t / dt)), n_steps)
        both_alive = alive_learned[:, step] & alive_true[:, step]

        row = {
            "time": t,
            "ensemble_survival": both_alive.float().mean().item(),
        }

        # Path-wise metrics
        if t <= max(path_times):
            row["MTE"] = compute_mte_at_step(
                traj_learned, traj_true, both_alive, step
            )
            row["RPD"] = compute_rpd_at_step(
                traj_learned, traj_true, both_alive, alive_true[:, step], step
            )

        # Distributional metrics
        if t >= min(dist_times):
            row["W2"] = compute_w2(
                traj_learned[:, step], traj_true[:, step],
                alive_learned[:, step], alive_true[:, step],
            )
            row["MMD"] = compute_mmd(
                traj_learned[:, step], traj_true[:, step],
                alive_learned[:, step], alive_true[:, step],
            )

        rows.append(row)

    return rows


# ============================================================================
# SDE creation
# ============================================================================

def create_manifold_sde(surface_name: str) -> ManifoldSDE:
    """Create manifold SDE with non-trivial dynamics (same as dynamics study)."""
    u, v = sp.symbols("u v", real=True)
    local_coord, chart = surface(SURFACE_MAP[surface_name], u, v)
    manifold = RiemannianManifold(local_coord, chart)

    local_drift = sp.Matrix([-v, u])
    local_diffusion = sp.Matrix([[1 + u**2 / 4, u + v], [0, 1 + v**2 / 4]])

    return ManifoldSDE(manifold, local_drift=local_drift, local_diffusion=local_diffusion)


# ============================================================================
# Plotting
# ============================================================================

def plot_mte_timeseries(df: pd.DataFrame, output_prefix: str):
    """Plot MTE vs time for each surface, colored by penalty, panels by sim_mode."""
    surfaces = df["surface"].unique()
    sim_modes = df["sim_mode"].unique()

    fig, axes = plt.subplots(
        len(surfaces), len(sim_modes),
        figsize=(6 * len(sim_modes), 4 * len(surfaces)),
        squeeze=False,
    )

    for i, surf in enumerate(surfaces):
        for j, mode in enumerate(sim_modes):
            ax = axes[i, j]
            sub = df[(df["surface"] == surf) & (df["sim_mode"] == mode)]
            sub = sub.dropna(subset=["MTE"])

            for penalty in sub["penalty"].unique():
                psub = sub[sub["penalty"] == penalty]
                ax.plot(psub["time"], psub["MTE"], "o-", label=penalty, markersize=4)

            ax.set_xlabel("Time")
            ax.set_ylabel("MTE")
            ax.set_title(f"{surf} — {mode}")
            ax.legend(fontsize=7)
            ax.set_yscale("log")
            ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(f"{output_prefix}_mte.png", dpi=150)
    plt.close(fig)
    print(f"  Saved {output_prefix}_mte.png")


def plot_w2_timeseries(df: pd.DataFrame, output_prefix: str):
    """Plot W2 vs time for each surface, colored by penalty, panels by sim_mode."""
    surfaces = df["surface"].unique()
    sim_modes = df["sim_mode"].unique()

    fig, axes = plt.subplots(
        len(surfaces), len(sim_modes),
        figsize=(6 * len(sim_modes), 4 * len(surfaces)),
        squeeze=False,
    )

    for i, surf in enumerate(surfaces):
        for j, mode in enumerate(sim_modes):
            ax = axes[i, j]
            sub = df[(df["surface"] == surf) & (df["sim_mode"] == mode)]
            sub = sub.dropna(subset=["W2"])

            for penalty in sub["penalty"].unique():
                psub = sub[sub["penalty"] == penalty]
                ax.plot(psub["time"], psub["W2"], "o-", label=penalty, markersize=4)

            ax.set_xlabel("Time")
            ax.set_ylabel("W2")
            ax.set_title(f"{surf} — {mode}")
            ax.legend(fontsize=7)
            ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(f"{output_prefix}_w2.png", dpi=150)
    plt.close(fig)
    print(f"  Saved {output_prefix}_w2.png")


def plot_metric_vs_distance(df: pd.DataFrame, metric: str, t_val: float, output_prefix: str):
    """Plot metric at a fixed time vs extrapolation distance."""
    surfaces = df["surface"].unique()
    sim_modes = df["sim_mode"].unique()

    fig, axes = plt.subplots(
        len(surfaces), len(sim_modes),
        figsize=(6 * len(sim_modes), 4 * len(surfaces)),
        squeeze=False,
    )

    for i, surf in enumerate(surfaces):
        for j, mode in enumerate(sim_modes):
            ax = axes[i, j]
            sub = df[(df["surface"] == surf) & (df["sim_mode"] == mode)]
            sub = sub[abs(sub["time"] - t_val) < 0.001].dropna(subset=[metric])

            for penalty in sub["penalty"].unique():
                psub = sub[sub["penalty"] == penalty].sort_values("distance")
                ax.plot(psub["distance"], psub[metric], "o-", label=penalty, markersize=4)

            ax.set_xlabel("Extrapolation distance")
            ax.set_ylabel(f"{metric} @ T={t_val}")
            ax.set_title(f"{surf} — {mode}")
            ax.legend(fontsize=7)
            ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fname = f"{output_prefix}_{metric.lower()}_vs_dist.png"
    fig.savefig(fname, dpi=150)
    plt.close(fig)
    print(f"  Saved {fname}")


def plot_example_trajectories(
    gt_traj: torch.Tensor,
    learned_traj: torch.Tensor,
    gt_alive: torch.Tensor,
    learned_alive: torch.Tensor,
    surface_name: str,
    penalty_name: str,
    sim_mode: str,
    output_prefix: str,
    n_examples: int = 3,
):
    """Plot a few example trajectories in 3D (ground truth vs learned)."""
    both_alive_final = gt_alive[:, -1] & learned_alive[:, -1]
    alive_indices = torch.where(both_alive_final)[0]

    if len(alive_indices) == 0:
        print(f"  No surviving trajectories for {surface_name}/{penalty_name}/{sim_mode}")
        return

    n_plot = min(n_examples, len(alive_indices))
    indices = alive_indices[:n_plot]

    fig = plt.figure(figsize=(5 * n_plot, 5))
    for k, idx in enumerate(indices):
        ax = fig.add_subplot(1, n_plot, k + 1, projection="3d")
        gt = gt_traj[idx].cpu().numpy()
        lr = learned_traj[idx].cpu().numpy()

        # Subsample for visibility (every 10th step)
        step = max(1, len(gt) // 100)
        ax.plot(gt[::step, 0], gt[::step, 1], gt[::step, 2],
                "b-", alpha=0.7, linewidth=1, label="Ground truth")
        ax.plot(lr[::step, 0], lr[::step, 1], lr[::step, 2],
                "r-", alpha=0.7, linewidth=1, label="Learned")
        ax.scatter(*gt[0], c="b", s=30, marker="o", zorder=5)
        ax.scatter(*lr[0], c="r", s=30, marker="o", zorder=5)

        ax.set_title(f"Traj {idx.item()}", fontsize=9)
        if k == 0:
            ax.legend(fontsize=7)

    fig.suptitle(f"{surface_name} | {penalty_name} | {sim_mode}", fontsize=11)
    fig.tight_layout()
    fname = f"{output_prefix}_traj_{surface_name}_{penalty_name}_{sim_mode}.png"
    fig.savefig(fname, dpi=150)
    plt.close(fig)
    print(f"  Saved {fname}")


# ============================================================================
# Main study orchestration
# ============================================================================

def run_trajectory_fidelity_study(
    surface_name: str,
    train_bound: float = 1.0,
    boundary: float = 3.0,
    n_train: int = 2000,
    n_traj: int = 200,
    T_max: float = 5.0,
    dt: float = 0.01,
    epochs: int = 500,
    device: str = "cuda",
    seed: int = 42,
    output_prefix: str = "traj_fidelity",
    init_region: str = "train",
    extrap_delta: float = 0.5,
) -> pd.DataFrame:
    """Run trajectory fidelity study for one surface.

    Trains all penalty configs, simulates ground-truth + learned trajectories,
    and computes path-wise and distributional metrics.

    Args:
        init_region: Where to start trajectories.
            "train" — uniform in [-train_bound, train_bound]^2 (default).
            "extrapolation" — ring region outside training domain.
        extrap_delta: Width of extrapolation ring (used when init_region="extrapolation").
    """
    region_label = f"[{train_bound}, {train_bound + extrap_delta}] ring" if init_region == "extrapolation" else f"[-{train_bound}, {train_bound}]"
    print(f"\n{'='*60}")
    print(f"Trajectory Fidelity Study: {surface_name}")
    print(f"Train: [-{train_bound}, {train_bound}], Boundary: [-{boundary}, {boundary}]")
    print(f"Init region: {init_region} ({region_label})")
    print(f"Trajectories: {n_traj}, T={T_max}, dt={dt}, Steps={int(T_max/dt)}")
    print(f"{'='*60}")

    n_steps = int(T_max / dt)

    # Create manifold SDE and lambdify
    print("\nCreating manifold SDE...")
    manifold_sde = create_manifold_sde(surface_name)
    sde = lambdify_sde(manifold_sde)

    # Sample training data
    print("Sampling training data...")
    train_data = sample_from_manifold(
        manifold_sde,
        [(-train_bound, train_bound), (-train_bound, train_bound)],
        n_samples=n_train,
        seed=seed,
        device=device,
    )

    # Sample initial conditions for trajectories
    if init_region == "extrapolation":
        initial_local = sample_ring_initial(
            n_traj, inner=train_bound, outer=train_bound + extrap_delta,
            device=device, seed=seed + 999,
        )
        n_traj = len(initial_local)  # may be slightly fewer if ring is thin
        print(f"  Extrapolation init: {n_traj} points in [{train_bound}, {train_bound + extrap_delta}] ring")
    else:
        torch.manual_seed(seed + 999)
        initial_local = (torch.rand(n_traj, 2, device=device) * 2 - 1) * train_bound
    initial_ambient = sde.chart(initial_local).to(device)

    # Pre-generate shared Brownian increments
    torch.manual_seed(seed + 1234)
    dW = torch.randn(n_traj, n_steps, 2, device=device)

    # Simulate ground truth (shared across all penalties)
    print("\nSimulating ground-truth trajectories...")
    gt_traj, gt_alive = simulate_ground_truth(
        initial_local, sde, n_steps, dt, dW, boundary,
    )
    gt_survival = gt_alive[:, -1].float().mean().item()
    print(f"  Ground truth survival at T={T_max}: {gt_survival:.1%}")

    all_results = []

    for penalty_name, loss_weights in PENALTY_CONFIGS.items():
        print(f"\n--- Training: {penalty_name} ---")

        trainer = MultiModelTrainer(TrainingConfig(
            epochs=epochs,
            n_samples=n_train,
            input_dim=3,
            hidden_dim=64,
            latent_dim=2,
            learning_rate=0.005,
            batch_size=32,
            test_size=0.03,
            print_interval=max(1, epochs // 5),
            device=device,
        ))
        trainer.add_model(make_model_config(penalty_name, loss_weights))

        data_loader = trainer.create_data_loader(train_data)
        for epoch in range(epochs):
            losses = trainer.train_epoch(data_loader)
            if (epoch + 1) % max(1, epochs // 5) == 0:
                print(f"  Epoch {epoch+1}: {losses[penalty_name]:.6f}")

        model = trainer.models[penalty_name]
        model.eval()

        for sim_mode in ["learned_latent", "end_to_end"]:
            print(f"  Simulating {sim_mode}...")

            if sim_mode == "learned_latent":
                learned_traj, learned_alive = simulate_learned_latent(
                    model, initial_local, sde, n_steps, dt, dW, boundary,
                )
            else:  # end_to_end
                learned_traj, learned_alive = simulate_end_to_end(
                    model, initial_ambient, sde, n_steps, dt, dW, boundary,
                )

            survival = learned_alive[:, -1].float().mean().item()
            print(f"    Survival at T={T_max}: {survival:.1%}")

            # Compute metrics at snapshot times
            metrics_rows = compute_metrics_at_snapshots(
                learned_traj, gt_traj, learned_alive, gt_alive,
                dt, PATH_TIMES, DIST_TIMES,
            )

            for row in metrics_rows:
                row["surface"] = surface_name
                row["penalty"] = penalty_name
                row["sim_mode"] = sim_mode
                all_results.append(row)

            # Print summary
            mte_1 = [r for r in metrics_rows if abs(r["time"] - 1.0) < 0.001]
            if mte_1 and "MTE" in mte_1[0]:
                print(f"    MTE@1.0 = {mte_1[0]['MTE']:.6f}")
            w2_5 = [r for r in metrics_rows if abs(r["time"] - 5.0) < 0.001]
            if w2_5 and "W2" in w2_5[0]:
                print(f"    W2@5.0  = {w2_5[0]['W2']:.6f}")

            # Example trajectory plots (only for T+K penalty to avoid clutter)
            if penalty_name in ("baseline", "T+F+K"):
                plot_example_trajectories(
                    gt_traj, learned_traj, gt_alive, learned_alive,
                    surface_name, penalty_name, sim_mode, output_prefix,
                )

    return pd.DataFrame(all_results)


def run_trajectory_fidelity_sweep(
    surface_name: str,
    train_bound: float = 1.0,
    boundary: float = 3.0,
    n_train: int = 2000,
    n_traj: int = 200,
    T_max: float = 5.0,
    dt: float = 0.01,
    epochs: int = 500,
    max_extrap_dist: float = 1.0,
    dist_step: float = 0.2,
    device: str = "cuda",
    seed: int = 42,
    output_prefix: str = "traj_fidelity_sweep",
) -> pd.DataFrame:
    """Run trajectory fidelity study sweeping multiple extrapolation distances.

    Trains all penalty configs once, then evaluates trajectory fidelity at
    distances [0.0, dist_step, 2*dist_step, ..., max_extrap_dist].
    distance=0.0 corresponds to the training region (interpolation).

    Args:
        max_extrap_dist: Maximum extrapolation distance beyond train_bound.
        dist_step: Step size between distance rings.
    """
    distances = [0.0] + [
        round(d, 4) for d in
        np.arange(dist_step, max_extrap_dist + dist_step / 2, dist_step).tolist()
    ]

    print(f"\n{'='*60}")
    print(f"Trajectory Fidelity Sweep: {surface_name}")
    print(f"Train: [-{train_bound}, {train_bound}], Boundary: [-{boundary}, {boundary}]")
    print(f"Distances: {distances}")
    print(f"Trajectories/distance: {n_traj}, T={T_max}, dt={dt}")
    print(f"{'='*60}")

    n_steps = int(T_max / dt)

    # Create manifold SDE and lambdify
    print("\nCreating manifold SDE...")
    manifold_sde = create_manifold_sde(surface_name)
    sde = lambdify_sde(manifold_sde)

    # Sample training data
    print("Sampling training data...")
    train_data = sample_from_manifold(
        manifold_sde,
        [(-train_bound, train_bound), (-train_bound, train_bound)],
        n_samples=n_train,
        seed=seed,
        device=device,
    )

    # Train all models once
    trained_models = {}
    for penalty_name, loss_weights in PENALTY_CONFIGS.items():
        print(f"\n--- Training: {penalty_name} ---")

        trainer = MultiModelTrainer(TrainingConfig(
            epochs=epochs,
            n_samples=n_train,
            input_dim=3,
            hidden_dim=64,
            latent_dim=2,
            learning_rate=0.005,
            batch_size=32,
            test_size=0.03,
            print_interval=max(1, epochs // 5),
            device=device,
        ))
        trainer.add_model(make_model_config(penalty_name, loss_weights))

        data_loader = trainer.create_data_loader(train_data)
        for epoch in range(epochs):
            losses = trainer.train_epoch(data_loader)
            if (epoch + 1) % max(1, epochs // 5) == 0:
                print(f"  Epoch {epoch+1}: {losses[penalty_name]:.6f}")

        model = trainer.models[penalty_name]
        model.eval()
        trained_models[penalty_name] = model

    # Sweep over distances
    all_results = []
    for dist in distances:
        print(f"\n{'='*40}")
        if dist == 0.0:
            print(f"Distance 0.0 (training region)")
        else:
            inner = train_bound + dist - dist_step
            outer = train_bound + dist
            print(f"Distance {dist:.2f} (ring [{inner:.2f}, {outer:.2f}])")
        print(f"{'='*40}")

        # Generate initial conditions for this distance
        if dist == 0.0:
            torch.manual_seed(seed + 999)
            initial_local = (torch.rand(n_traj, 2, device=device) * 2 - 1) * train_bound
        else:
            inner = train_bound + dist - dist_step
            outer = train_bound + dist
            initial_local = sample_ring_initial(
                n_traj, inner=inner, outer=outer,
                device=device, seed=seed + 999 + int(dist * 1000),
            )
        actual_n = len(initial_local)
        initial_ambient = sde.chart(initial_local).to(device)
        print(f"  {actual_n} initial points sampled")

        # Shared Brownian increments for this distance
        torch.manual_seed(seed + 1234 + int(dist * 1000))
        dW = torch.randn(actual_n, n_steps, 2, device=device)

        # Ground truth
        gt_traj, gt_alive = simulate_ground_truth(
            initial_local, sde, n_steps, dt, dW, boundary,
        )
        gt_survival = gt_alive[:, -1].float().mean().item()
        print(f"  GT survival at T={T_max}: {gt_survival:.1%}")

        for penalty_name, model in trained_models.items():
            for sim_mode in ["learned_latent", "end_to_end"]:
                if sim_mode == "learned_latent":
                    learned_traj, learned_alive = simulate_learned_latent(
                        model, initial_local, sde, n_steps, dt, dW, boundary,
                    )
                else:  # end_to_end
                    learned_traj, learned_alive = simulate_end_to_end(
                        model, initial_ambient, sde, n_steps, dt, dW, boundary,
                    )

                metrics_rows = compute_metrics_at_snapshots(
                    learned_traj, gt_traj, learned_alive, gt_alive,
                    dt, PATH_TIMES, DIST_TIMES,
                )

                for row in metrics_rows:
                    row["surface"] = surface_name
                    row["penalty"] = penalty_name
                    row["sim_mode"] = sim_mode
                    row["distance"] = dist
                    all_results.append(row)

                # Brief summary
                mte_1 = [r for r in metrics_rows if abs(r["time"] - 1.0) < 0.001]
                if mte_1 and "MTE" in mte_1[0] and not math.isnan(mte_1[0].get("MTE", float("nan"))):
                    print(f"  {penalty_name:10s} {sim_mode:20s} MTE@1.0={mte_1[0]['MTE']:.4f}")

    return pd.DataFrame(all_results)


# ============================================================================
# CLI
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="AEML Trajectory Fidelity Study")
    parser.add_argument("--surface", type=str, default="paraboloid",
                        choices=STUDY_SURFACES + ["all"])
    parser.add_argument("--train_bound", type=float, default=1.0)
    parser.add_argument("--boundary", type=float, default=3.0)
    parser.add_argument("--n_train", type=int, default=2000)
    parser.add_argument("--n_traj", type=int, default=200)
    parser.add_argument("--T_max", type=float, default=5.0)
    parser.add_argument("--dt", type=float, default=0.01)
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--init_region", type=str, default="train",
                        choices=["train", "extrapolation", "sweep"],
                        help="Where to start trajectories: 'train', 'extrapolation', or 'sweep' (multi-distance)")
    parser.add_argument("--extrap_delta", type=float, default=0.5,
                        help="Width of extrapolation ring (default 0.5)")
    parser.add_argument("--max_extrap_dist", type=float, default=1.0,
                        help="Max extrapolation distance for sweep mode (default 1.0)")
    parser.add_argument("--dist_step", type=float, default=0.2,
                        help="Distance step for sweep mode (default 0.2)")
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--output_prefix", type=str, default=None)

    args = parser.parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    surfaces = STUDY_SURFACES if args.surface == "all" else [args.surface]

    if args.init_region == "sweep":
        # Sweep mode: train once, evaluate at multiple distances
        region_tag = "sweep"
        output_csv = args.output or f"trajectory_fidelity_{region_tag}.csv"
        output_prefix = args.output_prefix or f"traj_fidelity_{region_tag}"

        all_dfs = []
        for surf in surfaces:
            df = run_trajectory_fidelity_sweep(
                surface_name=surf,
                train_bound=args.train_bound,
                boundary=args.boundary,
                n_train=args.n_train,
                n_traj=args.n_traj,
                T_max=args.T_max,
                dt=args.dt,
                epochs=args.epochs,
                max_extrap_dist=args.max_extrap_dist,
                dist_step=args.dist_step,
                device=device,
                seed=args.seed,
                output_prefix=output_prefix,
            )
            all_dfs.append(df)

        combined = pd.concat(all_dfs, ignore_index=True)
        combined.to_csv(output_csv, index=False)
        print(f"\nResults saved to {output_csv}")

        # Sweep-specific plots: metric vs distance
        print("\nGenerating sweep plots...")
        plot_metric_vs_distance(combined, "MTE", 1.0, output_prefix)
        plot_metric_vs_distance(combined, "W2", 5.0, output_prefix)

        # Summary table: MTE@1.0 by distance
        print("\n" + "=" * 70)
        print("TRAJECTORY FIDELITY SWEEP SUMMARY")
        print("=" * 70)

        for metric, t_val in [("MTE", 1.0), ("W2", 5.0)]:
            sub = combined[abs(combined["time"] - t_val) < 0.001].dropna(subset=[metric])
            if sub.empty:
                continue
            print(f"\n{metric} at T={t_val}:")
            pivot = sub.pivot_table(
                index=["surface", "sim_mode", "distance"],
                columns="penalty",
                values=metric,
            )
            print(pivot.round(4).to_string())

    else:
        # Single-region mode (train or extrapolation)
        region_tag = "extrap" if args.init_region == "extrapolation" else "train"
        output_csv = args.output or f"trajectory_fidelity_{region_tag}.csv"
        output_prefix = args.output_prefix or f"traj_fidelity_{region_tag}"

        all_dfs = []
        for surf in surfaces:
            df = run_trajectory_fidelity_study(
                surface_name=surf,
                train_bound=args.train_bound,
                boundary=args.boundary,
                n_train=args.n_train,
                n_traj=args.n_traj,
                T_max=args.T_max,
                dt=args.dt,
                epochs=args.epochs,
                device=device,
                seed=args.seed,
                output_prefix=output_prefix,
                init_region=args.init_region,
                extrap_delta=args.extrap_delta,
            )
            all_dfs.append(df)

        combined = pd.concat(all_dfs, ignore_index=True)
        combined.to_csv(output_csv, index=False)
        print(f"\nResults saved to {output_csv}")

        # Generate summary plots
        print("\nGenerating summary plots...")
        plot_mte_timeseries(combined, output_prefix)
        plot_w2_timeseries(combined, output_prefix)

        # Print summary table
        print("\n" + "=" * 70)
        print("TRAJECTORY FIDELITY SUMMARY")
        print("=" * 70)

        for metric in ["MTE", "W2"]:
            t_val = 1.0 if metric == "MTE" else 5.0
            sub = combined[abs(combined["time"] - t_val) < 0.001].dropna(subset=[metric])
            if sub.empty:
                continue
            print(f"\n{metric} at T={t_val}:")
            pivot = sub.pivot_table(
                index=["surface", "sim_mode"],
                columns="penalty",
                values=metric,
            )
            print(pivot.round(6).to_string())

        # End-to-end vs latent gap
        print("\n\nEND-TO-END vs LATENT GAP (MTE@1.0):")
        mte_sub = combined[abs(combined["time"] - 1.0) < 0.001].dropna(subset=["MTE"])
        if not mte_sub.empty:
            for surf in mte_sub["surface"].unique():
                print(f"\n  {surf}:")
                for pen in mte_sub["penalty"].unique():
                    lat = mte_sub[(mte_sub["surface"] == surf) &
                                  (mte_sub["penalty"] == pen) &
                                  (mte_sub["sim_mode"] == "learned_latent")]
                    e2e = mte_sub[(mte_sub["surface"] == surf) &
                                  (mte_sub["penalty"] == pen) &
                                  (mte_sub["sim_mode"] == "end_to_end")]
                    if not lat.empty and not e2e.empty:
                        gap = e2e["MTE"].values[0] - lat["MTE"].values[0]
                        print(f"    {pen:10s}: latent={lat['MTE'].values[0]:.6f}  "
                              f"e2e={e2e['MTE'].values[0]:.6f}  gap={gap:+.6f}")


if __name__ == "__main__":
    main()
