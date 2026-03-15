"""
Shared Müller-Brown dynamics module.

Provides MB potential, drift function, well centers, inter-well MFPT,
transition rate, and a D-general two-phase AE trainer.

Overdamped Langevin dynamics: drift = -∇V_MB, diffusion = √(2kT)·I.
FDT holds. Stationary distribution is Boltzmann ∝ exp(-V/kT).
Kramers escape rates are physically interpretable.
"""

import torch
import math
import numpy as np
from dataclasses import asdict

from src.numeric.losses import LossWeights
from src.numeric.training import MultiModelTrainer, TrainingConfig, TrainingPhase
from experiments.common import make_model_config
from experiments.highd_N_D_sweep import hidden_dims_for_D
from experiments.data_driven_sde import BATCH_SIZE, LR_AE, DEVICE


# ── MB potential parameters ──────────────────────────────────────────────

_A = torch.tensor([-200., -100., -170., 15.])
_a = torch.tensor([-1., -1., -6.5, 0.7])
_b = torch.tensor([0., 0., 11., 0.6])
_c = torch.tensor([-10., -10., -6.5, 0.7])
_x0 = torch.tensor([1., 0., -0.5, -1.])
_y0 = torch.tensor([0., 0.5, 1.5, 1.])

KT = 0.15
V_SCALE = 200.0

# Three MB wells in (u,v) ∈ [-1,1]² coordinates
WELLS_UV = torch.tensor([
    [-0.30,  0.55],   # well 1 (deepest, V≈-0.73)
    [ 0.57, -0.58],   # well 2 (V≈-0.54)
    [ 0.07, -0.23],   # well 3 (shallowest, V≈-0.40)
])

DEBOUNCE_STEPS = 5
WELL_NAMES = ["W1", "W2", "W3"]

TRAIN_BOUND = 0.8  # keeps data away from MB potential exponential walls


# ── MB potential and drift ───────────────────────────────────────────────

def mb_potential(uv: torch.Tensor) -> torch.Tensor:
    """Rescaled MB potential V(u,v)/V_SCALE in [-1,1]² coordinates.

    Args:
        uv: (B, 2) or (..., 2) local coordinates

    Returns:
        V: same leading shape, scalar potential values
    """
    u, v = uv[..., 0], uv[..., 1]
    x = (2.7 * u - 0.3) / 2
    y = (2.5 * v + 1.5) / 2
    dev = uv.device
    A, a, b, c = _A.to(dev), _a.to(dev), _b.to(dev), _c.to(dev)
    x0, y0 = _x0.to(dev), _y0.to(dev)
    V = torch.zeros_like(x)
    for i in range(4):
        V = V + A[i] * torch.exp(
            a[i] * (x - x0[i])**2
            + b[i] * (x - x0[i]) * (y - y0[i])
            + c[i] * (y - y0[i])**2
        )
    return V / V_SCALE


def mb_local_drift_fn(uv: torch.Tensor) -> torch.Tensor:
    """Negative gradient of MB potential: -∇V(u,v). Input (B,2) -> (B,2).

    Same interface as the rotation drift function in highd_N_D_sweep.
    """
    uv_g = uv.detach().requires_grad_(True)
    V = mb_potential(uv_g).sum()
    grad = torch.autograd.grad(V, uv_g)[0]
    return -grad.detach()


def mb_local_diffusion_fn(uv: torch.Tensor) -> torch.Tensor:
    """Isotropic diffusion for overdamped Langevin: σ = √(2kT)·I₂.

    Input (B,2) -> (B,2,2).  FDT-consistent with drift = -∇V.
    """
    B = uv.shape[0]
    sigma = math.sqrt(2 * KT)
    return sigma * torch.eye(2, device=uv.device).unsqueeze(0).expand(B, -1, -1)


# ── Inter-well metrics ───────────────────────────────────────────────────

def assign_wells_ambient(traj: torch.Tensor, well_centers_ambient: torch.Tensor) -> torch.Tensor:
    """Voronoi assignment of trajectory points to nearest well center.

    Args:
        traj: (B, T+1, D) ambient trajectory
        well_centers_ambient: (3, D) ambient coordinates of well centers

    Returns:
        assignment: (B, T+1) integer well index (0, 1, or 2)
    """
    dists = torch.cdist(
        traj,
        well_centers_ambient.unsqueeze(0).expand(traj.shape[0], -1, -1),
    )
    return dists.argmin(dim=-1)


def compute_interwell_mfpt(assignments: torch.Tensor, start_well: int, dt: float) -> dict:
    """First passage time from start_well to any other well (with debounce).

    Args:
        assignments: (B, T+1) well assignments
        start_well: starting well index
        dt: time step

    Returns:
        dict with 'mean', 'std', 'transition_frac', 'frac_to_W*'
    """
    B, T1 = assignments.shape
    fpt = torch.full((B,), float('nan'))
    targets = []

    for b in range(B):
        consecutive = 0
        candidate_well = -1
        for t in range(1, T1):
            w = assignments[b, t].item()
            if w != start_well:
                if w == candidate_well:
                    consecutive += 1
                else:
                    candidate_well = w
                    consecutive = 1
                if consecutive >= DEBOUNCE_STEPS:
                    fpt[b] = (t - DEBOUNCE_STEPS + 1) * dt
                    targets.append(candidate_well)
                    break
            else:
                candidate_well = -1
                consecutive = 0

    transitioned = torch.isfinite(fpt)
    n_trans = transitioned.sum().item()

    result = {
        "mean": fpt[transitioned].mean().item() if n_trans >= 2 else float('nan'),
        "std": fpt[transitioned].std().item() if n_trans >= 2 else float('nan'),
        "transition_frac": n_trans / B,
    }
    for i, name in enumerate(WELL_NAMES):
        if i != start_well:
            result[f"frac_to_{name}"] = targets.count(i) / max(n_trans, 1)

    return result


def compute_transition_rate(assignments: torch.Tensor, dt: float) -> dict:
    """Count transitions between wells per unit time (with debounce).

    Args:
        assignments: (B, T+1) well assignments
        dt: time step

    Returns:
        dict with 'rate', 'total_transitions', 'total_time', 'transitions_per_traj'
    """
    B, T1 = assignments.shape
    total_time = B * (T1 - 1) * dt
    total_transitions = 0

    for b in range(B):
        current_well = assignments[b, 0].item()
        consecutive = 0
        candidate_well = -1
        for t in range(1, T1):
            w = assignments[b, t].item()
            if w != current_well:
                if w == candidate_well:
                    consecutive += 1
                else:
                    candidate_well = w
                    consecutive = 1
                if consecutive >= DEBOUNCE_STEPS:
                    total_transitions += 1
                    current_well = candidate_well
                    consecutive = 0
                    candidate_well = -1
            else:
                candidate_well = -1
                consecutive = 0

    return {
        "total_transitions": total_transitions,
        "total_time": total_time,
        "rate": total_transitions / total_time if total_time > 0 else 0,
        "transitions_per_traj": total_transitions / B,
    }


# ── D-general two-phase AE trainer ──────────────────────────────────────

def train_ae_highd(surface, D, seed, n_train, epochs, loss_weights, train_data):
    """Train AE with two-phase schedule for K-containing configs.

    Works for any D (not hardcoded to D=3 like train_ae_with_twophase).

    Phase 1: same weights minus curvature (warmup)
    Phase 2: full weights (fine-tune with K)

    For non-K configs, trains flat for the full epoch budget.

    Args:
        surface: FourierAugmentedSurface instance
        D: ambient dimension
        seed: random seed
        n_train: number of training points
        epochs: total training epochs
        loss_weights: LossWeights for this condition
        train_data: DatasetBatch from sample_from_highd_manifold

    Returns:
        (ae, recon_per_dim) — same interface as _train_ae
    """
    hdims = hidden_dims_for_D(D)
    batch_size = min(n_train, BATCH_SIZE)

    trainer = MultiModelTrainer(TrainingConfig(
        epochs=epochs, n_samples=n_train, input_dim=D, hidden_dim=hdims[0],
        latent_dim=2, learning_rate=LR_AE, batch_size=batch_size,
        test_size=0.03, print_interval=max(1, epochs // 5), device=DEVICE,
    ))
    mc = make_model_config("ae", loss_weights, extrinsic_dim=D, hidden_dims=hdims)
    trainer.add_model(mc)
    loader = trainer.create_data_loader(train_data)

    has_K = loss_weights.curvature > 0
    if has_K:
        # Build warmup weights (same but curvature=0)
        warmup_kwargs = {k: v for k, v in asdict(loss_weights).items()
                         if k != "curvature" and v > 0}
        warmup_lw = LossWeights(**warmup_kwargs)

        phase1_epochs = epochs // 2
        phase2_epochs = epochs - phase1_epochs
        schedule = [
            TrainingPhase(epochs=phase1_epochs, loss_weights=warmup_lw, name="warmup"),
            TrainingPhase(epochs=phase2_epochs, loss_weights=loss_weights, name="finetune"),
        ]
        trainer.train_with_schedule(loader, mc.name, schedule,
                                    print_interval=max(1, epochs // 5))
    else:
        for epoch in range(epochs):
            losses = trainer.train_epoch(loader, {mc.name: loss_weights})
            if (epoch + 1) % max(1, epochs // 5) == 0:
                print(f"      Epoch {epoch+1}/{epochs}: loss={losses['ae']:.6f}")

    ae = trainer.models["ae"]
    ae.eval()

    # Recon quality check
    x = train_data.samples.to(DEVICE)
    with torch.no_grad():
        z = ae.encoder(x)
        x_hat = ae.decoder(z)
        recon_per_dim = ((x_hat - x) ** 2).sum(-1).mean().item() / D

    return ae, recon_per_dim
