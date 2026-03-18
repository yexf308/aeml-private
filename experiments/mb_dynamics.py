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

KT = 0.10
V_SCALE = 200.0

# Three MB wells in (u,v) ∈ [-1,1]² coordinates (rescaling scale=2.25)
# Original well locations: M1(-0.558,1.442), M2(0.623,0.028), M3(-0.050,0.467)
# Inverse: u = (x + 0.25)/2.25, v = (y - 1.0)/2.25
WELLS_UV = torch.tensor([
    [-0.1369,  0.1964],   # well 1 (deepest)
    [ 0.3880, -0.4320],   # well 2
    [ 0.0889, -0.2369],   # well 3 (shallowest)
])

DEBOUNCE_STEPS = 5
WELL_NAMES = ["W1", "W2", "W3"]

TRAIN_BOUND = 0.55  # dynamically relevant region (all wells inside, max drift ~140)

CORE_RADIUS = 0.08  # smaller than well-to-boundary gap (0.118) to avoid reflection artifacts
CORE_DWELL = 10     # stricter dwell confirmation

# Approximate saddle point between W1 and W2 (in rescaled (u,v) coords)
SADDLE_UV = torch.tensor([0.022, 0.006])


def importance_sample_mb(n_samples, seed=42, device="cpu",
                         saddle_frac=0.3, saddle_std=0.08):
    """Sample (u,v) with extra density near the MB saddle region.

    Args:
        n_samples: total number of points
        seed: random seed
        device: torch device
        saddle_frac: fraction of points to place near the saddle
        saddle_std: std of Gaussian around the saddle

    Returns:
        uv: (n_samples, 2) tensor of local coordinates
    """
    torch.manual_seed(seed)
    n_saddle = int(n_samples * saddle_frac)
    n_uniform = n_samples - n_saddle

    # Uniform points
    uv_uniform = (torch.rand(n_uniform, 2, device=device) * 2 - 1) * TRAIN_BOUND

    # Saddle-focused points (Gaussian, clamped to training domain)
    uv_saddle = SADDLE_UV.to(device) + torch.randn(n_saddle, 2, device=device) * saddle_std
    uv_saddle = uv_saddle.clamp(-TRAIN_BOUND, TRAIN_BOUND)

    return torch.cat([uv_uniform, uv_saddle], dim=0)


# ── MB potential and drift ───────────────────────────────────────────────

def mb_potential(uv: torch.Tensor) -> torch.Tensor:
    """Rescaled MB potential V(u,v)/V_SCALE in [-1,1]² coordinates.

    Args:
        uv: (B, 2) or (..., 2) local coordinates

    Returns:
        V: same leading shape, scalar potential values
    """
    u, v = uv[..., 0], uv[..., 1]
    # Rescaling: (u,v) ∈ [-1,1]² → original MB coords
    # scale=2.25 with offset to center wells in domain
    x = 2.25 * u - 0.25
    y = 2.25 * v + 1.0
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


# ── Core-set assignment and Markov metrics ─────────────────────────────

def assign_wells_coreset_2d(traj_uv, wells_uv=None, core_radius=CORE_RADIUS):
    """Core-set assignment: assign trajectory points to nearest well if
    within core_radius, otherwise -1 (unassigned).

    Args:
        traj_uv: (B, T+1, 2) trajectory in 2D coordinates
        wells_uv: (n_wells, 2) well centers. Defaults to WELLS_UV.
        core_radius: radius threshold for assignment

    Returns:
        assignments: (B, T+1) integer tensor (0..n_wells-1 or -1)
    """
    if wells_uv is None:
        wells_uv = WELLS_UV
    wells_uv = wells_uv.to(traj_uv.device)
    # (B, T+1, n_wells) distances
    dists = torch.cdist(traj_uv, wells_uv.unsqueeze(0).expand(traj_uv.shape[0], -1, -1))
    min_dist, nearest = dists.min(dim=-1)  # (B, T+1)
    assignments = torch.where(min_dist <= core_radius, nearest, torch.tensor(-1, device=traj_uv.device))
    return assignments


def compute_transition_matrix(assignments, lag_steps, n_wells=3):
    """Compute row-stochastic transition matrix from core-set assignments.

    For each (t, t+lag), if both assignments[t] and assignments[t+lag] are >= 0,
    increment count[i, j].  Vectorized with bincount (no Python loops).

    Args:
        assignments: (B, T+1) integer assignments (-1 = unassigned)
        lag_steps: number of steps for the lag
        n_wells: number of wells

    Returns:
        P_tau: (n_wells, n_wells) row-stochastic matrix
    """
    B, T1 = assignments.shape
    # Move to CPU for counting
    a = assignments.cpu()
    src = a[:, :-lag_steps].reshape(-1)       # (B * (T1 - lag_steps),)
    dst = a[:, lag_steps:].reshape(-1)
    valid = (src >= 0) & (dst >= 0)
    src_v = src[valid].long()
    dst_v = dst[valid].long()

    flat_idx = src_v * n_wells + dst_v
    counts = torch.bincount(flat_idx, minlength=n_wells * n_wells).reshape(n_wells, n_wells).float()

    # Row-normalize
    row_sums = counts.sum(dim=1, keepdim=True)
    P_tau = torch.where(row_sums > 0, counts / row_sums.clamp(min=1), torch.zeros_like(counts))
    return P_tau


def compute_implied_timescales(assignments, dt, lag_times=None, n_wells=3):
    """Compute implied timescales from core-set transition matrices.

    Args:
        assignments: (B, T+1) integer assignments
        dt: simulation time step
        lag_times: list of lag times (seconds). Default [0.05, 0.1, 0.2, 0.5].
        n_wells: number of wells

    Returns:
        dict with 't2' and 't3' arrays (implied timescales for 2nd and 3rd
        eigenvalues), and 'plateau_lag' (best plateau lag time).
    """
    if lag_times is None:
        lag_times = [0.05, 0.1, 0.2, 0.5]

    t2_list, t3_list = [], []
    for tau in lag_times:
        lag_steps = max(1, int(round(tau / dt)))
        P_tau = compute_transition_matrix(assignments, lag_steps, n_wells)
        eigvals = torch.linalg.eigvals(P_tau).real
        eigvals_sorted = eigvals[eigvals.abs().sort(descending=True).indices]

        ts = []
        for k in range(1, min(n_wells, len(eigvals_sorted))):
            lam_k = eigvals_sorted[k].item()
            if 0 < abs(lam_k) < 1:
                t_k = -tau / math.log(abs(lam_k))
            else:
                t_k = float('inf')
            ts.append(t_k)

        t2_list.append(ts[0] if len(ts) >= 1 else float('nan'))
        t3_list.append(ts[1] if len(ts) >= 2 else float('nan'))

    # Plateau detection with nan/inf safety
    t2_arr = np.array(t2_list)
    finite_mask = np.isfinite(t2_arr)
    if finite_mask.sum() >= 2:
        # Only consider consecutive finite pairs
        rel_changes = []
        for i in range(len(t2_arr) - 1):
            if finite_mask[i] and finite_mask[i + 1] and abs(t2_arr[i]) > 1e-12:
                rel_changes.append((abs(t2_arr[i + 1] - t2_arr[i]) / abs(t2_arr[i]), i + 1))
        if rel_changes:
            best_idx = min(rel_changes, key=lambda x: x[0])[1]
        else:
            best_idx = 0
    else:
        best_idx = 0

    return {
        't2': np.array(t2_list),
        't3': np.array(t3_list),
        'lag_times': np.array(lag_times),
        'plateau_lag': lag_times[best_idx],
    }


def compute_stationary_probabilities(assignments, n_wells=3):
    """Compute stationary (empirical) probabilities from assignments.

    Counts fraction of time spent in each well, ignoring -1 assignments.

    Args:
        assignments: (B, T+1) integer assignments
        n_wells: number of wells

    Returns:
        pi: (n_wells,) tensor of probabilities
    """
    counts = torch.zeros(n_wells, dtype=torch.float64)
    for w in range(n_wells):
        counts[w] = (assignments == w).sum().item()
    total = counts.sum()
    if total > 0:
        return (counts / total).float()
    return torch.zeros(n_wells)


def compute_pairwise_mfpt(assignments, dt, n_wells=3, dwell=CORE_DWELL):
    """Compute pairwise mean first passage times with dwell confirmation.

    For each trajectory, for each (i, j) pair, find first passage from
    core i to core j: trajectory must start in well i and first reach
    well j with at least `dwell` consecutive steps.

    Args:
        assignments: (B, T+1) integer assignments (-1 = unassigned)
        dt: simulation time step
        n_wells: number of wells
        dwell: number of consecutive steps required to confirm arrival

    Returns:
        mfpt: (n_wells, n_wells) matrix of mean FPTs (nan if no transitions)
    """
    B, T1 = assignments.shape
    # Collect passage times for each (i, j) pair
    fpt_lists = [[[] for _ in range(n_wells)] for _ in range(n_wells)]

    for b in range(B):
        traj = assignments[b]  # (T1,)
        # Find all passages: scan for well visits
        t = 0
        while t < T1:
            start_well = traj[t].item()
            if start_well < 0:
                t += 1
                continue
            # We are in well start_well at time t; look for first passage to another well
            t_start = t
            consecutive = 0
            candidate = -1
            found = False
            for s in range(t + 1, T1):
                w = traj[s].item()
                if w >= 0 and w != start_well:
                    if w == candidate:
                        consecutive += 1
                    else:
                        candidate = w
                        consecutive = 1
                    if consecutive >= dwell:
                        passage_time = (s - dwell + 1 - t_start) * dt
                        fpt_lists[start_well][candidate].append(passage_time)
                        t = s  # continue scanning from arrival
                        found = True
                        break
                elif w == start_well:
                    candidate = -1
                    consecutive = 0
                else:
                    # w == -1 (outside all cores), reset dwell counter
                    candidate = -1
                    consecutive = 0
            if not found:
                t = T1  # no more passages from this starting point
            else:
                t += 1

    mfpt = torch.full((n_wells, n_wells), float('nan'))
    for i in range(n_wells):
        for j in range(n_wells):
            if i != j and len(fpt_lists[i][j]) >= 1:
                mfpt[i, j] = np.mean(fpt_lists[i][j])
    return mfpt


def compute_exit_fraction(z_traj, train_bound=TRAIN_BOUND):
    """Fraction of trajectories that ever leave [-train_bound, train_bound]^d.

    Args:
        z_traj: (B, T+1, d) latent trajectory
        train_bound: domain boundary

    Returns:
        float: fraction of trajectories with any step outside the box
    """
    B = z_traj.shape[0]
    outside = (z_traj.abs() > train_bound).any(dim=-1)  # (B, T+1)
    ever_outside = outside.any(dim=1)  # (B,)
    return ever_outside.float().mean().item()


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
