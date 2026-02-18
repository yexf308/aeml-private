# SDE Trajectory Fidelity Study

## Goal

Validate that AEML's learned geometry faithfully reproduces manifold dynamics through actual trajectory simulation, not just pointwise drift/covariance comparison.

Core question: When you simulate an SDE using the learned geometry, how closely do the resulting trajectories match ground truth — and does this degrade over time?

## Simulation Modes

Two simulation modes, compared side-by-side:

1. **Ambient-space simulation**: Reconstruct drift/covariance from the decoder Jacobian in R^3, integrate there. Tests geometry accuracy.
2. **Latent-space simulation**: Integrate the SDE in the 2D latent coordinates, decode each step back to R^3. Tests the full encode-decode pipeline.

The gap between these two modes isolates how much the encoder/decoder mapping itself distorts dynamics, separate from geometry errors.

Both are compared against **ground-truth trajectories** from the symbolic `ManifoldSDE`.

## Surfaces

Three surfaces with distinct curvature profiles:

- **Paraboloid** — positive, mild, uniform curvature
- **Hyperbolic paraboloid** — negative (saddle) curvature
- **Sinusoidal** — spatially varying curvature

## Integrator

Euler-Maruyama with shared noise for fair path-wise comparison.

- Step size: `dt = 0.01`
- Time horizon: T = 5.0 (path-wise metrics up to T=1.0, distributional up to T=5.0)
- Ensemble size: 200 trajectories per configuration
- Initial conditions: sampled uniformly from the training region [-1, 1]^2

### Boundary handling

Hard boundary at [-1.5, 1.5]^2 in local coordinates (0.5 beyond training bound of 1.0). Trajectories that exit are frozen and excluded from metrics after that point. Surviving ensemble fraction is tracked at each time step.

### Noise sharing

For path-wise comparison, all three trajectory types (ground-truth, learned-ambient, learned-latent) use the same Brownian increments. For distributional comparison, independent noise is used.

## Metrics

### Path-wise (short horizon, T=0.1 to 1.0)

- **Mean trajectory error (MTE)**: At each time step `t`, average Euclidean distance between learned and true trajectories: `MTE(t) = (1/N) sum ||x_learned(t) - x_true(t)||`.
- **Relative path divergence (RPD)**: MTE normalized by mean displacement from initial condition: `RPD(t) = MTE(t) / mean(||x_true(t) - x_true(0)||)`.

### Distributional (medium horizon, T=1.0 to 5.0)

- **W2 (Wasserstein-2)**: Computed at snapshot times t = {1.0, 2.0, 3.0, 4.0, 5.0} using the POT library.
- **MMD (Maximum Mean Discrepancy)**: Gaussian kernel with median bandwidth heuristic. Computed at finer granularity (every 0.5 time units).

### Summary statistics per (surface, penalty, simulation mode)

- MTE at T=0.5 and T=1.0
- W2 and MMD at T=2.0 and T=5.0
- Ambient-vs-latent gap (difference in MTE/W2 between simulation modes)
- Surviving ensemble fraction

## Pipeline

1. **Train models** — One per penalty config, 3 surfaces x 6 penalties = 18 models. Reuses `MultiModelTrainer` and `PENALTY_CONFIGS` from `experiments/common.py`.

2. **Lambdify ground-truth SDE** — Convert symbolic `ManifoldSDE.ambient_drift` and `ambient_diffusion` to callable torch functions via `sp.lambdify`.

3. **Simulate trajectories** — For each (surface, penalty), run ground-truth, learned-ambient, and learned-latent trajectories with Euler-Maruyama.

4. **Compute metrics** — MTE/RPD at short horizons, W2/MMD at medium horizons, ensemble survival throughout.

5. **Output**:
   - CSV: columns `surface, penalty, sim_mode, time, MTE, RPD, W2, MMD, ensemble_survival`
   - Time-series plots of MTE and W2 vs. time, per surface, colored by penalty config
   - Example trajectory visualizations (true vs. learned paths in 3D), 2-3 per surface

## Module

`experiments/trajectory_fidelity_study.py`

## Dependencies

- `pot` (Python Optimal Transport) for W2 computation
- All other dependencies already in the project

## Penalty Configs

From `experiments/common.py`:

- baseline (no penalties)
- T (tangent bundle)
- K (curvature)
- T+K (tangent + curvature)
- T+F (tangent + diffeomorphism)
- T+F+K (tangent + diffeomorphism + curvature)
