# High-Dimensional N×D Sweep Findings

**Date:** 2026-03-10 (D-normalised smoothness fix)
**Experiment:** `experiments/highd_N_D_sweep.py`
**Setup:** 10 seeds, 500 AE epochs (two-phase: 250+250), 300 SDE epochs, N ∈ {20,50,100,200}, D ∈ {11,201}, surfaces: paraboloid, hyperbolic_paraboloid

## Three Conditions

| Condition | AE loss weights | Stage 2 smoothing |
|-----------|----------------|-------------------|
| baseline  | T+F            | none              |
| K         | T+F+K (sqrt-scaled) | none         |
| K+S       | T+F+K (sqrt-scaled) | Jacobian smooth (λ_smooth=0.5) |

## K Weight Scaling

```
λ_K = 0.1 × sqrt(D / D_ref),   D_ref = 11
```

| D   | λ_K  |
|-----|------|
| 11  | 0.10 |
| 201 | 0.43 |

## Jacobian Smoothness Regularization

Applied in Stage 2 (drift_net training) only. Penalizes the Frobenius norm of the drift_net Jacobian at augmented points:

```
L_smooth = λ_smooth × tr(J_bz^T g J_bz g^{-1}) / D,   at z + ε,  ε ~ N(0, σ_aug^2 I)
```

- `σ_aug = 0.1`, `λ_smooth = 0.5`
- **Normalised by D** (ambient dimension), matching `tangential_drift_loss` which also divides by D

### Critical bug fix (2026-03-10)

Previously `drift_smoothness_loss` normalised by `d` (latent dim = 2) while `tangential_drift_loss` normalised by `D` (ambient dim). This made the effective smoothing weight scale as `λ × D/d`:

| D   | Old effective λ | New effective λ |
|-----|-----------------|-----------------|
| 11  | 0.5 × 11/2 = 2.75 | 0.5 (correct) |
| 201 | 0.5 × 201/2 = 50.3 | 0.5 (correct) |

At D=201, smoothing was **18× too aggressive**, causing K+S to significantly hurt hyp_parab (+14-17% W2). After the fix, K+S is neutral on hyp_parab and still helps paraboloid.

## Results: D=11

### Paraboloid

K alone improves E_mu but **not** W2. K+S bridges the gap.

| N   | ΔE_mu (K) | ΔW2 (K) | ΔW2 (K+S) | K+S W2 p-value |
|-----|-----------|---------|------------|----------------|
| 20  | -41.1%**  | -6.0%+  | **-15.2%** | <0.001**       |
| 50  | -38.7%**  | -3.8%   | **-15.3%** | <0.001**       |
| 100 | -37.6%**  | -2.5%   | **-19.2%** | <0.001**       |
| 200 | -45.7%*   | -0.8%   | **-24.5%** | <0.001**       |

### Hyperbolic Paraboloid

K+S helps at N=20, 50, 200. Significant improvements up to -21%.

| N   | ΔE_mu (K) | ΔW2 (K) | ΔW2 (K+S) | K+S W2 p-value |
|-----|-----------|---------|------------|----------------|
| 20  | -53.6%+   | -6.4%   | **-20.6%** | 0.018*         |
| 50  | -52.5%**  | -3.5%+  | **-10.5%** | 0.003**        |
| 100 | -28.2%**  | +1.5%   | -7.7%      | 0.149          |
| 200 | -34.0%    | +3.8%   | **-18.6%** | 0.001**        |

## Results: D=201

### Paraboloid

K+S helps at N ≥ 50, up to -16% W2.

| N   | ΔE_mu (K) | ΔW2 (K) | ΔW2 (K+S) | K+S W2 p-value |
|-----|-----------|---------|------------|----------------|
| 20  | -11.3%*   | -0.4%   | +2.1%      | 0.612          |
| 50  | -23.7%**  | -2.1%   | **-12.8%** | 0.008**        |
| 100 | -32.4%*   | -9.6%*  | **-12.7%** | 0.015*         |
| 200 | -1.2%     | -7.5%*  | **-15.8%** | <0.001**       |

### Hyperbolic Paraboloid

K improves E_mu. K+S is **neutral** on trajectories (no harm, no significant help).

| N   | ΔE_mu (K) | ΔW2 (K) | ΔW2 (K+S) | K+S W2 p-value |
|-----|-----------|---------|------------|----------------|
| 20  | -11.9%**  | +3.7%   | +2.5%      | 0.613          |
| 50  | -25.2%**  | -5.9%   | -1.9%      | 0.666          |
| 100 | -29.0%*   | +0.4%   | +1.5%      | 0.736          |
| 200 | +22.9%    | -2.4%   | -1.6%      | 0.639          |

## Key Takeaways

1. **K improves E_mu consistently** across D=11 and D=201, both surfaces. The sqrt(D/D_ref) scaling keeps this stable.

2. **K alone does NOT improve W2.** The "chart-to-trajectory gap" — better chart ≠ better trajectories because drift_net overfits to noise.

3. **K+S bridges the gap.** At D=11: up to -25% W2 (paraboloid) and -21% (hyp_parab). At D=201: up to -16% W2 (paraboloid). Hyp_parab at D=201 is neutral.

4. **Smoothness normalisation matters.** The penalty must be normalised by D (not d) to maintain consistent effective weight across ambient dimensions. Old /d normalisation made smoothing 18× too aggressive at D=201.

5. **Complementary roles:** K gives geometric chart awareness (better E_mu). S regularises drift_net against noise (better W2). Together they form a complete pipeline.

## Experiment Files

- Results: `highd_N_D_sweep.csv` (480 rows, D-normalised smoothness)
- Old results (pre-fix): `highd_N_D_sweep_lam01.csv`, SLURM logs `nd_sweep_59238.log`, `nd_sweep_lam01_59288.log`
- Current SLURM log: `nd_sweep_59348.log`
- Scripts: `run_nd_sweep.sh`, `run_nd_sweep_lam01.sh`
