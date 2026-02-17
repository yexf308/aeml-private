# Design: Full-Space Curvature Loss (K-full)

**Date:** 2026-02-16
**Problem:** The curvature penalty (K) does not improve trajectory fidelity. Adding K to T+F consistently makes MTE worse (by 1-14%) even in modes that never use the Hessian.

## Root Cause Analysis

Five compounding reasons K fails:

1. **Optimization interference** — K's third-order gradient signals destabilize the first-order Jacobian that T and F optimize. Evidence: T+F+K has worse MTE than T+F in learned_latent mode (which never uses the Hessian), proving K degrades reconstruction quality.

2. **First-order error dominance** — Jacobian errors in `dphi @ b_z` produce larger trajectory errors than Hessian errors in `0.5 * tr(Λ·H)`.

3. **Temporal accumulation** — Over 500 Euler-Maruyama steps, first-order drift errors dwarf any Hessian improvement.

4. **Inversion amplification** — In end-to-end mode, Hessian errors are amplified by `g_inv @ dphi^T`, causing trajectory divergence.

5. **Normal-vs-full mismatch** — K only constrains the normal projection `(I-P) · 0.5·tr(Λ·H)`, but simulation uses the full Hessian contraction (all D components). Unconstrained tangential Hessian errors pollute trajectories.

## Solution: Full-Space Curvature Loss

### Core Change

Remove the normal projection from the curvature loss. Instead of matching the normal component of the Ito correction, match the **full ambient-space Ito correction vector**.

**Current K loss:**
```
L_K = ‖(I-P̂) · q_model - (I-P) · μ_true‖²
```

**Proposed K-full loss:**
```
L_Kf = ‖q_model - q_true‖²
```

where:
- `q_model = 0.5 * einsum(Λ_z, d²φ_model)` — model Ito correction in z-coords → ambient vector
- `q_true = 0.5 * einsum(σσ^T_uv, H_true_uv)` — true Ito correction in (u,v) coords → ambient vector

Both produce ambient-space vectors that are directly comparable (coordinate-free).

### Proposition 8 Preserved

The Hessian-free JVP identity still applies:
```
Σ_{ij} Λ_{ij} · ∂²φ_r/∂z_i∂z_j = Σ_k λ_k · ∇²φ_r(e_k, e_k)
```
Only the final `(I-P) @` projection step is removed. The eigendecomposition + double-JVP computation is unchanged.

### Chart Invariance

The full Ito correction `0.5 * tr(Λ·H)` in ambient space is chart-invariant — it's the same geometric quantity regardless of parameterization. Removing the normal projection does not break chart invariance.

## Implementation Plan

### 1. Data: Add `local_cov` to DatasetBatch

- Add `local_cov: torch.Tensor` field (shape `(B, d, d)`) to `DatasetBatch`
- Compute from `manifold_sde.local_covariance` during `sample_from_manifold()`
- This provides the true `σσ^T` in local coordinates for the loss target
- Update `as_tuple()`, `from_tuple()`, and embedding utilities

### 2. Geometry: Add full-space curvature functions

In `geometry.py`:
- `curvature_drift_explicit_full(d2phi, local_cov)` — like `curvature_drift_explicit` but without `nhat @`
- `curvature_drift_hessian_free_full(decoder, z, local_cov)` — like `curvature_drift_hessian_free` but without `normal_proj @`

### 3. Loss: Add `curvature_full` weight

In `losses.py`:
- Add `curvature_full: float = 0.0` to `LossWeights`
- In `autoencoder_loss()`, when `curvature_full > 0`:
  - Model side: `0.5 * einsum(local_cov_z, d2phi)` where `local_cov_z = pinv(dphi) @ Λ @ pinv^T`
  - Target side: `0.5 * einsum(local_cov_true, hessians_true)` (precomputed from new data field)
  - Loss: `mean(‖model_ito - true_ito‖²)`

### 4. Experiment: New penalty configs + weight sweep

Add to `PENALTY_CONFIGS`:
```python
"T+Kf":     LossWeights(tangent_bundle=1.0, curvature_full=1.0),
"T+F+Kf":   LossWeights(tangent_bundle=1.0, diffeo=1.0, curvature_full=1.0),
```

Weight sweep experiment: test `curvature_full` in {0.1, 0.3, 0.5, 1.0} using the trajectory fidelity sweep infrastructure.

### 5. Fallback: Two-phase training

If full-space K still hurts (optimization interference persists):
- Phase 1: Train T+F for N epochs
- Phase 2: Freeze encoder, add Kf, fine-tune decoder for M epochs

## Success Criteria

- T+F+Kf should have **lower** MTE@1.0 than T+F in learned_latent mode (proving no optimization interference)
- T+F+Kf should have lower W2@5.0 than T+F in learned_ambient mode (proving Hessian improvement helps simulation)
- No catastrophic W2 blowups for T+F+Kf in the sweep

## Files to Modify

- `src/numeric/datasets.py` — add `local_cov` field
- `src/numeric/datagen.py` — compute and store local covariance
- `src/numeric/geometry.py` — add `_full` variants of curvature functions
- `src/numeric/losses.py` — add `curvature_full` weight and loss computation
- `experiments/common.py` — add new penalty configs
- `experiments/trajectory_fidelity_study.py` — add weight sweep capability
