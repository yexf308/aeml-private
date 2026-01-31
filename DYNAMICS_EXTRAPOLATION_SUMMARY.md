# Dynamics Extrapolation Study Summary

## Overview

This study tests whether different penalty terms help the autoencoder match **dynamics** (drift and covariance) when extrapolating beyond the training region. Unlike reconstruction-only tests, this evaluates the geometric consistency conditions derived from Itô's lemma and the Feynman-Kac formula.

## Theoretical Background

For a manifold SDE, the normal component of the ambient drift satisfies:

$$(I - P)b = \frac{1}{2} \text{II} : \Lambda$$

where:
- $P$ is the tangent space projector
- $b$ is the ambient drift
- $\text{II}$ is the second fundamental form (curvature)
- $\Lambda$ is the ambient covariance matrix

This means **curvature matching is essential for dynamics consistency**, not just reconstruction.

## Experiment Setup

| Parameter | Value |
|-----------|-------|
| Surface | Paraboloid |
| Training region | $[-1, 1] \times [-1, 1]$ |
| Test distances | 0.0, 0.1, 0.2, 0.3, 0.4, 0.5 |
| Dynamics | Non-trivial SDE (not RBM) |
| Training samples | 2000 |
| Test samples | 500 per distance |
| Epochs | 500 |

## Penalties Tested

| Symbol | Name | Weight | Description |
|--------|------|--------|-------------|
| T | Tangent Bundle | 1.0 | $\|\hat{P} - P\|_F^2$ - tangent space alignment |
| K | Curvature | 1.0 | $\|(I-P)b - \frac{1}{2}q\|^2$ - normal drift matching |
| F | Diffeomorphism | 1.0 | $\|\phi(\psi(x)) - x\|^2$ - invertibility constraint |

Combinations tested: `baseline`, `T`, `K`, `T+K`, `T+F`, `T+F+K`

## Results

### Normal Drift Matching (Key Metric for Dynamics)

The normal drift error measures how well the learned projector captures the true normal component of the drift:

| Distance | baseline | T | K | T+K | T+F | T+F+K | True |
|----------|----------|---|---|-----|-----|-------|------|
| 0.0 | 2.92 | 2.80 | 2.78 | 2.78 | 2.83 | **2.78** | 2.78 |
| 0.1 | 4.35 | 3.47 | 3.31 | 3.26 | 3.41 | **3.12** | 3.04 |
| 0.2 | 5.02 | 3.83 | 3.54 | 3.47 | 3.64 | **3.22** | 3.00 |
| 0.3 | 6.58 | 4.77 | 4.37 | 4.25 | 4.46 | **3.84** | 3.33 |
| 0.4 | 8.25 | 5.82 | 5.22 | 5.05 | 5.28 | **4.49** | 3.57 |
| 0.5 | 10.20 | 7.09 | 6.36 | 6.15 | 6.46 | **5.43** | 3.89 |

**Best: T+F+K** at all distances.

### Relative Error vs True Normal Drift (at dist=0.5)

| Penalty | Normal Drift Error | Relative Error |
|---------|-------------------|----------------|
| **T+F+K** | 5.43 | **39.6%** |
| T+K | 6.15 | 58.1% |
| K | 6.36 | 63.5% |
| T+F | 6.46 | 66.0% |
| T | 7.09 | 82.3% |
| baseline | 10.20 | 162.2% |

### Curvature (K) Improvement vs Distance

| Distance | T | T+K | Improvement |
|----------|---|-----|-------------|
| 0.0 (interpolation) | 2.80 | 2.78 | +0.5% |
| 0.1 | 3.47 | 3.26 | +6.1% |
| 0.2 | 3.83 | 3.47 | +9.6% |
| 0.3 | 4.77 | 4.25 | +10.8% |
| 0.4 | 5.82 | 5.05 | +13.1% |
| 0.5 | 7.09 | 6.15 | **+13.3%** |

**Key finding**: Curvature penalty benefit **increases with extrapolation distance**.

### Reconstruction Error (at dist=0.5)

| Penalty | Reconstruction Error |
|---------|---------------------|
| **T+F** | **0.480** |
| T+F+K | 0.617 |
| T | 1.387 |
| baseline | 1.388 |
| T+K | 1.421 |
| K | 1.570 |

### Tangent Alignment Error (at dist=0.5)

| Penalty | Tangent Error |
|---------|---------------|
| **T+F+K** | **0.007** |
| T+F | 0.012 |
| T+K | 0.017 |
| K | 0.020 |
| T | 0.029 |
| baseline | 0.074 |

## Analysis

### Effect of Each Penalty

#### Tangent Bundle (T)
- Dramatically improves tangent alignment (0.074 → 0.029)
- Reduces normal drift error by ~30%
- Essential baseline for all other penalties

#### Curvature (K)
- Improves normal drift matching by 10-13% when combined with T
- Benefit increases with extrapolation distance
- Slightly hurts reconstruction (focuses on dynamics over point-wise accuracy)
- Most effective when combined with T+F

#### Diffeomorphism (F)
- Dramatically improves reconstruction (1.39 → 0.48)
- Creates smooth, invertible mapping that extrapolates well
- Essential for reconstruction-based extrapolation

### Synergy: T+F+K

The combination T+F+K achieves:
- **Best normal drift matching** (5.43 vs 7.09 for T alone)
- **Best tangent alignment** (0.007)
- Good reconstruction (0.617, only slightly worse than T+F)

Each component contributes:
- **T**: First-order structure (tangent space)
- **F**: Global smoothness (invertibility)
- **K**: Second-order structure (curvature/dynamics)

## Conclusions

1. **Curvature (K) is important for dynamics matching**, validating the theoretical prediction that $(I-P)b = \frac{1}{2}\text{II}:\Lambda$.

2. **K's benefit increases with extrapolation distance**: At interpolation, K helps minimally (+0.5%), but at dist=0.5, K improves normal drift matching by 13.3%.

3. **Previous reconstruction-only tests were misleading**: Curvature doesn't directly improve reconstruction - it ensures geometric consistency with the observed dynamics.

4. **Recommended configurations**:
   - For **dynamics consistency**: Use **T+F+K**
   - For **reconstruction only**: Use **T+F**
   - For **curvature matching within training**: Use **T+K**

5. **Diffeomorphism (F) remains essential** for reconstruction-based extrapolation, but **curvature (K) is essential for dynamics-based extrapolation**.

## Files

- `dynamics_extrapolation_results.csv` - Raw experimental data
- `experiments/dynamics_extrapolation_study.py` - Experiment script
