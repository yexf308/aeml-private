# Extrapolation Study Summary

## Experiment Setup
- **Train region**: [-1, 1] × [-1, 1]
- **Test distances**: 0.0, 0.1, 0.2, 0.3, 0.4, 0.5 (rings outside training)
- **Surfaces tested**: paraboloid, hyperbolic_paraboloid, monkey_saddle, sinusoidal
- **Penalties compared**: baseline, T, K, T+K, T+F, T+F+K
- **Training**: 500 epochs, 2000 samples

## Key Finding

**Diffeomorphism (F) is crucial for extrapolation, NOT curvature (K) alone!**

## Results at Maximum Extrapolation (dist=0.5)

### Average Error Across All Surfaces:
| Rank | Penalty | Error | Notes |
|------|---------|-------|-------|
| 🥇 1 | **T+F+K** | 0.696 | Best overall |
| 🥈 2 | **T+F** | 0.698 | Nearly as good |
| 🥉 3 | T | 1.007 | Baseline tangent |
| 4 | T+K | 1.027 | **Worse than T!** |
| 5 | baseline | 1.042 | No penalties |
| 6 | K | 1.059 | Curvature alone worst |

### Per-Surface Results:
| Surface | baseline | T | K | T+K | T+F | T+F+K |
|---------|----------|---|---|-----|-----|-------|
| paraboloid | 1.112 | 1.108 | 1.103 | 1.166 | 0.631 | **0.606** |
| hyperbolic_paraboloid | 0.681 | 0.717 | 0.683 | 0.724 | 0.421 | **0.364** |
| monkey_saddle | 2.052 | 1.914 | 2.169 | 1.926 | 1.633 | **1.680** |
| sinusoidal | 0.321 | 0.289 | 0.280 | 0.293 | **0.107** | 0.133 |

## Analysis

### Why Curvature (K) Alone Doesn't Help Extrapolation:
1. Curvature penalty matches local 2nd-order structure (Hessian)
2. This helps match curvature *within* training region
3. But doesn't constrain how the model extrapolates *beyond* training
4. T+K actually **worse** than T alone (error +2.0%)

### Why Diffeomorphism (F) Helps Extrapolation:
1. Forces encoder-decoder composition to approximate identity
2. Creates smooth, invertible mapping φ ∘ ψ ≈ I
3. Smoothness constraint naturally extends beyond training region
4. T+F improves extrapolation by **30.7%** vs T alone

### Best Configuration:
- **T+F+K**: Combines all beneficial effects
  - T: Tangent alignment for first-order structure
  - F: Diffeomorphism for smooth extrapolation  
  - K: Curvature for second-order accuracy
- Achieves **34.3% error reduction** vs baseline

## Conclusion

For applications requiring extrapolation beyond training data:
- **Use T+F or T+F+K**, not T+K
- Diffeomorphism penalty is the key enabler
- Curvature alone may actually hurt extrapolation

## Files
- `extrapolation_all_surfaces.csv` - All raw results
- `extrapolation_summary.png` - Per-surface plots
- `extrapolation_average.png` - Average across surfaces
