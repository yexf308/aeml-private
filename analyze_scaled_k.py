"""Compare scaled-K results (D=201) against original fixed-K results."""
import pandas as pd
import numpy as np
from scipy import stats

# Load original results
orig = pd.read_csv('highd_N_D_sweep.csv')
# Filter to D=201, N in {20, 50}, seeds 42-4042
orig = orig[(orig['D'] == 201) & (orig['N'].isin([20, 50])) &
            (orig['seed'].isin([42, 1042, 2042, 3042, 4042]))]

# Load scaled results
try:
    scaled = pd.read_csv('highd_N_D_sweep_scaled_k.csv')
except FileNotFoundError:
    print("Scaled results CSV not found yet.")
    exit(1)

print("=" * 80)
print("COMPARISON: Fixed K (0.1) vs Scaled K (1.83) at D=201")
print("=" * 80)

for metric in ['E_mu', 'E_Sigma', 'W2@1.0']:
    print(f"\n{'─'*60}")
    print(f"  {metric}")
    print(f"{'─'*60}")
    print(f"{'Surface':<20} {'N':>3}  {'Fixed K':>18}  {'Scaled K':>18}  {'Δ':>8}")

    for surface in ['paraboloid', 'hyperbolic_paraboloid']:
        for N in [20, 50]:
            # Fixed K
            mask_o = (orig['surface'] == surface) & (orig['N'] == N)
            b_o = orig[mask_o & (orig['condition'] == 'baseline')].sort_values('seed')[metric].values
            k_o = orig[mask_o & (orig['condition'] == 'K(Phat)')].sort_values('seed')[metric].values

            # Scaled K
            mask_s = (scaled['surface'] == surface) & (scaled['N'] == N)
            b_s = scaled[mask_s & (scaled['condition'] == 'baseline')].sort_values('seed')[metric].values
            k_s = scaled[mask_s & (scaled['condition'] == 'K(Phat)')].sort_values('seed')[metric].values

            if len(b_o) == 0 or len(k_o) == 0 or len(b_s) == 0 or len(k_s) == 0:
                continue

            # Delta-of-means
            delta_fixed = (k_o.mean() - b_o.mean()) / b_o.mean() * 100
            delta_scaled = (k_s.mean() - b_s.mean()) / b_s.mean() * 100

            # Paired t-test
            _, p_fixed = stats.ttest_rel(k_o, b_o)
            _, p_scaled = stats.ttest_rel(k_s, b_s)

            sig_f = '**' if p_fixed < 0.01 else ('*' if p_fixed < 0.05 else ('+' if p_fixed < 0.1 else ''))
            sig_s = '**' if p_scaled < 0.01 else ('*' if p_scaled < 0.05 else ('+' if p_scaled < 0.1 else ''))

            n_help_f = (k_o < b_o).sum()
            n_help_s = (k_s < b_s).sum()

            print(f"{surface:<20} {N:>3}  "
                  f"{delta_fixed:>+6.1f}% p={p_fixed:.3f}{sig_f:<2} {n_help_f}/5  "
                  f"{delta_scaled:>+6.1f}% p={p_scaled:.3f}{sig_s:<2} {n_help_s}/5  "
                  f"{'BETTER' if abs(delta_scaled) > abs(delta_fixed) and delta_scaled < 0 else ''}")

# Also check: did scaling K hurt reconstruction?
print(f"\n{'─'*60}")
print(f"  Reconstruction quality check (per-dim recon error)")
print(f"{'─'*60}")
for surface in ['paraboloid', 'hyperbolic_paraboloid']:
    for N in [20, 50]:
        mask_s = (scaled['surface'] == surface) & (scaled['N'] == N)
        b_drift = scaled[mask_s & (scaled['condition'] == 'baseline')]['drift_loss'].values
        k_drift = scaled[mask_s & (scaled['condition'] == 'K(Phat)')]['drift_loss'].values
        if len(b_drift) > 0 and len(k_drift) > 0:
            print(f"{surface:<20} {N:>3}  baseline drift_loss={b_drift.mean():.6f}  "
                  f"K drift_loss={k_drift.mean():.6f}")
