"""
Generate LaTeX table for paper: trajectory fidelity with drift smoothing.

Combines existing paper data (baseline, T+F, T+F+K) with new smoothing data (T+F+S)
to produce the updated tab:traj_fidelity table.

Usage:
    python -m experiments.generate_smoothing_table
"""
import numpy as np
import pandas as pd
from scipy import stats


def load_existing_data():
    """Load existing paper trajectory data for baseline/T+F/T+F+K."""
    # Main file has paraboloid, hyp_parab, sinusoidal
    df1 = pd.read_csv("paper_trajectory.csv")
    df2 = pd.read_csv("paper_trajectory_qdome.csv")
    df = pd.concat([df1, df2], ignore_index=True)

    # Filter to t=1.0, learned_latent for W2
    mask_w2 = (df["time"] == 1.0) & (df["sim_mode"] == "learned_latent")
    df_w2 = df[mask_w2][["surface", "config", "seed", "W2"]].copy()

    # E_mu from end_to_end at t=1.0
    mask_emu = (df["time"] == 1.0) & (df["sim_mode"] == "end_to_end")
    df_emu = df[mask_emu][["surface", "config", "seed", "drift_coeff_error"]].copy()
    df_emu = df_emu.rename(columns={"drift_coeff_error": "E_mu"})

    # Merge
    df_merged = df_w2.merge(df_emu, on=["surface", "config", "seed"], how="outer")
    # Drop duplicates
    df_merged = df_merged.drop_duplicates(subset=["surface", "config", "seed"])
    return df_merged


def load_smoothing_data():
    """Load new smoothing trajectory data."""
    df = pd.read_csv("paper_trajectory_smoothing.csv")
    # Rename columns to match
    out = df[["surface", "config", "seed", "W2@1.0", "E_mu_held"]].copy()
    out = out.rename(columns={"W2@1.0": "W2", "E_mu_held": "E_mu"})
    return out


def format_val(mean, std, bold=False):
    """Format mean ± std for LaTeX."""
    s = f"${mean:.2f} \\pm {std:.2f}$"
    if bold:
        s = f"$\\mathbf{{{mean:.2f} \\pm {std:.2f}}}$"
    return s


def paired_test(a, b):
    """Paired t-test, return (delta%, p-value, n_wins)."""
    mask = np.isfinite(a) & np.isfinite(b)
    if mask.sum() < 2:
        return 0, 1.0, 0
    ac, bc = a[mask], b[mask]
    delta = (bc.mean() - ac.mean()) / ac.mean() * 100
    n_wins = int((bc < ac).sum())
    _, p = stats.ttest_rel(bc, ac)
    return delta, p, n_wins


def main():
    df_existing = load_existing_data()
    df_smooth = load_smoothing_data()

    surfaces = ["paraboloid", "hyperbolic_paraboloid", "quartic_dome", "sinusoidal"]
    surface_labels = {
        "paraboloid": "Paraboloid",
        "hyperbolic_paraboloid": "Hyp.\\ parab.",
        "quartic_dome": "Quartic dome",
        "sinusoidal": "Sinusoidal",
    }
    configs_existing = ["baseline", "T+F", "T+F+K"]
    configs_smooth = ["T+F", "T+F+S"]

    print("=" * 80)
    print("UPDATED TRAJECTORY FIDELITY TABLE")
    print("=" * 80)

    # For each surface, print mean ± std for all 4 configs
    for surf in surfaces:
        print(f"\n{surface_labels[surf]}:")
        for metric_name, metric_col in [("W2", "W2"), ("E_mu", "E_mu")]:
            vals = {}
            for config in configs_existing:
                d = df_existing[(df_existing["surface"] == surf) & (df_existing["config"] == config)]
                v = d[metric_col].dropna().values
                if len(v) > 0:
                    vals[config] = (v.mean(), v.std(), v)
                    print(f"  {metric_name} {config}: {v.mean():.3f} ± {v.std():.3f} (n={len(v)})")

            # T+F+S from smoothing data
            d = df_smooth[(df_smooth["surface"] == surf) & (df_smooth["config"] == "T+F+S")]
            v = d[metric_col].dropna().values
            if len(v) > 0:
                vals["T+F+S"] = (v.mean(), v.std(), v)
                print(f"  {metric_name} T+F+S: {v.mean():.3f} ± {v.std():.3f} (n={len(v)})")

            # T+F from smoothing data (for paired comparison)
            d_tf = df_smooth[(df_smooth["surface"] == surf) & (df_smooth["config"] == "T+F")]
            v_tf = d_tf[metric_col].dropna().values
            if len(v_tf) > 0:
                print(f"  {metric_name} T+F (new run): {v_tf.mean():.3f} ± {v_tf.std():.3f} (n={len(v_tf)})")

            # Paired test: T+F+S vs T+F (from same run)
            if "T+F+S" in vals and len(v_tf) == len(v):
                delta, p, n_wins = paired_test(v_tf, v)
                sig = "**" if p < 0.01 else ("*" if p < 0.05 else ("+" if p < 0.1 else ""))
                print(f"    T+F+S vs T+F (paired): {delta:+.1f}% {n_wins}/{len(v_tf)} wins p={p:.4f} {sig}")

    # Generate LaTeX table
    print("\n" + "=" * 80)
    print("LATEX TABLE")
    print("=" * 80)

    print(r"""\begin{table}[ht]
\centering
\caption{Trajectory fidelity and SDE coefficient errors
($N=20$, 10 seeds, $t=1.0$).
$W_2$ is the Wasserstein-2 distance between learned and true
endpoint clouds (learned-latent simulation).
$\cE_\mu$ is the per-seed median
$\norm{D\phi_\theta\,\hat\mu + \frac{1}{2}q(\phi_\theta,\hat\Sigma) - b}_2^2$ over surviving
particles (end-to-end trajectories), averaged over seeds.
Best value in each row is \textbf{bold}.}
\label{tab:traj_fidelity}
\small
\begin{tabular}{llcccc}
\toprule
Surface & Metric & baseline & T+F & T+F+K & T+F+S\\
\midrule""")

    for i, surf in enumerate(surfaces):
        label = surface_labels[surf]

        for metric_name, metric_col in [("$W_2$", "W2"), ("$\\cE_\\mu$", "E_mu")]:
            all_vals = {}
            for config in configs_existing:
                d = df_existing[(df_existing["surface"] == surf) & (df_existing["config"] == config)]
                v = d[metric_col].dropna().values
                if len(v) > 0:
                    all_vals[config] = (v.mean(), v.std())

            d = df_smooth[(df_smooth["surface"] == surf) & (df_smooth["config"] == "T+F+S")]
            v = d[metric_col].dropna().values
            if len(v) > 0:
                all_vals["T+F+S"] = (v.mean(), v.std())

            # Find best (lowest mean)
            if all_vals:
                best_config = min(all_vals, key=lambda k: all_vals[k][0])
            else:
                best_config = None

            cells = []
            for config in ["baseline", "T+F", "T+F+K", "T+F+S"]:
                if config in all_vals:
                    m, s = all_vals[config]
                    bold = (config == best_config)
                    cells.append(format_val(m, s, bold))
                else:
                    cells.append("---")

            prefix = label if metric_name == "$W_2$" else ""
            row = f"{prefix}\n & {metric_name} & {' & '.join(cells)}\\\\"
            print(row)

        if i < len(surfaces) - 1:
            print(r"\midrule")

    print(r"""\bottomrule
\end{tabular}
\end{table}""")


if __name__ == "__main__":
    main()
