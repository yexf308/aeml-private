"""Generate publication-quality figures for the paper.

Reads pre-computed CSV results and produces PDF + PNG figures.

Usage:
    python -m experiments.plot_paper_figures           # all figures
    python -m experiments.plot_paper_figures --fig1     # extrapolation only
    python -m experiments.plot_paper_figures --fig2     # trajectory MTE only
    python -m experiments.plot_paper_figures --fig3     # trajectory W2 only
"""

import argparse
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Shared style
# ---------------------------------------------------------------------------

FOCUS = ["baseline", "T+F", "T+F+K"]
COLORS = {
    "baseline": "#999999",
    "T+F": "#2176AE",
    "T+F+K": "#D64045",
    "T+K": "#E8A838",
}
LINESTYLES = {"baseline": "--", "T+F": "-", "T+F+K": "-", "T+K": ":"}
MARKERS = {"baseline": "o", "T+F": "s", "T+F+K": "D", "T+K": "^"}
LABELS = {"baseline": "Baseline", "T+F": "T+F", "T+F+K": "T+F+K", "T+K": "T+K"}

SURFACE_TITLES = {
    "paraboloid": "Paraboloid",
    "hyperbolic_paraboloid": "Hyp. paraboloid",
    "monkey_saddle": "Monkey saddle",
    "sinusoidal": "Sinusoidal",
}

OUTPUT_DIR = Path("Autoencoder-Paper")


def _apply_style():
    """Set global matplotlib style for publication figures."""
    mpl.rcParams.update({
        "font.family": "serif",
        "font.size": 9,
        "axes.labelsize": 10,
        "axes.titlesize": 10,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "legend.fontsize": 8,
        "lines.linewidth": 1.5,
        "lines.markersize": 4,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.alpha": 0.15,
        "savefig.dpi": 200,
        "savefig.bbox": "tight",
        "figure.constrained_layout.use": True,
    })


def _save(fig, name):
    """Save figure as PDF and PNG to the output directory."""
    for ext in ("pdf", "png"):
        path = OUTPUT_DIR / f"{name}.{ext}"
        fig.savefig(path, dpi=200)
        print(f"  saved {path}")


# ---------------------------------------------------------------------------
# Figure 1: Reconstruction extrapolation (1×4)
# ---------------------------------------------------------------------------

def plot_extrapolation():
    """Reconstruction error vs extrapolation distance for four surfaces."""
    df = pd.read_csv("extrapolation_all_surfaces.csv")
    surfaces = ["paraboloid", "hyperbolic_paraboloid", "monkey_saddle", "sinusoidal"]
    conditions = ["baseline", "T+F", "T+F+K", "T+K"]

    fig, axes = plt.subplots(1, 4, figsize=(7.0, 1.9), sharey=False)

    for ax, surf in zip(axes, surfaces):
        ds = df[df["surface"] == surf]
        # Shade region between baseline and T+F
        bl = ds[ds["penalty"] == "baseline"].sort_values("distance")
        tf = ds[ds["penalty"] == "T+F"].sort_values("distance")
        if len(bl) == len(tf):
            ax.fill_between(
                bl["distance"].values,
                bl["reconstruction_error"].values,
                tf["reconstruction_error"].values,
                color=COLORS["T+F"], alpha=0.08,
            )
        # Plot each condition
        for cond in conditions:
            sub = ds[ds["penalty"] == cond].sort_values("distance")
            if sub.empty:
                continue
            ax.plot(
                sub["distance"], sub["reconstruction_error"],
                color=COLORS[cond], ls=LINESTYLES[cond],
                marker=MARKERS[cond], markersize=3.5,
                label=LABELS[cond],
            )
        ax.set_xlabel(r"Distance $\delta$")
        ax.set_title(SURFACE_TITLES[surf])

    axes[0].set_ylabel("Reconstruction error")
    # Single legend on rightmost panel
    axes[-1].legend(loc="upper left", framealpha=0.9)

    _save(fig, "fig_extrapolation")
    plt.close(fig)
    print("Figure 1 (extrapolation) done.")


# ---------------------------------------------------------------------------
# Figure 2: Trajectory MTE time series (1×3)
# ---------------------------------------------------------------------------

def plot_traj_mte():
    """MTE vs time for three surfaces (end_to_end sim mode)."""
    df = pd.read_csv("trajectory_fidelity_train.csv")
    surfaces = ["paraboloid", "hyperbolic_paraboloid", "sinusoidal"]
    times = [0.1, 0.2, 0.5, 1.0]

    fig, axes = plt.subplots(1, 3, figsize=(5.5, 2.0), sharey=False)

    for ax, surf in zip(axes, surfaces):
        ds = df[(df["surface"] == surf) & (df["sim_mode"] == "end_to_end")]
        ds = ds[ds["time"].isin(times) & ds["MTE"].notna()]
        for cond in FOCUS:
            sub = ds[ds["penalty"] == cond].sort_values("time")
            if sub.empty:
                continue
            ax.plot(
                sub["time"], sub["MTE"],
                color=COLORS[cond], ls=LINESTYLES[cond],
                marker=MARKERS[cond], markersize=3.5,
                label=LABELS[cond],
            )
        ax.set_xlabel("Time")
        ax.set_title(SURFACE_TITLES[surf])
        ax.set_yscale("log")
        ax.set_xticks(times)

    axes[0].set_ylabel("MTE (log scale)")
    axes[-1].legend(loc="best", framealpha=0.9)

    _save(fig, "fig_traj_mte")
    plt.close(fig)
    print("Figure 2 (trajectory MTE) done.")


# ---------------------------------------------------------------------------
# Figure 3: Trajectory W2 time series (1×3)
# ---------------------------------------------------------------------------

def plot_traj_w2():
    """W2 distance vs time for three surfaces (end_to_end sim mode)."""
    df = pd.read_csv("trajectory_fidelity_train.csv")
    surfaces = ["paraboloid", "hyperbolic_paraboloid", "sinusoidal"]
    times = [1.0, 2.0, 3.0, 4.0, 5.0]

    fig, axes = plt.subplots(1, 3, figsize=(5.5, 2.0), sharey=False)

    for ax, surf in zip(axes, surfaces):
        ds = df[(df["surface"] == surf) & (df["sim_mode"] == "end_to_end")]
        ds = ds[ds["time"].isin(times) & ds["W2"].notna()]
        for cond in FOCUS:
            sub = ds[ds["penalty"] == cond].sort_values("time")
            if sub.empty:
                continue
            ax.plot(
                sub["time"], sub["W2"],
                color=COLORS[cond], ls=LINESTYLES[cond],
                marker=MARKERS[cond], markersize=3.5,
                label=LABELS[cond],
            )
        ax.set_xlabel("Time")
        ax.set_title(SURFACE_TITLES[surf])
        ax.set_xticks(times)

    axes[0].set_ylabel(r"$W_2$ distance")
    axes[-1].legend(loc="best", framealpha=0.9)

    _save(fig, "fig_traj_w2")
    plt.close(fig)
    print("Figure 3 (trajectory W2) done.")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    _apply_style()
    parser = argparse.ArgumentParser(description="Generate paper figures.")
    parser.add_argument("--fig1", action="store_true", help="Extrapolation figure")
    parser.add_argument("--fig2", action="store_true", help="Trajectory MTE figure")
    parser.add_argument("--fig3", action="store_true", help="Trajectory W2 figure")
    args = parser.parse_args()

    run_all = not (args.fig1 or args.fig2 or args.fig3)

    if run_all or args.fig1:
        plot_extrapolation()
    if run_all or args.fig2:
        plot_traj_mte()
    if run_all or args.fig3:
        plot_traj_w2()


if __name__ == "__main__":
    main()
