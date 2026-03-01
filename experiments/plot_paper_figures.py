"""Generate publication-quality figures for the paper.

Reads pre-computed CSV results and produces PDF + PNG figures.

Usage:
    python -m experiments.plot_paper_figures           # all figures
    python -m experiments.plot_paper_figures --fig1     # extrapolation only
    python -m experiments.plot_paper_figures --fig2     # trajectory MTE only
    python -m experiments.plot_paper_figures --fig3     # trajectory W2 only
    python -m experiments.plot_paper_figures --fig6     # extrapolation region schematic
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
# Figure 6: Extrapolation region schematic (2-panel)
# ---------------------------------------------------------------------------

RING_COLORS = ["#2176AE", "#5BA4CF", "#8DC8E8", "#C0DFEE", "#DAE9F2", "#EDF4F8"]
TRAIN_COLOR = "#D64045"
TRAIN_BOUND = 1.0
DIST_STEP = 0.1
DISTANCES = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]


def plot_extrap_regions():
    """Schematic showing training region and extrapolation rings.

    Left panel: 2D (u,v) domain.
    Right panel: 3D paraboloid surface with same coloring.
    """
    from matplotlib.patches import Rectangle
    from matplotlib.colors import LinearSegmentedColormap

    res = 300  # grid resolution
    outer = TRAIN_BOUND + DISTANCES[-1]
    u = np.linspace(-outer, outer, res)
    v = np.linspace(-outer, outer, res)
    U, V = np.meshgrid(u, v)
    Z_parab = U**2 + V**2  # paraboloid: z = u^2 + v^2

    # Assign each point to a distance band (0 = training, 1..5 = rings)
    band = np.full(U.shape, np.nan)
    # Training region
    train_mask = (np.abs(U) <= TRAIN_BOUND) & (np.abs(V) <= TRAIN_BOUND)
    band[train_mask] = -1  # sentinel for training

    for i, dist in enumerate(DISTANCES[1:]):  # skip 0.0
        inner = TRAIN_BOUND + dist - DIST_STEP
        ring_outer = TRAIN_BOUND + dist
        in_outer = (np.abs(U) <= ring_outer) & (np.abs(V) <= ring_outer)
        in_inner = (np.abs(U) <= inner) & (np.abs(V) <= inner)
        ring_mask = in_outer & ~in_inner
        band[ring_mask] = i

    # Build a custom colormap: training (red) then rings (blue gradient)
    from matplotlib.colors import ListedColormap
    cmap_list = [TRAIN_COLOR] + RING_COLORS[:len(DISTANCES) - 1]
    cmap = ListedColormap(cmap_list)
    # Map band values: -1 -> 0, 0 -> 1, 1 -> 2, etc.
    band_mapped = band + 1  # training=-1 -> 0, ring0=0 -> 1, ...
    vmin, vmax = 0, len(DISTANCES) - 1

    fig = plt.figure(figsize=(5.5, 2.5))

    # --- Left panel: 2D domain ---
    ax_2d = fig.add_subplot(1, 2, 1)
    ax_2d.pcolormesh(U, V, band_mapped, cmap=cmap, vmin=vmin, vmax=vmax,
                     shading="auto", rasterized=True)
    # Annotate training region
    rect = Rectangle((-TRAIN_BOUND, -TRAIN_BOUND), 2 * TRAIN_BOUND, 2 * TRAIN_BOUND,
                      linewidth=1.0, edgecolor="k", facecolor="none", linestyle="-")
    ax_2d.add_patch(rect)
    ax_2d.text(0, 0, "Train", ha="center", va="center", fontsize=8,
               fontweight="bold", color="white")
    # Annotate selected rings with arrows pointing to them
    for dist, y_off in [(0.2, 0.55), (0.5, -0.55)]:
        ring_center_x = TRAIN_BOUND + dist - DIST_STEP / 2
        ax_2d.annotate(
            rf"$\delta\!=\!{dist}$",
            xy=(ring_center_x, y_off),
            xytext=(ring_center_x + 0.22, y_off),
            fontsize=6.5, ha="left", va="center", color="k",
            arrowprops=dict(arrowstyle="-", color="k", lw=0.5),
        )
    ax_2d.set_xlabel(r"$u$")
    ax_2d.set_ylabel(r"$v$")
    ax_2d.set_title("Local coordinates")
    ax_2d.set_aspect("equal")
    ax_2d.set_xlim(-outer - 0.05, outer + 0.05)
    ax_2d.set_ylim(-outer - 0.05, outer + 0.05)

    # --- Right panel: 3D surface ---
    ax_3d = fig.add_subplot(1, 2, 2, projection="3d")
    # Use facecolors mapped from bands
    norm = plt.Normalize(vmin=vmin, vmax=vmax)
    fc = cmap(norm(band_mapped))
    # Mask NaN regions: set z to NaN so surface isn't drawn there
    nan_mask = np.isnan(band)
    Z_plot = Z_parab.copy()
    Z_plot[nan_mask] = np.nan

    ax_3d.plot_surface(U, V, Z_plot, facecolors=fc, rstride=2, cstride=2,
                       linewidth=0, antialiased=True, shade=True)
    # Draw training boundary on the surface
    tb = TRAIN_BOUND
    n_bnd = 50
    sides = [
        (np.linspace(-tb, tb, n_bnd), np.full(n_bnd, -tb)),
        (np.full(n_bnd, tb), np.linspace(-tb, tb, n_bnd)),
        (np.linspace(tb, -tb, n_bnd), np.full(n_bnd, tb)),
        (np.full(n_bnd, -tb), np.linspace(tb, -tb, n_bnd)),
    ]
    for bu, bv in sides:
        bz = bu**2 + bv**2
        ax_3d.plot(bu, bv, bz, color="k", linewidth=1.0)

    ax_3d.set_xlabel(r"$u$", labelpad=1)
    ax_3d.set_ylabel(r"$v$", labelpad=1)
    ax_3d.set_zlabel(r"$z$", labelpad=1)
    ax_3d.set_title("Paraboloid")
    ax_3d.view_init(elev=30, azim=-50)
    ax_3d.tick_params(labelsize=6, pad=0)
    ax_3d.xaxis.pane.fill = False
    ax_3d.yaxis.pane.fill = False
    ax_3d.zaxis.pane.fill = False

    _save(fig, "fig_extrap_regions")
    plt.close(fig)
    print("Figure 6 (extrapolation regions) done.")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    _apply_style()
    parser = argparse.ArgumentParser(description="Generate paper figures.")
    parser.add_argument("--fig1", action="store_true", help="Extrapolation figure")
    parser.add_argument("--fig2", action="store_true", help="Trajectory MTE figure")
    parser.add_argument("--fig3", action="store_true", help="Trajectory W2 figure")
    parser.add_argument("--fig6", action="store_true", help="Extrapolation region schematic")
    args = parser.parse_args()

    run_all = not (args.fig1 or args.fig2 or args.fig3 or args.fig6)

    if run_all or args.fig1:
        plot_extrapolation()
    if run_all or args.fig2:
        plot_traj_mte()
    if run_all or args.fig3:
        plot_traj_w2()
    if run_all or args.fig6:
        plot_extrap_regions()


if __name__ == "__main__":
    main()
