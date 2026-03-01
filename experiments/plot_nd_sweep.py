"""
Plot the N×D sweep results for the paper.

Figure 1 (nd_sweep_benefit.pdf):
  2×2 panel grid (rows: MTE, W2; cols: paraboloid, hyp. paraboloid).
  Each panel: x-axis = N, two lines for D=11 and D=201.
  y-axis = relative K benefit Δ(%) with SEM error bars.
  Significance stars above each point.

Figure 2 (nd_sweep_chart_quality.pdf):
  LOCA-style chart quality comparison.
  For a representative (N, D) cell, compare baseline vs K(Phat):
    - Decoded surface colored by latent coordinate
    - Pairwise distance scatter (true ambient vs learned ambient)

Usage:
    python -m experiments.plot_nd_sweep
    python -m experiments.plot_nd_sweep --fig1-only
    python -m experiments.plot_nd_sweep --fig2-only --seed 42 --n-train 20 --d-val 201
"""

import argparse
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from scipy import stats


# ── Figure 1: K benefit vs N ──────────────────────────────────────────────

def compute_benefit(df, surface, D, N, metric):
    """Compute relative benefit of K(Phat) over baseline for a (surface, D, N) cell."""
    base = df[(df["surface"] == surface) & (df["D"] == D) &
              (df["N"] == N) & (df["condition"] == "baseline")].sort_values("seed")
    kphat = df[(df["surface"] == surface) & (df["D"] == D) &
               (df["N"] == N) & (df["condition"] == "K(Phat)")].sort_values("seed")

    if len(base) == 0 or len(kphat) == 0:
        return None

    b_vals = base[metric].values
    k_vals = kphat[metric].values

    # Per-seed relative change (%)
    rel_change = (k_vals - b_vals) / b_vals * 100

    mean_delta = rel_change.mean()
    sem_delta = rel_change.std(ddof=1) / np.sqrt(len(rel_change))

    # Paired t-test on raw values
    t_stat, p_val = stats.ttest_rel(k_vals, b_vals)

    return {
        "mean_delta": mean_delta,
        "sem_delta": sem_delta,
        "p_val": p_val,
        "n_seeds": len(b_vals),
    }


def sig_marker(p):
    if p < 0.01:
        return "**"
    elif p < 0.05:
        return "*"
    elif p < 0.1:
        return "\u2020"  # dagger
    return ""


def plot_benefit_figure(df, outpath="nd_sweep_benefit"):
    """Create the 2×2 benefit panel figure."""
    surfaces = ["paraboloid", "hyperbolic_paraboloid"]
    surface_labels = ["Paraboloid", "Hyperbolic paraboloid"]
    metrics = ["MTE@1.0", "W2@1.0"]
    metric_labels = [r"$\Delta\,$MTE (%)", r"$\Delta\,W_2$ (%)"]
    D_values = [11, 201]
    N_values = sorted(df["N"].unique())

    # Use categorical x positions (equal spacing) with numeric labels
    x_ticks = np.arange(len(N_values))

    # Style
    colors = {11: "#2176AE", 201: "#D64045"}
    markers = {11: "o", 201: "s"}
    fills = {11: "#2176AE", 201: "white"}  # open markers for D=201

    fig, axes = plt.subplots(2, 2, figsize=(7.0, 5.5), sharex=True)

    for j, (surface, surf_label) in enumerate(zip(surfaces, surface_labels)):
        for i, (metric, met_label) in enumerate(zip(metrics, metric_labels)):
            ax = axes[i, j]

            for D in D_values:
                means, sems, ps = [], [], []
                for N in N_values:
                    res = compute_benefit(df, surface, D, N, metric)
                    if res is None:
                        means.append(np.nan)
                        sems.append(0)
                        ps.append(1.0)
                    else:
                        means.append(res["mean_delta"])
                        sems.append(res["sem_delta"])
                        ps.append(res["p_val"])

                means = np.array(means)
                sems = np.array(sems)

                # Small x-offset to avoid overlap
                offset = -0.06 if D == 11 else 0.06
                x_pos = x_ticks + offset

                ax.errorbar(
                    x_pos, means, yerr=sems,
                    marker=markers[D], color=colors[D],
                    markerfacecolor=fills[D],
                    markeredgecolor=colors[D], markeredgewidth=1.5,
                    linewidth=1.8, markersize=7, capsize=4, capthick=1.2,
                    label=f"$D={D}$  (codim. {D-2})",
                    zorder=3,
                )

                # Significance markers — place above each point
                for k, (xp, m, p) in enumerate(zip(x_pos, means, ps)):
                    star = sig_marker(p)
                    if star:
                        # Place above the error bar (or below if positive)
                        if m <= 0:
                            y_pos = m - sems[k] - 2.5
                            va = "top"
                        else:
                            y_pos = m + sems[k] + 1.5
                            va = "bottom"
                        ax.text(xp, y_pos, star,
                                ha="center", va=va, fontsize=10,
                                color=colors[D], fontweight="bold")

            # Zero line
            ax.axhline(0, color="0.4", linewidth=0.8, linestyle="-", alpha=0.4)

            ax.set_ylabel(met_label, fontsize=10)
            ax.tick_params(labelsize=9)

            if i == 0:
                ax.set_title(surf_label, fontsize=11, fontweight="bold",
                             pad=8)
            if i == 1:
                ax.set_xlabel("Training set size $N$", fontsize=10)
                ax.set_xticks(x_ticks)
                ax.set_xticklabels(N_values)

            # Only show legend in top-left
            if i == 0 and j == 0:
                ax.legend(fontsize=8.5, loc="lower right",
                          framealpha=0.95, edgecolor="0.7",
                          handlelength=2.5)

            # Minimal spine styling
            for spine in ["top", "right"]:
                ax.spines[spine].set_visible(False)
            ax.spines["left"].set_color("0.6")
            ax.spines["bottom"].set_color("0.6")

    # Synchronize y-limits per row
    for i in range(2):
        ymin = min(axes[i, jj].get_ylim()[0] for jj in range(2))
        ymax = max(axes[i, jj].get_ylim()[1] for jj in range(2))
        pad = (ymax - ymin) * 0.12
        for jj in range(2):
            axes[i, jj].set_ylim(ymin - pad, ymax + pad)

    # Add "K helps" annotation with downward arrow in bottom-left panel
    ax0 = axes[1, 0]
    ax0.annotate(
        "$K$ helps $\\downarrow$",
        xy=(0.97, 0.03), xycoords="axes fraction",
        fontsize=8, color="0.45", fontstyle="italic",
        ha="right", va="bottom",
    )

    fig.tight_layout(h_pad=0.8, w_pad=1.2)

    for fmt in ["pdf", "png"]:
        fig.savefig(f"{outpath}.{fmt}", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {outpath}.pdf and {outpath}.png")


# ── Figure 2: Chart quality panels (LOCA-style) ──────────────────────────

def plot_chart_quality(seed=42, n_train=20, D_val=201,
                       surface_name="paraboloid",
                       epochs_ae=500, epochs_sde=300,
                       outpath="nd_sweep_chart_quality"):
    """LOCA-style chart quality comparison: baseline vs K(Phat).

    Panel layout (2 rows × 3 cols):
      Row 1: T+F baseline
      Row 2: T+F+K
      Col 1: decoded surface colored by u (1st latent coord)
      Col 2: decoded surface colored by v (2nd latent coord)
      Col 3: pairwise distance scatter (true vs learned)
    """
    import torch
    import copy
    from src.numeric.losses import LossWeights
    from src.numeric.sde_nets import DriftNet, DiffusionNet
    from src.numeric.sde_training import SDEPipelineTrainer
    from src.numeric.training import MultiModelTrainer, TrainingConfig
    from src.numeric.highd_manifolds import (
        FourierAugmentedSurface,
        sample_from_highd_manifold,
    )
    from experiments.common import make_model_config
    from experiments.data_driven_sde import (
        TRAIN_BOUND, BATCH_SIZE, LR_AE, DEVICE,
    )
    from experiments.highd_N_D_sweep import (
        local_drift_fn, local_diffusion_fn,
        WARMUP_LW, FULL_LW, hidden_dims_for_D,
    )

    torch.manual_seed(seed)
    np.random.seed(seed)

    hdims = hidden_dims_for_D(D_val)
    surface = FourierAugmentedSurface(surface_name, D_val)
    batch_size = min(n_train, BATCH_SIZE)

    train_data = sample_from_highd_manifold(
        surface, local_drift_fn, local_diffusion_fn,
        [(-TRAIN_BOUND, TRAIN_BOUND), (-TRAIN_BOUND, TRAIN_BOUND)],
        n_samples=n_train, seed=seed, device=DEVICE,
    )
    x = train_data.samples.to(DEVICE)

    # Also sample a dense test grid for visualization
    n_vis = 400
    test_data = sample_from_highd_manifold(
        surface, local_drift_fn, local_diffusion_fn,
        [(-TRAIN_BOUND, TRAIN_BOUND), (-TRAIN_BOUND, TRAIN_BOUND)],
        n_samples=n_vis, seed=seed + 9999, device=DEVICE,
    )
    x_test = test_data.samples.to(DEVICE)

    # Recover the latent coords used for visualization
    # The samples are in the ambient space; we need the original (u,v) coords
    # sample_from_highd_manifold uses uniform sampling; reconstruct u,v
    torch.manual_seed(seed + 9999)
    np.random.seed(seed + 9999)
    bounds = [(-TRAIN_BOUND, TRAIN_BOUND), (-TRAIN_BOUND, TRAIN_BOUND)]
    uv_test = np.column_stack([
        np.random.uniform(bounds[i][0], bounds[i][1], n_vis)
        for i in range(2)
    ])
    uv_test_t = torch.tensor(uv_test, dtype=torch.float32, device=DEVICE)

    # Phase 1: T+F warmup
    phase1_epochs = max(1, epochs_ae // 2)

    trainer = MultiModelTrainer(TrainingConfig(
        epochs=epochs_ae, n_samples=n_train, input_dim=D_val, hidden_dim=hdims[0],
        latent_dim=2, learning_rate=LR_AE, batch_size=batch_size,
        test_size=0.03, print_interval=max(1, phase1_epochs // 5), device=DEVICE,
    ))
    mc = make_model_config("ae", WARMUP_LW, extrinsic_dim=D_val, hidden_dims=hdims)
    trainer.add_model(mc)
    loader = trainer.create_data_loader(train_data)

    for epoch in range(phase1_epochs):
        losses = trainer.train_epoch(loader)
        if (epoch + 1) % max(1, phase1_epochs // 5) == 0:
            print(f"  Phase1 Epoch {epoch+1}/{phase1_epochs}: loss={losses['ae']:.6f}")

    phase1_state = copy.deepcopy(trainer.models["ae"].state_dict())
    phase1_optim_state = copy.deepcopy(trainer.optimizers["ae"].state_dict())
    phase1_sched_state = copy.deepcopy(trainer.schedulers["ae"].state_dict())

    # Train both conditions
    models = {}
    phase2_epochs = epochs_ae - phase1_epochs

    for cond_label, phase2_lw in [("T+F (baseline)", None), ("T+F+K", FULL_LW)]:
        print(f"\n  Training condition: {cond_label}")
        t2 = MultiModelTrainer(TrainingConfig(
            epochs=epochs_ae, n_samples=n_train, input_dim=D_val, hidden_dim=hdims[0],
            latent_dim=2, learning_rate=LR_AE, batch_size=batch_size,
            test_size=0.03, print_interval=max(1, phase2_epochs // 5), device=DEVICE,
        ))
        mc2 = make_model_config("ae", WARMUP_LW, extrinsic_dim=D_val, hidden_dims=hdims)
        t2.add_model(mc2)
        t2._has_local_cov = True
        t2.models["ae"].load_state_dict(phase1_state)
        t2.optimizers["ae"].load_state_dict(phase1_optim_state)
        t2.schedulers["ae"].load_state_dict(phase1_sched_state)

        if phase2_lw is not None:
            warmup_frac = 0.2
            warmup_epochs = int(phase2_epochs * warmup_frac)
            for epoch in range(phase2_epochs):
                if epoch < warmup_epochs:
                    ramp = (epoch + 1) / warmup_epochs
                    lw_epoch = LossWeights(
                        tangent_bundle=phase2_lw.tangent_bundle,
                        diffeo=phase2_lw.diffeo,
                        curvature=phase2_lw.curvature * ramp,
                    )
                else:
                    lw_epoch = phase2_lw
                ep_losses = t2.train_epoch(loader, {mc2.name: lw_epoch})
                if (epoch + 1) % max(1, phase2_epochs // 5) == 0:
                    print(f"    Phase2 Epoch {epoch+1}/{phase2_epochs}: "
                          f"loss={ep_losses['ae']:.6f}")

        ae = t2.models["ae"]
        ae.eval()
        models[cond_label] = ae

    # ── Generate visualization data ──
    results = {}
    for cond_label, ae in models.items():
        with torch.no_grad():
            # Encode test points
            z = ae.encoder(x_test)
            # Decode back
            x_recon = ae.decoder(z)

        results[cond_label] = {
            "z": z.cpu().numpy(),
            "x_recon": x_recon.cpu().numpy(),
        }

    x_test_np = x_test.cpu().numpy()
    uv_np = uv_test

    # ── Plot ──
    fig = plt.figure(figsize=(11.0, 6.8))

    # Use gridspec: 2 rows × 4 cols (col 0 for row label, cols 1-3 for panels)
    gs = fig.add_gridspec(2, 4, width_ratios=[0.05, 1, 1, 1.1],
                          hspace=0.32, wspace=0.45)

    cond_labels = ["T+F (baseline)", "T+F+K"]
    row_colors = ["#555555", "#D64045"]
    scatter_refs = {}  # store scatter objects for shared colorbars

    for row, cond_label in enumerate(cond_labels):
        z = results[cond_label]["z"]
        x_recon = results[cond_label]["x_recon"]

        # Row label in column 0
        ax_label = fig.add_subplot(gs[row, 0])
        ax_label.axis("off")
        ax_label.text(0.5, 0.5, cond_label,
                      fontsize=12, fontweight="bold", color=row_colors[row],
                      rotation=90, ha="center", va="center",
                      transform=ax_label.transAxes)

        # Col 1: Latent space colored by u
        ax = fig.add_subplot(gs[row, 1])
        sc = ax.scatter(z[:, 0], z[:, 1], c=uv_np[:, 0], cmap="coolwarm",
                        s=18, alpha=0.85, edgecolors="0.3", linewidths=0.2)
        ax.set_xlabel(r"$z_1$", fontsize=9)
        ax.set_ylabel(r"$z_2$", fontsize=9)
        ax.set_aspect("equal")
        ax.tick_params(labelsize=8)
        for sp in ["top", "right"]:
            ax.spines[sp].set_visible(False)
        if row == 0:
            ax.set_title("Latent space (color $= u$)", fontsize=10, pad=6)
        scatter_refs[("u", row)] = (sc, ax)

        # Col 2: Latent space colored by v
        ax = fig.add_subplot(gs[row, 2])
        sc2 = ax.scatter(z[:, 0], z[:, 1], c=uv_np[:, 1], cmap="viridis",
                         s=18, alpha=0.85, edgecolors="0.3", linewidths=0.2)
        ax.set_xlabel(r"$z_1$", fontsize=9)
        ax.set_ylabel(r"$z_2$", fontsize=9)
        ax.set_aspect("equal")
        ax.tick_params(labelsize=8)
        for sp in ["top", "right"]:
            ax.spines[sp].set_visible(False)
        if row == 0:
            ax.set_title("Latent space (color $= v$)", fontsize=10, pad=6)
        scatter_refs[("v", row)] = (sc2, ax)

        # Col 3: Pairwise distance scatter
        ax = fig.add_subplot(gs[row, 3])

        # Subsample pairs for visibility
        n_pts = len(x_test_np)
        n_pairs = min(3000, n_pts * (n_pts - 1) // 2)
        rng = np.random.RandomState(42)
        idx_i = rng.randint(0, n_pts, n_pairs)
        idx_j = rng.randint(0, n_pts, n_pairs)
        mask = idx_i < idx_j
        idx_i, idx_j = idx_i[mask], idx_j[mask]

        # True ambient distances
        d_true = np.linalg.norm(x_test_np[idx_i] - x_test_np[idx_j], axis=1)
        # Learned latent distances
        d_latent = np.linalg.norm(z[idx_i] - z[idx_j], axis=1)

        ax.scatter(d_true, d_latent, s=4, alpha=0.25, color=row_colors[row],
                   edgecolors="none", rasterized=True)

        # Best-fit line
        dmax = max(d_true.max(), d_latent.max())
        slope, intercept = np.polyfit(d_true, d_latent, 1)
        d_fit = np.linspace(0, dmax, 100)
        ax.plot(d_fit, slope * d_fit + intercept, "k-", linewidth=1.2, alpha=0.7)

        # Correlation and reconstruction error
        r = np.corrcoef(d_true, d_latent)[0, 1]
        recon_err = np.sqrt(((x_recon - x_test_np) ** 2).mean())
        ax.text(0.05, 0.95,
                f"$r = {r:.3f}$\nslope $= {slope:.2f}$\nRMSE $= {recon_err:.3f}$",
                transform=ax.transAxes, fontsize=8.5,
                verticalalignment="top",
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                          edgecolor="0.7", alpha=0.9))

        ax.set_xlabel("Ambient distance $\\|x_i - x_j\\|$", fontsize=9)
        ax.set_ylabel("Latent distance $\\|z_i - z_j\\|$", fontsize=9)
        ax.tick_params(labelsize=8)
        for sp in ["top", "right"]:
            ax.spines[sp].set_visible(False)
        if row == 0:
            ax.set_title("Distance preservation", fontsize=10, pad=6)

    # Add horizontal colorbars below bottom-row latent panels
    # (must do after figure layout is finalized)
    fig.canvas.draw()
    for coord, label in [("u", "$u$"), ("v", "$v$")]:
        sc_obj, ax_ref = scatter_refs[(coord, 1)]
        # Use inset_axes for robust positioning
        from mpl_toolkits.axes_grid1.inset_locator import inset_axes
        cax = inset_axes(ax_ref, width="60%", height="5%",
                         loc="lower center", borderpad=-2.5)
        cb = fig.colorbar(sc_obj, cax=cax, orientation="horizontal")
        cb.set_label(label, fontsize=8, labelpad=1)
        cb.ax.tick_params(labelsize=7)

    fig.suptitle(
        f"{surface_name.replace('_', ' ').title()},  "
        f"$N={n_train}$,  $D={D_val}$ (codim. {D_val - 2})",
        fontsize=13, fontweight="bold", y=0.99,
    )

    for fmt in ["pdf", "png"]:
        fig.savefig(f"{outpath}.{fmt}", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {outpath}.pdf and {outpath}.png")


# ── Figure 3: Paired seed comparison ──────────────────────────────────────

def plot_paired_seeds(df, outpath="nd_sweep_paired"):
    """Show per-seed MTE and W2 for baseline vs K(Phat) at key (N,D) cells.

    Layout: 2 rows (paraboloid, hyp. paraboloid) × 2 cols (D=11 N=20, D=201 N=200)
    Each panel: paired dot plot with lines connecting same-seed values.
    """
    cells = [
        (11, 20, "$D=11,\\; N=20$"),
        (11, 200, "$D=11,\\; N=200$"),
        (201, 20, "$D=201,\\; N=20$"),
        (201, 200, "$D=201,\\; N=200$"),
    ]
    surfaces = ["paraboloid", "hyperbolic_paraboloid"]
    surface_labels = ["Paraboloid", "Hyp. paraboloid"]
    metric = "MTE@1.0"

    fig, axes = plt.subplots(2, 4, figsize=(12, 5.0), sharey="row")

    cond_colors = {"baseline": "#2176AE", "K(Phat)": "#D64045"}

    for row, (surface, surf_label) in enumerate(zip(surfaces, surface_labels)):
        for col, (D, N, cell_label) in enumerate(cells):
            ax = axes[row, col]

            base = df[(df["surface"] == surface) & (df["D"] == D) &
                       (df["N"] == N) & (df["condition"] == "baseline")
                      ].sort_values("seed")
            kphat = df[(df["surface"] == surface) & (df["D"] == D) &
                        (df["N"] == N) & (df["condition"] == "K(Phat)")
                       ].sort_values("seed")

            if len(base) == 0:
                ax.set_visible(False)
                continue

            b_vals = base[metric].values
            k_vals = kphat[metric].values
            seeds = np.arange(len(b_vals))

            # Paired lines
            for s in range(len(seeds)):
                ax.plot([0, 1], [b_vals[s], k_vals[s]],
                        color="0.7", linewidth=0.8, zorder=1)

            # Scatter
            ax.scatter(np.zeros(len(seeds)), b_vals,
                       color=cond_colors["baseline"], s=45, zorder=2,
                       edgecolors="white", linewidths=0.5, label="T+F")
            ax.scatter(np.ones(len(seeds)), k_vals,
                       color=cond_colors["K(Phat)"], s=45, zorder=2,
                       edgecolors="white", linewidths=0.5, label="T+F+K")

            # Mean bars
            for xp, vals, c in [(0, b_vals, cond_colors["baseline"]),
                                 (1, k_vals, cond_colors["K(Phat)"])]:
                ax.plot([xp - 0.15, xp + 0.15], [vals.mean(), vals.mean()],
                        color=c, linewidth=2.5, zorder=3)

            # p-value
            _, p = stats.ttest_rel(k_vals, b_vals)
            delta = ((k_vals - b_vals) / b_vals * 100).mean()
            star = sig_marker(p)
            if star:
                p_text = f"$\\Delta={delta:+.1f}\\%${star}"
            else:
                p_text = f"$\\Delta={delta:+.1f}\\%$ n.s."
            ax.text(0.5, 0.95, p_text,
                    transform=ax.transAxes, ha="center", va="top",
                    fontsize=8.5,
                    bbox=dict(boxstyle="round,pad=0.2", facecolor="white",
                              edgecolor="0.8", alpha=0.9))

            ax.set_xlim(-0.4, 1.4)
            ax.set_xticks([0, 1])
            ax.set_xticklabels(["T+F", "T+F+K"], fontsize=9)
            ax.tick_params(labelsize=8)

            if row == 0:
                ax.set_title(cell_label, fontsize=9.5)
            if col == 0:
                ax.set_ylabel(f"{surf_label}\nMTE @ $t=1$", fontsize=9)

            for sp in ["top", "right"]:
                ax.spines[sp].set_visible(False)

            if row == 0 and col == 0:
                ax.legend(fontsize=7, loc="lower left",
                          framealpha=0.9, edgecolor="0.7")

    fig.tight_layout(w_pad=0.8, h_pad=1.0)
    for fmt in ["pdf", "png"]:
        fig.savefig(f"{outpath}.{fmt}", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {outpath}.pdf and {outpath}.png")


def main():
    parser = argparse.ArgumentParser(description="Plot N×D sweep results")
    parser.add_argument("--fig1-only", action="store_true",
                        help="Only generate benefit line plot (from CSV)")
    parser.add_argument("--fig2-only", action="store_true",
                        help="Only generate LOCA-style chart quality (trains models)")
    parser.add_argument("--fig3-only", action="store_true",
                        help="Only generate paired seed comparison (from CSV)")
    parser.add_argument("--csv", default="highd_N_D_sweep.csv")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-train", type=int, default=20)
    parser.add_argument("--d-val", type=int, default=201)
    parser.add_argument("--surface", default="paraboloid")
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--outdir", default="Autoencoder-Paper")
    args = parser.parse_args()

    if not args.fig2_only and not args.fig3_only:
        df = pd.read_csv(args.csv)
        plot_benefit_figure(df, outpath=f"{args.outdir}/nd_sweep_benefit")

    if args.fig3_only or (not args.fig1_only and not args.fig2_only):
        df = pd.read_csv(args.csv)
        plot_paired_seeds(df, outpath=f"{args.outdir}/nd_sweep_paired")

    if args.fig2_only or (not args.fig1_only and not args.fig3_only):
        plot_chart_quality(
            seed=args.seed, n_train=args.n_train, D_val=args.d_val,
            surface_name=args.surface, epochs_ae=args.epochs,
            outpath=f"{args.outdir}/nd_sweep_chart_quality",
        )


if __name__ == "__main__":
    main()
