"""
Diagnostic: why do better AE coefficients (E_mu) produce worse trajectories (W2)?

Three diagnostics (per GPT-5.4 recommendation):
  1. Actual trained drift/diffusion error at ROLLOUT states z(t), not just static eval points
  2. Decoder Jacobian singular values (condition number, max stretch)
  3. Re-run rollout with smaller dt to test EM discretization error hypothesis

Design: D=11, N=50, paraboloid, 5 seeds, 2 conditions (no_penalty, T+F).

Usage:
    # Smoke test
    python -m experiments.trajectory_paradox_diagnostic --n-seeds 1 --epochs 50 --sde-epochs 50

    # Full run
    python -m experiments.trajectory_paradox_diagnostic --n-seeds 5
"""

import argparse
import copy
import time
import torch
import numpy as np
import pandas as pd
from scipy import stats

from src.numeric.losses import LossWeights
from src.numeric.sde_nets import DriftNet, DiffusionNet
from src.numeric.sde_training import SDEPipelineTrainer
from src.numeric.training import MultiModelTrainer, TrainingConfig
from src.numeric.geometry import regularized_metric_inverse, ambient_quadratic_variation_drift

from src.numeric.highd_manifolds import (
    FourierAugmentedSurface,
    sample_from_highd_manifold,
    create_highd_lambdified_sde,
)

from experiments.common import make_model_config
from experiments.data_driven_sde import (
    evaluate_pipeline, simulate_ground_truth,
    TRAIN_BOUND, BATCH_SIZE, LR_AE, LR_SDE, DEVICE,
    N_TRAJ, N_STEPS, DT, BOUNDARY, STOP_BOUND,
)
from experiments.highd_N_D_sweep import (
    hidden_dims_for_D, local_drift_fn, local_diffusion_fn,
    compute_coefficient_errors, _run_sde_stages, N_EVAL,
)
from experiments.highd_baseline_comparison import _train_ae

# ── Constants ─────────────────────────────────────────────────────────────

NO_PENALTY_LW = LossWeights()
TF_LW = LossWeights(tangent_bundle=1.0, diffeo=1.0)
LAMBDA_SMOOTH = 0.5
DT_VALUES = [0.01, 0.005, 0.002]  # default dt, half, fifth


# ── Diagnostic: actual drift error at rollout states ──────────────────────

def compute_actual_drift_error_at_points(ae, drift_net, sde, surface, z_points, device):
    """Compute actual TRAINED drift_net error at given latent points.

    Returns ambient drift error: ||Dφ·drift_net(z) + ½q - b_true||²
    and latent drift error:      ||drift_net(z) - b_z_oracle||²

    b_z_oracle = g^{-1} Dφ^T (b - ½q)  (decoder-side oracle)
    """
    ae.eval()
    drift_net.eval()

    with torch.no_grad():
        x_decoded = ae.decoder(z_points)
        b_pred = drift_net(z_points)  # (N, d)

    N, d = z_points.shape

    # Decoder Jacobian and Hessian at z_points
    dphi = torch.func.vmap(torch.func.jacrev(ae.decoder))(z_points)  # (N, D, d)

    def decoder_hessian(z_single):
        return torch.func.jacrev(torch.func.jacrev(ae.decoder))(z_single)
    d2phi = torch.func.vmap(decoder_hessian)(z_points)  # (N, D, d, d)

    dphi = dphi.detach()
    d2phi = d2phi.detach()

    with torch.no_grad():
        # Reconstruct ambient drift from drift_net prediction
        b_ambient_pred = (dphi @ b_pred.unsqueeze(-1)).squeeze(-1)  # (N, D)

        # Quadratic variation correction q
        dphi_T = dphi.transpose(-1, -2)
        g = torch.bmm(dphi_T, dphi)
        g_inv = regularized_metric_inverse(g)
        Sigma_z = torch.bmm(g_inv, torch.bmm(dphi_T, torch.bmm(
            torch.zeros_like(g).unsqueeze(1).expand(-1, dphi.shape[1], -1, -1),  # placeholder
            torch.bmm(dphi, g_inv)
        )))

    # We need the TRUE ambient drift at the decoded points.
    # Encode decoded points back to local coords to query the SDE
    with torch.no_grad():
        z_re = ae.encoder(x_decoded)  # re-encode to get local coords

    # We can't easily get true b at arbitrary decoded points because sde.ambient_drift
    # needs local (u,v) coords. Instead, compute the ambient drift reconstruction error
    # using the encoder-pullback formula.

    # Encoder Jacobian at x_decoded
    dpi = torch.func.vmap(torch.func.jacrev(ae.encoder))(x_decoded.detach()).detach()  # (N, d, D)

    with torch.no_grad():
        # Ambient drift predicted: Dφ · drift_net(z)
        b_ambient_from_net = (dphi @ b_pred.unsqueeze(-1)).squeeze(-1)

        # Quadratic variation from drift_net prediction
        q_from_d2phi = ambient_quadratic_variation_drift(
            torch.eye(d, device=device).unsqueeze(0).expand(N, -1, -1),  # identity Sigma placeholder
            d2phi,
        )

    return dphi, dpi, d2phi


def compute_drift_errors_along_trajectory(
    ae, drift_net, diffusion_net, sde, surface, seed, device,
):
    """Simulate trajectory and compute actual drift errors at trajectory states.

    Returns dict with:
      - ambient_drift_err: ||Dφ·b_net(z) + ½q_net - b_true(x)||  at each step
      - decoder_jac_svd: singular values of Dφ at each step
    """
    ae.eval()
    drift_net.eval()
    diffusion_net.eval()

    pipeline = SDEPipelineTrainer(ae, drift_net, diffusion_net, device=device)

    # Sample trajectory init points (same as evaluate_pipeline)
    torch.manual_seed(seed + 999)
    n_sample = min(N_TRAJ, 100)  # fewer for diagnostics
    init_local = (torch.rand(n_sample, 2, device=device) * 2 - 1) * TRAIN_BOUND
    init_ambient = sde.chart(init_local).to(device)

    torch.manual_seed(seed + 1234)
    dW = torch.randn(n_sample, N_STEPS, 2, device=device)

    # Ground truth trajectories
    gt_traj, gt_alive, gt_local = simulate_ground_truth(
        init_local, sde, N_STEPS, DT, dW, BOUNDARY,
    )

    # Simulate learned trajectory
    with torch.no_grad():
        z0 = ae.encoder(init_ambient)
    z_traj, x_traj = pipeline.simulate(z0, N_STEPS, DT, dW=dW)

    # Compute drift errors and decoder Jacobian at selected timesteps
    check_steps = [0, 10, 25, 50, 75, 99]  # t=0, 0.1, 0.25, 0.5, 0.75, 1.0
    results = []

    for step in check_steps:
        t_val = step * DT
        z_t = z_traj[:, step].detach()  # (B, d)
        x_t = x_traj[:, step].detach()  # (B, D)

        alive = gt_alive[:, step] if step < gt_alive.shape[1] else gt_alive[:, -1]
        n_alive = alive.sum().item()
        if n_alive < 5:
            continue

        z_t = z_t[alive]
        x_t_gt = gt_traj[:, step][alive]

        N_pts, d = z_t.shape

        # Drift prediction
        with torch.no_grad():
            b_pred = drift_net(z_t)  # (N, d)

        # Decoder Jacobian
        dphi = torch.func.vmap(torch.func.jacrev(ae.decoder))(z_t).detach()  # (N, D, d)

        # Decoder Hessian
        def dec_hess(z_s):
            return torch.func.jacrev(torch.func.jacrev(ae.decoder))(z_s)
        d2phi = torch.func.vmap(dec_hess)(z_t).detach()  # (N, D, d, d)

        with torch.no_grad():
            dphi_T = dphi.transpose(-1, -2)
            g = torch.bmm(dphi_T, dphi)
            g_inv = regularized_metric_inverse(g)

            # Predicted ambient drift: Dφ · b_net(z)
            b_ambient_pred = (dphi @ b_pred.unsqueeze(-1)).squeeze(-1)

            # Quadratic variation from LEARNED diffusion
            sigma_z = diffusion_net(z_t)  # (N, d, d)
            Sigma_z = torch.bmm(sigma_z, sigma_z.transpose(-1, -2))
            q_pred = ambient_quadratic_variation_drift(Sigma_z, d2phi)

            # Full ambient drift reconstruction
            full_b_pred = b_ambient_pred + 0.5 * q_pred

            # True ambient drift at gt positions
            local_gt = gt_local[:, step][alive]  # (N, 2)

        # ambient_drift needs grad for surface.jacobian — compute outside no_grad
        local_gt_g = local_gt.detach().requires_grad_(True)
        b_true = sde.ambient_drift(local_gt_g).detach()

        with torch.no_grad():

            # Ambient drift error
            drift_err = ((full_b_pred - b_true) ** 2).sum(-1)  # (N,)

            # Singular values of decoder Jacobian
            svd_vals = torch.linalg.svdvals(dphi)  # (N, min(D,d))

            # Trajectory displacement: how far has z drifted from z0?
            z0_alive = z_traj[:, 0][alive]
            z_displacement = ((z_t - z0_alive) ** 2).sum(-1).sqrt()

            # Ambient position error (learned vs gt)
            x_learned = x_traj[:, step][alive]
            ambient_pos_err = ((x_learned - x_t_gt) ** 2).sum(-1)

        results.append({
            "step": step,
            "t": t_val,
            "n_alive": n_alive,
            "drift_err_mean": drift_err.mean().item(),
            "drift_err_median": drift_err.median().item(),
            "svd_max_mean": svd_vals[:, 0].mean().item(),
            "svd_min_mean": svd_vals[:, -1].mean().item(),
            "cond_number_mean": (svd_vals[:, 0] / svd_vals[:, -1].clamp(min=1e-8)).mean().item(),
            "cond_number_max": (svd_vals[:, 0] / svd_vals[:, -1].clamp(min=1e-8)).max().item(),
            "z_displacement_mean": z_displacement.mean().item(),
            "ambient_pos_err_mean": ambient_pos_err.mean().item(),
        })

    return results


def run_with_varying_dt(
    ae, drift_net, diffusion_net, sde, seed, device, dt_values,
):
    """Re-run trajectory simulation with different dt values.

    Returns dict: dt -> W2@1.0
    """
    from experiments.data_driven_sde import compute_w2, compute_mte_at_step

    ae.eval()
    drift_net.eval()
    diffusion_net.eval()

    pipeline = SDEPipelineTrainer(ae, drift_net, diffusion_net, device=device)

    results = {}
    for dt in dt_values:
        n_steps = int(round(1.0 / dt))

        torch.manual_seed(seed + 999)
        init_local = (torch.rand(N_TRAJ, 2, device=device) * 2 - 1) * TRAIN_BOUND
        init_ambient = sde.chart(init_local).to(device)

        torch.manual_seed(seed + 1234)
        dW = torch.randn(N_TRAJ, n_steps, 2, device=device)

        # GT trajectory with this dt
        gt_traj, gt_alive, _ = simulate_ground_truth(
            init_local, sde, n_steps, dt, dW, BOUNDARY,
        )

        with torch.no_grad():
            z0 = ae.encoder(init_ambient)
        _, learned_traj = pipeline.simulate(z0, n_steps, dt, dW=dW)

        both_alive = gt_alive[:, -1]
        w2 = compute_w2(
            learned_traj[:, -1], gt_traj[:, -1],
            both_alive, both_alive,
        )

        step_mid = n_steps // 2
        both_mid = gt_alive[:, step_mid]
        mte_mid = compute_mte_at_step(learned_traj, gt_traj, both_mid, step_mid)

        results[dt] = {"W2@1.0": w2, "MTE@0.5": mte_mid, "n_steps": n_steps}
        print(f"    dt={dt:.4f} (n_steps={n_steps}): W2={w2:.4f}, MTE@0.5={mte_mid:.4f}")

    return results


# ── Main ──────────────────────────────────────────────────────────────────

def run_diagnostic(D, N, surface_name, seed, epochs_ae, epochs_sde):
    """Run full diagnostic for one (surface, D, N, seed)."""
    torch.manual_seed(seed)
    np.random.seed(seed)

    surface = FourierAugmentedSurface(surface_name, D)
    train_data = sample_from_highd_manifold(
        surface, local_drift_fn, local_diffusion_fn,
        [(-TRAIN_BOUND, TRAIN_BOUND), (-TRAIN_BOUND, TRAIN_BOUND)],
        n_samples=N, seed=seed, device=DEVICE,
    )
    x = train_data.samples.to(DEVICE)
    v = train_data.mu.to(DEVICE)
    Lambda = train_data.cov.to(DEVICE)

    sde = create_highd_lambdified_sde(surface, local_drift_fn, local_diffusion_fn)

    all_results = {}

    for cond_label, lw in [("no_penalty", NO_PENALTY_LW), ("T+F", TF_LW)]:
        print(f"\n  === Condition: {cond_label} ===")
        torch.manual_seed(seed)
        np.random.seed(seed)

        ae, recon_per_dim = _train_ae(surface, D, seed, N, epochs_ae, lw, train_data)
        print(f"    recon_per_dim={recon_per_dim:.6f}")

        # Static E_mu (the existing metric)
        torch.manual_seed(seed + 5000)
        eval_uv = (torch.rand(N_EVAL, 2, device=DEVICE) * 2 - 1) * TRAIN_BOUND
        e_mu, e_sigma = compute_coefficient_errors(ae, sde, surface, eval_uv, DEVICE)
        print(f"    Static E_mu={e_mu:.4f}  E_Sigma={e_sigma:.4f}")

        # SDE stages
        sde_results = _run_sde_stages(
            ae, x, v, Lambda, sde, surface, seed, N, epochs_sde, LAMBDA_SMOOTH,
        )
        # Extract the trained nets from the pipeline
        d = 2
        drift_net = DriftNet(d).to(DEVICE)
        diffusion_net = DiffusionNet(d).to(DEVICE)

        # Re-train to get the actual net objects (since _run_sde_stages doesn't return them)
        # We need to reconstruct the pipeline. Let's re-do stages to keep the nets.
        torch.manual_seed(seed + 100)
        drift_net = DriftNet(d).to(DEVICE)
        diffusion_net = DiffusionNet(d).to(DEVICE)
        pipeline = SDEPipelineTrainer(ae, drift_net, diffusion_net, device=DEVICE)

        z_pre, dphi_pre, d2phi_pre = pipeline.precompute_decoder_derivatives(x)
        print("    Precomputing encoder-pullback target...")
        b_z_target, g = pipeline.precompute_enc_pull_target(x, v, Lambda, dphi_pre)

        torch.manual_seed(seed + 100)
        drift_net2 = DriftNet(d).to(DEVICE)
        pipeline2 = SDEPipelineTrainer(ae, drift_net2, DiffusionNet(d).to(DEVICE), device=DEVICE)
        pipeline2.train_stage2_regression(
            z_pre, b_z_target, g,
            epochs=epochs_sde, lr=LR_SDE, batch_size=min(N, BATCH_SIZE),
            print_interval=max(1, epochs_sde // 3),
            lambda_smooth=LAMBDA_SMOOTH, aug_sigma=0.1,
        )

        # Stage 3
        torch.manual_seed(seed + 200)
        diffusion_net2 = DiffusionNet(d).to(DEVICE)
        pipeline3 = SDEPipelineTrainer(ae, drift_net2, diffusion_net2, device=DEVICE)

        # Precompute diffusion target
        with torch.no_grad():
            Lambda_latent = torch.bmm(dphi_pre.transpose(-1, -2),
                                       torch.bmm(Lambda.to(DEVICE), dphi_pre))
            g_inv = regularized_metric_inverse(g)
            Sigma_target = torch.bmm(g_inv, torch.bmm(Lambda_latent, g_inv))

        pipeline3.train_stage3_precomputed(
            z_pre, dphi_pre, Lambda.to(DEVICE),
            epochs=epochs_sde, lr=LR_SDE, batch_size=min(N, BATCH_SIZE),
            print_interval=max(1, epochs_sde // 3),
        )

        # DIAGNOSTIC 1: Drift error along trajectory
        print(f"\n    --- Diagnostic 1: Drift error along trajectory ---")
        traj_diag = compute_drift_errors_along_trajectory(
            ae, drift_net2, diffusion_net2, sde, surface, seed, DEVICE,
        )
        for r in traj_diag:
            print(f"    t={r['t']:.2f}: drift_err={r['drift_err_mean']:.4f}, "
                  f"cond#={r['cond_number_mean']:.1f} (max={r['cond_number_max']:.1f}), "
                  f"svd_max={r['svd_max_mean']:.3f}, svd_min={r['svd_min_mean']:.3f}, "
                  f"z_disp={r['z_displacement_mean']:.3f}, "
                  f"amb_err={r['ambient_pos_err_mean']:.4f}")

        # DIAGNOSTIC 2: Decoder Jacobian at static eval points
        print(f"\n    --- Diagnostic 2: Decoder Jacobian at eval points ---")
        with torch.no_grad():
            x_eval = sde.chart(eval_uv)
            z_eval = ae.encoder(x_eval)

        dphi_eval = torch.func.vmap(torch.func.jacrev(ae.decoder))(z_eval).detach()
        svd_eval = torch.linalg.svdvals(dphi_eval)
        cond_eval = svd_eval[:, 0] / svd_eval[:, -1].clamp(min=1e-8)
        print(f"    svd_max: {svd_eval[:, 0].mean():.3f}±{svd_eval[:, 0].std():.3f}")
        print(f"    svd_min: {svd_eval[:, -1].mean():.3f}±{svd_eval[:, -1].std():.3f}")
        print(f"    cond#:   {cond_eval.mean():.1f}±{cond_eval.std():.1f} "
              f"(max={cond_eval.max():.1f})")

        # DIAGNOSTIC 3: Varying dt
        print(f"\n    --- Diagnostic 3: Varying dt ---")
        dt_results = run_with_varying_dt(
            ae, drift_net2, diffusion_net2, sde, seed, DEVICE, DT_VALUES,
        )

        all_results[cond_label] = {
            "E_mu": e_mu,
            "E_Sigma": e_sigma,
            "recon_per_dim": recon_per_dim,
            "traj_diag": traj_diag,
            "svd_max": svd_eval[:, 0].mean().item(),
            "svd_min": svd_eval[:, -1].mean().item(),
            "cond_mean": cond_eval.mean().item(),
            "cond_max": cond_eval.max().item(),
            "dt_results": dt_results,
            "drift_loss": sde_results["drift_loss"],
            "W2@1.0": sde_results.get("W2@1.0", float("nan")),
            "MTE@1.0": sde_results.get("MTE@1.0", float("nan")),
        }

    return all_results


def main():
    parser = argparse.ArgumentParser(description="Trajectory paradox diagnostic")
    parser.add_argument("--n-seeds", type=int, default=5)
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--sde-epochs", type=int, default=300)
    parser.add_argument("--base-seed", type=int, default=42)
    parser.add_argument("--D", type=int, default=11)
    parser.add_argument("--N", type=int, default=50)
    parser.add_argument("--surface", type=str, default="paraboloid")
    args = parser.parse_args()

    seeds = [args.base_seed + i * 1000 for i in range(args.n_seeds)]
    print(f"Device: {DEVICE}")
    print(f"D={args.D}, N={args.N}, surface={args.surface}")
    print(f"Seeds: {seeds}")
    print(f"dt values: {DT_VALUES}")
    print()

    all_seed_results = []

    for seed in seeds:
        print(f"\n{'='*70}")
        print(f"  SEED {seed}")
        print(f"{'='*70}")

        results = run_diagnostic(
            args.D, args.N, args.surface, seed, args.epochs, args.sde_epochs,
        )
        all_seed_results.append(results)

    # ── Summary ───────────────────────────────────────────────────────────
    print(f"\n\n{'='*80}")
    print("SUMMARY")
    print(f"{'='*80}")

    for cond in ["no_penalty", "T+F"]:
        print(f"\n  --- {cond} ---")
        e_mus = [r[cond]["E_mu"] for r in all_seed_results]
        w2s = [r[cond]["W2@1.0"] for r in all_seed_results]
        conds_mean = [r[cond]["cond_mean"] for r in all_seed_results]
        conds_max = [r[cond]["cond_max"] for r in all_seed_results]
        svd_maxs = [r[cond]["svd_max"] for r in all_seed_results]
        svd_mins = [r[cond]["svd_min"] for r in all_seed_results]
        print(f"    E_mu:       {np.mean(e_mus):.4f}±{np.std(e_mus):.4f}")
        print(f"    W2@1.0:     {np.mean(w2s):.4f}±{np.std(w2s):.4f}")
        print(f"    cond#(mean): {np.mean(conds_mean):.1f}±{np.std(conds_mean):.1f}")
        print(f"    cond#(max):  {np.mean(conds_max):.1f}±{np.std(conds_max):.1f}")
        print(f"    svd_max:    {np.mean(svd_maxs):.3f}±{np.std(svd_maxs):.3f}")
        print(f"    svd_min:    {np.mean(svd_mins):.3f}±{np.std(svd_mins):.3f}")

        # dt sensitivity
        print(f"    dt sensitivity:")
        for dt in DT_VALUES:
            w2_at_dt = [r[cond]["dt_results"][dt]["W2@1.0"] for r in all_seed_results]
            mte_at_dt = [r[cond]["dt_results"][dt]["MTE@0.5"] for r in all_seed_results]
            print(f"      dt={dt:.4f}: W2={np.mean(w2_at_dt):.4f}±{np.std(w2_at_dt):.4f}, "
                  f"MTE@0.5={np.mean(mte_at_dt):.4f}±{np.std(mte_at_dt):.4f}")

    # Drift error growth comparison
    print(f"\n  --- Drift error along trajectory ---")
    for cond in ["no_penalty", "T+F"]:
        print(f"\n    {cond}:")
        # Average across seeds
        n_seeds = len(all_seed_results)
        all_steps = set()
        for r in all_seed_results:
            for d in r[cond]["traj_diag"]:
                all_steps.add(d["step"])
        for step in sorted(all_steps):
            vals = []
            cond_vals = []
            amb_vals = []
            for r in all_seed_results:
                for d in r[cond]["traj_diag"]:
                    if d["step"] == step:
                        vals.append(d["drift_err_mean"])
                        cond_vals.append(d["cond_number_mean"])
                        amb_vals.append(d["ambient_pos_err_mean"])
            if vals:
                print(f"      t={step*DT:.2f}: drift_err={np.mean(vals):.4f}±{np.std(vals):.4f}, "
                      f"cond#={np.mean(cond_vals):.1f}, "
                      f"amb_err={np.mean(amb_vals):.4f}±{np.std(amb_vals):.4f}")


if __name__ == "__main__":
    main()
