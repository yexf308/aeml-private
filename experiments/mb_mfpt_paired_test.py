"""
Smoke test: paired vs independent noise for MB MFPT.

Trains baseline + T+F on paraboloid D=11, 3 seeds.
Compares paired-noise MFPT error vs independent-noise MFPT error.
"""

import torch
import numpy as np
from scipy.stats import wilcoxon

from src.numeric.losses import LossWeights
from src.numeric.sde_training import SDEPipelineTrainer
from src.numeric.highd_manifolds import (
    FourierAugmentedSurface,
    create_highd_lambdified_sde,
    sample_from_highd_manifold,
)

from experiments.data_driven_sde import (
    simulate_ground_truth, BATCH_SIZE, LR_SDE, DEVICE, DT, BOUNDARY,
)
from experiments.mfpt_full_ablation import train_pipeline
from experiments.mb_dynamics import (
    mb_local_drift_fn, mb_local_diffusion_fn,
    train_ae_highd,
    WELLS_UV, TRAIN_BOUND,
    assign_wells_ambient, compute_interwell_mfpt,
)
from experiments.mb_mfpt import MB_DRIFT_HIDDEN, MB_SDE_EPOCHS

SURFACE = "paraboloid"
D = 11
N_TRAIN = 200
EPOCHS = 4000
N_TRAJ = 1000
LONG_T = 20.0
LONG_N_STEPS = int(LONG_T / DT)
SEEDS = [42, 1042, 2042]

CONDITIONS = {
    "baseline": LossWeights(),
    "T+F": LossWeights(tangent_bundle=1.0, diffeo=1.0),
}


def run_seed(seed):
    print(f"\n{'='*60}")
    print(f"  Seed {seed}")
    print(f"{'='*60}")

    surf = FourierAugmentedSurface(SURFACE, D)
    sde = create_highd_lambdified_sde(
        surf, mb_local_drift_fn, mb_local_diffusion_fn,
    )

    # Training data (DatasetBatch format expected by train_ae_highd)
    bounds = [(-TRAIN_BOUND, TRAIN_BOUND), (-TRAIN_BOUND, TRAIN_BOUND)]
    train_data = sample_from_highd_manifold(
        surf, mb_local_drift_fn, mb_local_diffusion_fn,
        bounds, N_TRAIN, seed=seed, device=DEVICE,
    )
    x_train = train_data.samples
    v_train = train_data.mu       # ambient drift
    Lam_train = train_data.cov    # ambient covariance

    # Wells in ambient
    wells_ambient = sde.chart(WELLS_UV.to(DEVICE))

    # Initial conditions at well 1
    torch.manual_seed(seed + 999)
    init_local = torch.randn(N_TRAJ, 2, device=DEVICE) * 0.1
    init_local[:, 0] += WELLS_UV[0, 0]
    init_local[:, 1] += WELLS_UV[0, 1]
    init_local.clamp_(-TRAIN_BOUND, TRAIN_BOUND)
    init_ambient = sde.chart(init_local).to(DEVICE)

    # SHARED noise for paired design
    torch.manual_seed(seed + 1234)
    dW_shared = torch.randn(N_TRAJ, LONG_N_STEPS, 2, device=DEVICE)

    # GT trajectories (using shared noise)
    gt_traj, gt_alive, _ = simulate_ground_truth(
        init_local, sde, LONG_N_STEPS, DT, dW_shared, BOUNDARY * 3,
    )
    gt_assign = assign_wells_ambient(gt_traj, wells_ambient)
    gt_mfpt_info = compute_interwell_mfpt(gt_assign, start_well=0, dt=DT)

    results = {}
    for cond_label, lw in CONDITIONS.items():
        print(f"\n  --- {cond_label} ---")
        torch.manual_seed(seed)
        np.random.seed(seed)

        ae, recon = train_ae_highd(
            surf, D, seed, N_TRAIN, EPOCHS, lw, train_data,
        )
        print(f"    recon={recon:.6f}")

        pipeline = train_pipeline(
            ae, x_train, v_train, Lam_train, seed, N_TRAIN, MB_SDE_EPOCHS,
            drift_hidden=MB_DRIFT_HIDDEN,
        )

        with torch.no_grad():
            z0 = ae.encoder(init_ambient)

        # PAIRED: same dW as GT
        _, x_traj_paired = pipeline.simulate(
            z0, LONG_N_STEPS, DT, dW=dW_shared, boundary=BOUNDARY * 3,
        )
        lr_assign_paired = assign_wells_ambient(x_traj_paired, wells_ambient)
        lr_mfpt_paired = compute_interwell_mfpt(lr_assign_paired, start_well=0, dt=DT)

        # INDEPENDENT: different dW
        torch.manual_seed(seed + 5678)
        dW_indep = torch.randn(N_TRAJ, LONG_N_STEPS, 2, device=DEVICE)
        _, x_traj_indep = pipeline.simulate(
            z0, LONG_N_STEPS, DT, dW=dW_indep, boundary=BOUNDARY * 3,
        )
        lr_assign_indep = assign_wells_ambient(x_traj_indep, wells_ambient)
        lr_mfpt_indep = compute_interwell_mfpt(lr_assign_indep, start_well=0, dt=DT)

        gt_m = gt_mfpt_info['mean']
        paired_m = lr_mfpt_paired['mean']
        indep_m = lr_mfpt_indep['mean']

        paired_err = abs(paired_m - gt_m) / gt_m if gt_m > 0 else float('nan')
        indep_err = abs(indep_m - gt_m) / gt_m if gt_m > 0 else float('nan')

        print(f"    GT MFPT={gt_m:.3f}")
        print(f"    Paired:  learned={paired_m:.3f}  err={paired_err:.1%}")
        print(f"    Indep:   learned={indep_m:.3f}  err={indep_err:.1%}")

        results[cond_label] = {
            'paired_err': paired_err,
            'indep_err': indep_err,
            'paired_mfpt': paired_m,
            'indep_mfpt': indep_m,
        }

    return gt_mfpt_info['mean'], results


if __name__ == "__main__":
    all_results = []
    for seed in SEEDS:
        gt_m, res = run_seed(seed)
        all_results.append((gt_m, res))

    print(f"\n{'='*60}")
    print("  SUMMARY")
    print(f"{'='*60}")

    for cond in CONDITIONS:
        paired_errs = [r[1][cond]['paired_err'] for r in all_results]
        indep_errs = [r[1][cond]['indep_err'] for r in all_results]
        print(f"\n  {cond}:")
        print(f"    Paired errors:  {[f'{e:.1%}' for e in paired_errs]}  mean={np.mean(paired_errs):.1%}")
        print(f"    Indep  errors:  {[f'{e:.1%}' for e in indep_errs]}  mean={np.mean(indep_errs):.1%}")

    # Compare baseline vs T+F paired errors
    bl_paired = [r[1]['baseline']['paired_err'] for r in all_results]
    tf_paired = [r[1]['T+F']['paired_err'] for r in all_results]
    bl_indep = [r[1]['baseline']['indep_err'] for r in all_results]
    tf_indep = [r[1]['T+F']['indep_err'] for r in all_results]

    print(f"\n  baseline vs T+F (paired):  {[f'{e:.1%}' for e in bl_paired]} vs {[f'{e:.1%}' for e in tf_paired]}")
    print(f"  baseline vs T+F (indep):   {[f'{e:.1%}' for e in bl_indep]} vs {[f'{e:.1%}' for e in tf_indep]}")
