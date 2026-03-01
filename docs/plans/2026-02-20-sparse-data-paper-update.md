# Sparse-Data Paper Update — Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Re-run all paper experiments with N=20 (sparse data), two-phase AE training, 10-seed statistical rigor, and rewrite all experiment sections of `Autoencoder-Paper/paper.tex`.

**Architecture:** Create a unified multi-seed experiment runner `experiments/paper_experiments.py` that handles ablation, extrapolation, and dynamics extrapolation studies. Reuse existing `experiments/multiseed_K_study.py` for trajectory fidelity. All studies use N=20 training points and two-phase AE training when K is active. Results saved to CSV, then transcribed into LaTeX tables.

**Tech Stack:** PyTorch, scipy.stats (paired t-test), pandas, numpy, sympy, LaTeX

---

## Key Files Reference

- `src/numeric/training.py` — `MultiModelTrainer`, `TrainingConfig`, `TrainingPhase`, `train_with_schedule()`
- `src/numeric/losses.py` — `LossWeights`, `autoencoder_loss()`
- `src/numeric/performance_stats.py` — `compute_losses_per_sample()`
- `src/numeric/geometry.py` — `regularized_metric_inverse()`
- `src/numeric/datagen.py` — `sample_from_manifold()`
- `experiments/common.py` — `SURFACE_MAP`, `PENALTY_CONFIGS`, `make_model_config()`, `sample_ring_region()`, `create_test_datasets()`
- `experiments/multiseed_K_study.py` — Two-phase AE pattern, `paired_ttest()`, `run_single_seed()`
- `experiments/ablation_study.py` — `run_single_ablation()`, `PENALTY_CONFIGS_CURATED`
- `experiments/extrapolation_study.py` — `run_extrapolation_study()`, `evaluate_reconstruction()`
- `experiments/dynamics_extrapolation_study.py` — `compute_dynamics_metrics()`, `run_dynamics_extrapolation_study()`
- `experiments/trajectory_fidelity_study.py` — Simulation modes, metrics
- `Autoencoder-Paper/paper.tex` — Main paper (sections 7.1–8)
- `multiseed_K_study.csv` — Existing 10-seed trajectory fidelity results

## Two-Phase AE Training Pattern (copy from multiseed_K_study.py)

Every training function must implement this when a config has K active:

```python
from src.numeric.training import TrainingPhase

has_K = lw.curvature > 0
warmup_lw = LossWeights(tangent_bundle=lw.tangent_bundle, diffeo=lw.diffeo)

if has_K:
    phase1_epochs = epochs // 2
    phase2_epochs = epochs - phase1_epochs
    schedule = [
        TrainingPhase(epochs=phase1_epochs, loss_weights=warmup_lw, name="T+F-warmup"),
        TrainingPhase(epochs=phase2_epochs, loss_weights=lw, name="T+F+K-finetune"),
    ]
    trainer.train_with_schedule(loader, model_name, schedule, print_interval=max(1, epochs // 5))
else:
    for epoch in range(epochs):
        losses = trainer.train_epoch(loader)
        if (epoch + 1) % max(1, epochs // 5) == 0:
            print(f"  Epoch {epoch+1}: loss={losses[model_name]:.6f}")
```

Note: `warmup_lw` strips K from the loss weights. This means configs like T+K get warmup with T only, T+F+K gets warmup with T+F, K-only gets warmup with nothing (baseline). For K-only and F+K, `warmup_lw` should preserve whatever non-K weights exist.

More precisely:
```python
warmup_kwargs = {k: v for k, v in asdict(lw).items() if k != 'curvature' and v > 0}
warmup_lw = LossWeights(**warmup_kwargs)
```

---

### Task 1: Create `experiments/paper_experiments.py` — Ablation Study Runner

**Files:**
- Create: `experiments/paper_experiments.py`

**Step 1: Write the ablation study multi-seed runner**

This function trains one AE per (surface, penalty_config, seed), evaluates on a test set (sampled separately with seed+10000), and returns a dict of metrics.

```python
"""
Multi-seed paper experiments: ablation, extrapolation, dynamics extrapolation.

All studies use N=20, two-phase AE training, 10 seeds.

Usage:
    python -m experiments.paper_experiments ablation --n-seeds 10
    python -m experiments.paper_experiments extrapolation --n-seeds 10
    python -m experiments.paper_experiments dynamics --n-seeds 10
"""

import argparse
import time
from dataclasses import asdict

import numpy as np
import pandas as pd
import torch

from src.numeric.datagen import sample_from_manifold
from src.numeric.losses import LossWeights
from src.numeric.performance_stats import compute_losses_per_sample
from src.numeric.training import (
    ModelConfig, MultiModelTrainer, TrainingConfig, TrainingPhase,
)
from src.symbolic.manifold_sdes import ManifoldSDE
from src.symbolic.riemannian import RiemannianManifold
from src.symbolic.surfaces import surface
import sympy as sp

from experiments.common import (
    SURFACE_MAP, PENALTY_CONFIGS, make_model_config,
    sample_ring_region, create_test_datasets,
)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
N_TRAIN = 20
TRAIN_BOUND = 1.0
EPOCHS_AE = 500
BATCH_SIZE = 32
LR = 0.005

# Curated ablation configs (8 configs matching paper Table 1)
ABLATION_CONFIGS = {
    "baseline": LossWeights(),
    "T": LossWeights(tangent_bundle=1.0),
    "K": LossWeights(curvature=0.1),
    "F": LossWeights(diffeo=1.0),
    "T+K": LossWeights(tangent_bundle=1.0, curvature=0.1),
    "T+F": LossWeights(tangent_bundle=1.0, diffeo=1.0),
    "F+K": LossWeights(diffeo=1.0, curvature=0.1),
    "T+F+K": LossWeights(tangent_bundle=1.0, diffeo=1.0, curvature=0.1),
}


def create_manifold_sde(surface_name: str, with_dynamics: bool = False) -> ManifoldSDE:
    """Create manifold SDE for given surface."""
    u, v = sp.symbols("u v", real=True)
    local_coord, chart = surface(SURFACE_MAP[surface_name], u, v)
    manifold = RiemannianManifold(local_coord, chart)

    if with_dynamics:
        local_drift = sp.Matrix([-v, u])
        local_diffusion = sp.Matrix([[1 + u**2/4, u + v], [0, 1 + v**2/4]])
    else:
        local_drift = None
        local_diffusion = None

    return ManifoldSDE(manifold, local_drift=local_drift, local_diffusion=local_diffusion)


def train_ae_with_twophase(
    train_data, model_name, lw, epochs=EPOCHS_AE, hidden_dims=None
):
    """Train AE with two-phase schedule when K is active.

    Returns trained model.
    """
    if hidden_dims is None:
        hidden_dims = [64, 64]

    has_K = lw.curvature > 0
    # Build warmup weights: same as lw but without curvature
    warmup_kwargs = {k: v for k, v in asdict(lw).items() if k != 'curvature' and v > 0}
    warmup_lw = LossWeights(**warmup_kwargs)

    trainer = MultiModelTrainer(TrainingConfig(
        epochs=epochs, n_samples=N_TRAIN, input_dim=3, hidden_dim=64,
        latent_dim=2, learning_rate=LR, batch_size=BATCH_SIZE,
        test_size=0.03, print_interval=max(1, epochs // 5), device=DEVICE,
    ))
    trainer.add_model(make_model_config(
        model_name, warmup_lw if has_K else lw, hidden_dims=hidden_dims
    ))
    loader = trainer.create_data_loader(train_data)

    if has_K:
        phase1 = epochs // 2
        phase2 = epochs - phase1
        schedule = [
            TrainingPhase(epochs=phase1, loss_weights=warmup_lw, name="warmup"),
            TrainingPhase(epochs=phase2, loss_weights=lw, name="finetune"),
        ]
        trainer.train_with_schedule(
            loader, model_name, schedule, print_interval=max(1, epochs // 5)
        )
    else:
        for epoch in range(epochs):
            losses = trainer.train_epoch(loader)
            if (epoch + 1) % max(1, epochs // 5) == 0:
                print(f"      AE Epoch {epoch+1}: loss={losses[model_name]:.6f}")

    model = trainer.models[model_name]
    model.eval()
    return model


def paired_ttest(vals_a, vals_b):
    """Paired t-test: is mean(a - b) significantly != 0?"""
    diffs = np.array(vals_a) - np.array(vals_b)
    n = len(diffs)
    if n < 2:
        return float("nan"), float("nan")
    mean_d = diffs.mean()
    std_d = diffs.std(ddof=1)
    if std_d < 1e-15:
        return mean_d, 0.0
    t_stat = mean_d / (std_d / np.sqrt(n))
    from scipy import stats
    p_val = stats.t.sf(abs(t_stat), df=n - 1) * 2
    return mean_d, p_val
```

**Step 2: Add the ablation study runner**

```python
def run_ablation_seed(surface_name, penalty_name, lw, seed, epochs=EPOCHS_AE):
    """Run one ablation: train AE, evaluate on held-out test set."""
    torch.manual_seed(seed)
    np.random.seed(seed)

    manifold_sde = create_manifold_sde(surface_name, with_dynamics=True)

    # Training data
    train_data = sample_from_manifold(
        manifold_sde,
        [(-TRAIN_BOUND, TRAIN_BOUND)] * 2,
        n_samples=N_TRAIN, seed=seed, device=DEVICE,
    )

    # Separate test data (independent seed)
    test_data = sample_from_manifold(
        manifold_sde,
        [(-TRAIN_BOUND, TRAIN_BOUND)] * 2,
        n_samples=500, seed=seed + 10000, device=DEVICE,
    )

    # Train
    model = train_ae_with_twophase(train_data, penalty_name, lw, epochs)

    # Evaluate
    test_losses = compute_losses_per_sample(model, test_data)

    return {
        "surface": surface_name,
        "penalty": penalty_name,
        "seed": seed,
        "reconstruction": float(test_losses["reconstruction"].mean()),
        "tangent": float(test_losses["tangent"].mean()),
        "curvature": float(test_losses["curvature"].mean()),
    }


def run_ablation_study(surfaces, configs, seeds, epochs=EPOCHS_AE):
    """Run full multi-seed ablation study."""
    rows = []
    total = len(surfaces) * len(configs) * len(seeds)
    i = 0
    for surf in surfaces:
        for seed in seeds:
            for name, lw in configs.items():
                i += 1
                print(f"[{i}/{total}] {surf} | {name} | seed={seed}")
                row = run_ablation_seed(surf, name, lw, seed, epochs)
                rows.append(row)
    return pd.DataFrame(rows)
```

**Step 3: Run ablation study**

Run: `PYTHONUNBUFFERED=1 python -m experiments.paper_experiments ablation --n-seeds 10 --output paper_ablation.csv`

Expected: CSV with 80 rows (1 surface × 8 configs × 10 seeds), columns: surface, penalty, seed, reconstruction, tangent, curvature.

**Step 4: Commit**

```bash
git add experiments/paper_experiments.py
git commit -m "Add unified multi-seed paper experiment runner (ablation)"
```

---

### Task 2: Add Extrapolation Study to `experiments/paper_experiments.py`

**Files:**
- Modify: `experiments/paper_experiments.py`

**Step 1: Add the extrapolation study runner**

This trains one AE per (surface, penalty, seed), then evaluates reconstruction error at different extrapolation distances.

```python
def run_extrapolation_seed(surface_name, penalty_name, lw, seed, epochs=EPOCHS_AE):
    """Train AE, evaluate reconstruction at extrapolation distances."""
    torch.manual_seed(seed)
    np.random.seed(seed)

    manifold_sde = create_manifold_sde(surface_name)

    train_data = sample_from_manifold(
        manifold_sde,
        [(-TRAIN_BOUND, TRAIN_BOUND)] * 2,
        n_samples=N_TRAIN, seed=seed, device=DEVICE,
    )

    model = train_ae_with_twophase(train_data, penalty_name, lw, epochs)

    # Evaluate at different distances
    distances = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]
    test_datasets = create_test_datasets(
        manifold_sde, TRAIN_BOUND, distances, 0.1, 500, DEVICE, seed + 10000
    )

    rows = []
    for dist, test_data in test_datasets.items():
        model.eval()
        with torch.no_grad():
            x = test_data.samples
            x_hat = model(x)
            recon_err = ((x_hat - x) ** 2).sum(dim=-1).sqrt().mean().item()

        rows.append({
            "surface": surface_name,
            "penalty": penalty_name,
            "seed": seed,
            "distance": dist,
            "reconstruction_error": recon_err,
        })

    return rows


def run_extrapolation_study(surfaces, configs, seeds, epochs=EPOCHS_AE):
    """Run full multi-seed extrapolation study."""
    rows = []
    total = len(surfaces) * len(configs) * len(seeds)
    i = 0
    for surf in surfaces:
        for seed in seeds:
            for name, lw in configs.items():
                i += 1
                print(f"[{i}/{total}] {surf} | {name} | seed={seed}")
                rows.extend(run_extrapolation_seed(surf, name, lw, seed, epochs))
    return pd.DataFrame(rows)
```

**Step 2: Run extrapolation study**

Run: `PYTHONUNBUFFERED=1 python -m experiments.paper_experiments extrapolation --n-seeds 10 --output paper_extrapolation.csv`

Expected: CSV with 1440 rows (4 surfaces × 6 configs × 10 seeds × 6 distances).

**Step 3: Commit**

```bash
git add experiments/paper_experiments.py
git commit -m "Add multi-seed extrapolation study"
```

---

### Task 3: Add Dynamics Extrapolation Study to `experiments/paper_experiments.py`

**Files:**
- Modify: `experiments/paper_experiments.py`

**Step 1: Add the dynamics extrapolation runner**

Reuse `compute_dynamics_metrics()` from `experiments/dynamics_extrapolation_study.py`.

```python
from experiments.dynamics_extrapolation_study import compute_dynamics_metrics

def run_dynamics_extrap_seed(surface_name, penalty_name, lw, seed, epochs=EPOCHS_AE):
    """Train AE, evaluate dynamics metrics at extrapolation distances."""
    torch.manual_seed(seed)
    np.random.seed(seed)

    manifold_sde = create_manifold_sde(surface_name, with_dynamics=True)

    train_data = sample_from_manifold(
        manifold_sde,
        [(-TRAIN_BOUND, TRAIN_BOUND)] * 2,
        n_samples=N_TRAIN, seed=seed, device=DEVICE,
    )

    model = train_ae_with_twophase(train_data, penalty_name, lw, epochs)

    distances = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]
    test_datasets = create_test_datasets(
        manifold_sde, TRAIN_BOUND, distances, 0.1, 500, DEVICE, seed + 10000
    )

    rows = []
    for dist, test_data in test_datasets.items():
        metrics = compute_dynamics_metrics(model, test_data)
        rows.append({
            "surface": surface_name,
            "penalty": penalty_name,
            "seed": seed,
            "distance": dist,
            **metrics,
        })

    return rows


def run_dynamics_extrap_study(surfaces, configs, seeds, epochs=EPOCHS_AE):
    """Run full multi-seed dynamics extrapolation study."""
    rows = []
    total = len(surfaces) * len(configs) * len(seeds)
    i = 0
    for surf in surfaces:
        for seed in seeds:
            for name, lw in configs.items():
                i += 1
                print(f"[{i}/{total}] {surf} | {name} | seed={seed}")
                rows.extend(run_dynamics_extrap_seed(surf, name, lw, seed, epochs))
    return pd.DataFrame(rows)
```

**Step 2: Run dynamics extrapolation study**

Run: `PYTHONUNBUFFERED=1 python -m experiments.paper_experiments dynamics --n-seeds 10 --output paper_dynamics.csv`

Expected: CSV with 360 rows (1 surface × 6 configs × 10 seeds × 6 distances).

**Step 3: Commit**

```bash
git add experiments/paper_experiments.py
git commit -m "Add multi-seed dynamics extrapolation study"
```

---

### Task 4: Add CLI and summary statistics to `experiments/paper_experiments.py`

**Files:**
- Modify: `experiments/paper_experiments.py`

**Step 1: Add the CLI main() function and summary printer**

```python
def print_ablation_summary(df):
    """Print ablation summary: mean ± std per config, with paired t-tests vs baseline."""
    print("\n" + "=" * 90)
    print("ABLATION SUMMARY (mean ± std across seeds)")
    print("=" * 90)

    metrics = ["reconstruction", "tangent", "curvature"]
    penalties = df["penalty"].unique()

    for surf in df["surface"].unique():
        print(f"\n  Surface: {surf}")
        print(f"  {'Penalty':<12s}", end="")
        for m in metrics:
            print(f"  {m:>20s}", end="")
        print()
        print("  " + "-" * 74)

        for pen in penalties:
            sub = df[(df["surface"] == surf) & (df["penalty"] == pen)]
            print(f"  {pen:<12s}", end="")
            for m in metrics:
                vals = sub[m].values
                print(f"  {vals.mean():>8.4f} ± {vals.std():>6.4f}", end="")
            print()


def main():
    parser = argparse.ArgumentParser(description="Multi-seed paper experiments")
    sub = parser.add_subparsers(dest="study", required=True)

    # Ablation
    ab = sub.add_parser("ablation")
    ab.add_argument("--surfaces", nargs="+", default=["paraboloid"])
    ab.add_argument("--n-seeds", type=int, default=10)
    ab.add_argument("--epochs", type=int, default=EPOCHS_AE)
    ab.add_argument("--base-seed", type=int, default=42)
    ab.add_argument("--output", type=str, default="paper_ablation.csv")

    # Extrapolation
    ex = sub.add_parser("extrapolation")
    ex.add_argument("--surfaces", nargs="+",
                     default=["paraboloid", "hyperbolic_paraboloid", "monkey_saddle", "sinusoidal"])
    ex.add_argument("--n-seeds", type=int, default=10)
    ex.add_argument("--epochs", type=int, default=EPOCHS_AE)
    ex.add_argument("--base-seed", type=int, default=42)
    ex.add_argument("--output", type=str, default="paper_extrapolation.csv")

    # Dynamics extrapolation
    dy = sub.add_parser("dynamics")
    dy.add_argument("--surfaces", nargs="+", default=["paraboloid"])
    dy.add_argument("--n-seeds", type=int, default=10)
    dy.add_argument("--epochs", type=int, default=EPOCHS_AE)
    dy.add_argument("--base-seed", type=int, default=42)
    dy.add_argument("--output", type=str, default="paper_dynamics.csv")

    args = parser.parse_args()
    seeds = [args.base_seed + i * 1000 for i in range(args.n_seeds)]
    print(f"Device: {DEVICE}")
    print(f"Seeds ({len(seeds)}): {seeds}")

    t0 = time.time()

    if args.study == "ablation":
        df = run_ablation_study(args.surfaces, ABLATION_CONFIGS, seeds, args.epochs)
        df.to_csv(args.output, index=False)
        print_ablation_summary(df)

    elif args.study == "extrapolation":
        df = run_extrapolation_study(args.surfaces, PENALTY_CONFIGS, seeds, args.epochs)
        df.to_csv(args.output, index=False)

    elif args.study == "dynamics":
        df = run_dynamics_extrap_study(args.surfaces, PENALTY_CONFIGS, seeds, args.epochs)
        df.to_csv(args.output, index=False)

    print(f"\nSaved to {args.output}")
    print(f"Total time: {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
```

**Step 2: Smoke test all three subcommands**

Run:
```bash
PYTHONUNBUFFERED=1 python -m experiments.paper_experiments ablation --n-seeds 1 --epochs 50
PYTHONUNBUFFERED=1 python -m experiments.paper_experiments extrapolation --n-seeds 1 --epochs 50 --surfaces paraboloid
PYTHONUNBUFFERED=1 python -m experiments.paper_experiments dynamics --n-seeds 1 --epochs 50
```

Expected: Each produces a CSV without errors.

**Step 3: Commit**

```bash
git add experiments/paper_experiments.py
git commit -m "Add CLI and summary statistics to paper experiments"
```

---

### Task 5: Run All Paper Experiments (Full 10-Seed)

**Files:**
- Output: `paper_ablation.csv`, `paper_extrapolation.csv`, `paper_dynamics.csv`

**Step 1: Run ablation study**

```bash
PYTHONUNBUFFERED=1 python -m experiments.paper_experiments ablation --n-seeds 10 --output paper_ablation.csv 2>&1 | tee paper_ablation.log
```

Expected: ~80 runs, CSV with 80 rows. Should take ~10 minutes.

**Step 2: Run extrapolation study**

```bash
PYTHONUNBUFFERED=1 python -m experiments.paper_experiments extrapolation --n-seeds 10 --output paper_extrapolation.csv 2>&1 | tee paper_extrapolation.log
```

Expected: ~240 runs, CSV with 1440 rows. Should take ~40 minutes.

**Step 3: Run dynamics extrapolation study**

```bash
PYTHONUNBUFFERED=1 python -m experiments.paper_experiments dynamics --n-seeds 10 --output paper_dynamics.csv 2>&1 | tee paper_dynamics.log
```

Expected: ~60 runs, CSV with 360 rows. Should take ~10 minutes.

**Step 4: Verify all CSVs exist and have expected row counts**

```bash
wc -l paper_ablation.csv paper_extrapolation.csv paper_dynamics.csv
```

---

### Task 6: Update Section 7.1 (Synthetic Surfaces & Setup)

**Files:**
- Modify: `Autoencoder-Paper/paper.tex:2423-2480`

**Step 1: Rewrite the data generation and architecture paragraphs**

Replace lines 2447-2457 with updated text reflecting N=20, two-phase training:

```latex
\paragraph{Data generation.}
On each surface we generate $m=20$ points by importance sampling with the
normalised volume density
$\sqrt{\det g(u,v)}/\int_{[-1,1]^2}\sqrt{\det g}\,du\,dv$.
This sparse-data setting models the realistic scenario in which only a
handful of observations are available from the manifold.
Test-set metrics are evaluated on 500 independently sampled points.
All experiments are replicated over 10 random seeds; we report
mean $\pm$ standard deviation and, where appropriate, paired-$t$-test
$p$-values.

\paragraph{Architecture.}
Both the encoder and decoder consist of $2$ hidden layers with $64$ neurons
each and $\tanh$ activation.  The latent dimension is $d=2$.
Training uses Adam with learning rate $0.005$, batch size $32$, and $500$
epochs.

\paragraph{Two-phase training.}
When the curvature penalty~K is active, training is split into two phases:
(i)~$250$ epochs of warm-up with only the non-K penalties, followed by
(ii)~$250$ epochs of fine-tuning with the full penalty set including~K.
This schedule prevents the curvature penalty from destabilising the
autoencoder in the early training phase, which is critical in the
sparse-data regime (see Section~\ref{ssec:traj_fidelity}).
```

**Step 2: Commit**

```bash
cd Autoencoder-Paper && git add paper.tex && git commit -m "Update Section 7.1: N=20 sparse data, two-phase training"
```

---

### Task 7: Update Section 7.2 (Ablation Study)

**Files:**
- Modify: `Autoencoder-Paper/paper.tex:2482-2541`

**Step 1: Read ablation CSV and compute the table values**

```bash
python -c "
import pandas as pd
df = pd.read_csv('paper_ablation.csv')
for pen in ['baseline','T','K','F','T+K','T+F','F+K','T+F+K']:
    sub = df[df['penalty']==pen]
    r = sub['reconstruction']
    t = sub['tangent']
    c = sub['curvature']
    print(f'{pen:<10s} & {r.mean():.3f} $\\pm$ {r.std():.3f} & {t.mean():.3f} $\\pm$ {t.std():.3f} & {c.mean():.3f} $\\pm$ {c.std():.3f} \\\\')
"
```

**Step 2: Replace the ablation table and analysis text**

Update Table 1 with mean ± std format. Update the analysis bullets to reflect the N=20 results. Key things to check:
- Does T+F+K still achieve the best balance?
- Does K alone still crush curvature error?
- Does F alone still hurt (worse than baseline)?
- The narrative should note that with N=20, these patterns may differ slightly from dense-data

**Step 3: Commit**

```bash
cd Autoencoder-Paper && git add paper.tex && git commit -m "Update Section 7.2: ablation table with N=20, 10-seed mean±std"
```

---

### Task 8: Update Section 7.3 (Extrapolation Study)

**Files:**
- Modify: `Autoencoder-Paper/paper.tex:2544-2646`

**Step 1: Read extrapolation CSV and compute table values at delta=0 and delta=0.5**

```bash
python -c "
import pandas as pd
df = pd.read_csv('paper_extrapolation.csv')
for dist in [0.0, 0.5]:
    print(f'\ndist={dist}:')
    sub = df[df['distance']==dist]
    for surf in ['paraboloid','hyperbolic_paraboloid','monkey_saddle','sinusoidal']:
        for pen in ['baseline','T','T+K','T+F','T+F+K','K']:
            s = sub[(sub['surface']==surf) & (sub['penalty']==pen)]
            r = s['reconstruction_error']
            print(f'  {surf:<25s} {pen:<8s} {r.mean():.3f} ± {r.std():.3f}')
"
```

**Step 2: Replace the extrapolation table and analysis**

Update Table 2 with mean ± std format across 10 seeds. Update the analysis bullets. The figure (`extrapolation_summary.png`) should be regenerated from the new data.

**Step 3: Regenerate extrapolation figure**

Write a small script or modify `paper_experiments.py` to generate `extrapolation_summary.png` from the CSV data.

**Step 4: Commit**

```bash
cd Autoencoder-Paper && git add paper.tex extrapolation_summary.png && git commit -m "Update Section 7.3: extrapolation with N=20, 10-seed mean±std"
```

---

### Task 9: Update Section 7.4 (Dynamics Extrapolation)

**Files:**
- Modify: `Autoencoder-Paper/paper.tex:2649-2716`

**Step 1: Read dynamics CSV and compute table values**

```bash
python -c "
import pandas as pd
df = pd.read_csv('paper_dynamics.csv')
for dist in [0.0, 0.5]:
    print(f'\ndist={dist}:')
    sub = df[df['distance']==dist]
    for pen in ['baseline','T','K','T+K','T+F','T+F+K']:
        s = sub[sub['penalty']==pen]
        print(f'  {pen:<8s} cov_tan={s[\"cov_tangent\"].mean():.3f}±{s[\"cov_tangent\"].std():.3f}  '
              f'drift_tan={s[\"drift_tangent\"].mean():.4f}±{s[\"drift_tangent\"].std():.4f}')
"
```

**Step 2: Replace Table 3 and analysis**

Update with mean ± std format. Check that K still dominates dynamics metrics.

**Step 3: Commit**

```bash
cd Autoencoder-Paper && git add paper.tex && git commit -m "Update Section 7.4: dynamics extrapolation with N=20, 10-seed mean±std"
```

---

### Task 10: Rewrite Section 7.5 (Trajectory Fidelity → Multi-Seed K Study)

**Files:**
- Modify: `Autoencoder-Paper/paper.tex:2718-2820`

**Step 1: Read existing multiseed_K_study.csv and compute table values**

```bash
python -c "
import pandas as pd, numpy as np
df = pd.read_csv('multiseed_K_study.csv')
metrics = ['MTE@1.0', 'W2@1.0', 'MMD@1.0']
for surf in ['paraboloid', 'hyperbolic_paraboloid', 'sinusoidal']:
    tf = df[(df['surface']==surf) & (df['reg']=='T+F')].sort_values('seed')
    tfk = df[(df['surface']==surf) & (df['reg']=='T+F+K')].sort_values('seed')
    print(f'\n{surf}:')
    for m in metrics:
        mean_tf, std_tf = tf[m].mean(), tf[m].std()
        mean_tfk, std_tfk = tfk[m].mean(), tfk[m].std()
        delta_pct = (mean_tfk - mean_tf) / mean_tf * 100
        # paired t-test
        diffs = tfk[m].values - tf[m].values
        n = len(diffs)
        t_stat = diffs.mean() / (diffs.std(ddof=1) / np.sqrt(n))
        from scipy.stats import t as tdist
        p = tdist.sf(abs(t_stat), df=n-1) * 2
        print(f'  {m}: T+F={mean_tf:.3f}±{std_tf:.3f}  T+F+K={mean_tfk:.3f}±{std_tfk:.3f}  '
              f'delta={delta_pct:+.1f}%  p={p:.4f}')
"
```

**Step 2: Replace Section 7.5 content**

The new section should contain:

1. **Motivation**: In the sparse-data regime (N=20), does curvature regularization improve trajectory fidelity?

2. **Two-phase training explanation**: Brief (already covered in 7.1, just cross-reference)

3. **Experimental setup**: 10 seeds, N=20, 3 surfaces, T+F vs T+F+K, paired t-test. Full SDE pipeline (Stage 1: AE, Stage 2: drift_net, Stage 3: diffusion_net).

4. **Results table**: For each surface: T+F mean±std, T+F+K mean±std, % change, p-value for MTE, W2, MMD.

5. **Developable surface analysis**: Sinusoidal z=sin(u+v) has zero Gaussian curvature (developable surface, rank-1 Hessian). K cannot help because there is no curvature signal.

6. **Figures**: Keep trajectory plots (paraboloid and hyperbolic_paraboloid T+F+K vs baseline).

Example LaTeX table structure:

```latex
\begin{table}[ht]
\centering
\caption{Multi-seed trajectory fidelity ($N=20$, 10 seeds, paired $t$-test).
T+F+K vs T+F: negative $\Delta$ = K helps.
${}^{**}p<0.01$, ${}^{*}p<0.05$, ${}^{+}p<0.1$.}
\label{tab:traj_fidelity}
\small
\begin{tabular}{llcccc}
\toprule
Surface & Metric & T+F & T+F+K & $\Delta$ (\%) & $p$-value\\
\midrule
Paraboloid & MTE & $X \pm Y$ & $X \pm Y$ & $-6.6$ & $0.009^{**}$\\
           & W2  & ... & ... & $-13.8$ & $0.005^{**}$\\
           & MMD & ... & ... & ... & ...\\
\midrule
Hyp.\ parab. & MTE & ... \\
             & W2  & ... \\
             & MMD & ... \\
\midrule
Sinusoidal & MTE & ... \\
           & W2  & ... \\
           & MMD & ... \\
\bottomrule
\end{tabular}
\end{table}
```

**Step 3: Add developable surface remark**

```latex
\begin{rem}[Developable surfaces]\label{rem:developable}
The sinusoidal surface $z=\sin(u+v)$ has Gaussian curvature identically zero:
its Hessian is rank-$1$ with $\det(H)=0$ everywhere, making it a
\emph{developable surface}.
Since the curvature penalty targets the normal drift condition
$(I-P)(b - \tfrac12 q) = 0$, and this condition is trivially satisfied
on a flat surface, the K penalty carries no geometric signal.
This explains why T+F+K and T+F perform comparably on the sinusoidal surface,
and illustrates that the curvature regularisation acts as a geometric inductive
bias that is specifically effective when the surface has nonzero intrinsic curvature.
\end{rem}
```

**Step 4: Commit**

```bash
cd Autoencoder-Paper && git add paper.tex && git commit -m "Rewrite Section 7.5: multi-seed K study with developable surface analysis"
```

---

### Task 11: Update Section 8 (Discussion)

**Files:**
- Modify: `Autoencoder-Paper/paper.tex:2825-2950`

**Step 1: Update discussion to reflect sparse-data findings**

Key changes to the discussion:

1. Add a new paragraph about the sparse-data regime:

```latex
\paragraph{Curvature as geometric inductive bias.}
The multi-seed trajectory fidelity study (Section~\ref{ssec:traj_fidelity})
reveals that the curvature penalty serves as a \emph{geometric inductive bias}
whose value is most pronounced in the sparse-data regime.
With $N=2{,}000$ training points, the autoencoder learns a sufficiently
accurate chart that additional curvature supervision provides no measurable
benefit---the data itself supplies enough geometric information.
With only $N=20$ observations, however, the curvature penalty
significantly improves trajectory fidelity: on the paraboloid, T+F+K
reduces MTE by $6.6\%$ ($p=0.009$) and W2 by $13.8\%$ ($p=0.005$)
relative to T+F alone, with $9$ out of $10$ random seeds showing
improvement.
This finding aligns with the classical statistical learning perspective:
geometric regularisation compensates for limited data by encoding structural
prior knowledge about the manifold.
```

2. Add a sentence about the developable surface finding:

```latex
Conversely, on the sinusoidal surface---a \emph{developable surface}
with identically zero Gaussian curvature---the K penalty has no effect,
confirming that its value derives specifically from intrinsic curvature
information.
```

3. Update the paragraph about complementary roles of K and F (lines 2841-2867) to mention the sparse-data dependence.

**Step 2: Commit**

```bash
cd Autoencoder-Paper && git add paper.tex && git commit -m "Update Section 8: sparse-data geometric inductive bias discussion"
```

---

### Task 12: Final Review and Cleanup

**Files:**
- Review: `Autoencoder-Paper/paper.tex` (all modified sections)
- Review: `experiments/paper_experiments.py`

**Step 1: Check paper compiles**

```bash
cd Autoencoder-Paper && pdflatex paper.tex
```

Expected: No errors. Check for overfull hboxes or formatting issues.

**Step 2: Verify all table numbers match CSV data**

Cross-check every number in the LaTeX tables against the corresponding CSV files.

**Step 3: Run tests to ensure nothing is broken**

```bash
python -m pytest tests/ -v
```

Expected: All tests pass.

**Step 4: Final commit**

```bash
git add -A && git commit -m "Paper update: all experiments at N=20 with 10-seed statistical rigor"
```
