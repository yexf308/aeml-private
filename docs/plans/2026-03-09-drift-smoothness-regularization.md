# Drift Smoothness Regularization Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add a metric-weighted latent Jacobian penalty to drift_net training so that curvature regularization K translates to better trajectory statistics (W2), not just better pointwise coefficients (E_mu).

**Architecture:** Add `drift_smoothness_loss()` to `sde_losses.py` that computes `tr(J_bz^T g J_bz g^{-1}) / d` at augmented (perturbed) latent points, using `torch.func.jacrev` for the exact d×d Jacobian and the chart's metric g from the frozen decoder. Integrate into `train_stage2` via a `lambda_smooth` parameter. Validate on D=3 paraboloid N=20 with 10 seeds.

**Tech Stack:** PyTorch, torch.func (jacrev/vmap), existing AE/SDE pipeline

---

### Task 1: `drift_smoothness_loss` function

**Files:**
- Modify: `src/numeric/sde_losses.py` (add function after line 75)
- Test: `tests/test_sde_nets.py` (add test class)

**Step 1: Write the failing test**

Add to `tests/test_sde_nets.py`:

```python
class TestDriftSmoothnessLoss:
    def test_loss_runs(self, ae, drift_net):
        """Smoothness loss computes without error."""
        from src.numeric.sde_losses import drift_smoothness_loss
        z_aug = torch.randn(6, 2)
        ae.eval()
        for p in ae.parameters():
            p.requires_grad_(False)
        loss = drift_smoothness_loss(ae.decoder, drift_net, z_aug)
        assert loss.dim() == 0
        assert torch.isfinite(loss)
        assert loss.item() >= 0

    def test_gradient_flows_to_drift_net_only(self, ae, drift_net):
        """Gradient should flow to drift_net, not AE."""
        from src.numeric.sde_losses import drift_smoothness_loss
        z_aug = torch.randn(6, 2)
        ae.eval()
        for p in ae.parameters():
            p.requires_grad_(False)
        loss = drift_smoothness_loss(ae.decoder, drift_net, z_aug)
        loss.backward()
        drift_has_grad = any(
            p.grad is not None and p.grad.abs().sum() > 0
            for p in drift_net.parameters()
        )
        assert drift_has_grad
        for p in ae.parameters():
            assert p.grad is None or p.grad.abs().sum() == 0

    def test_smoother_drift_has_lower_loss(self):
        """A constant drift should have lower smoothness loss than a wiggly one."""
        from src.numeric.sde_losses import drift_smoothness_loss
        from src.numeric.autoencoders import AutoEncoder
        import torch.nn as nn
        ae = AutoEncoder(3, 2, [8], nn.Tanh(), nn.Tanh(), False)
        ae.eval()
        for p in ae.parameters():
            p.requires_grad_(False)
        z = torch.randn(10, 2)

        # Constant drift (zero Jacobian)
        const_net = DriftNet(2, [8, 8])
        with torch.no_grad():
            for p in const_net.parameters():
                p.zero_()
        loss_const = drift_smoothness_loss(ae.decoder, const_net, z)

        # Random drift (nonzero Jacobian)
        rand_net = DriftNet(2, [8, 8])
        loss_rand = drift_smoothness_loss(ae.decoder, rand_net, z)

        assert loss_const < loss_rand, \
            f"Constant drift ({loss_const:.6f}) should be smoother than random ({loss_rand:.6f})"
```

**Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_sde_nets.py::TestDriftSmoothnessLoss -v`
Expected: FAIL with "cannot import name 'drift_smoothness_loss'"

**Step 3: Write minimal implementation**

Add to `src/numeric/sde_losses.py` after `tangential_drift_loss` (after line 75):

```python
def drift_smoothness_loss(
    decoder,
    drift_net,
    z_aug: Tensor,
) -> Tensor:
    """
    Metric-weighted latent Jacobian penalty for drift_net smoothness.

    L = E_z[ tr(J_bz^T g J_bz g^{-1}) / d ]

    where J_bz = d(b_z)/dz (d x d) and g = Dphi^T Dphi is the metric from
    the frozen decoder.  A better chart (from K regularization) produces a
    more accurate g, making this penalty geometrically meaningful.

    Args:
        decoder: Frozen decoder (requires_grad=False on params).
        drift_net: DriftNet being trained.
        z_aug: Augmented latent points, shape (B, d).

    Returns:
        Scalar loss (non-negative).
    """
    import torch

    d = z_aug.shape[-1]

    # Metric tensor from frozen decoder
    with torch.no_grad():
        dphi = decoder.jacobian_network(z_aug)          # (B, D, d)
        g = dphi.mT @ dphi                              # (B, d, d)
        ginv = regularized_metric_inverse(g)             # (B, d, d)

    # Exact Jacobian of drift_net: J_bz[i,j] = d(b_z)_i / dz_j
    J_bz = torch.func.vmap(torch.func.jacrev(drift_net))(z_aug)  # (B, d, d)

    # tr(J_bz^T g J_bz g^{-1}) / d
    M = J_bz.mT @ g @ J_bz @ ginv                       # (B, d, d)
    return torch.diagonal(M, dim1=-2, dim2=-1).sum(-1).mean() / d
```

**Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_sde_nets.py::TestDriftSmoothnessLoss -v`
Expected: 3 PASS

**Step 5: Commit**

```bash
git add src/numeric/sde_losses.py tests/test_sde_nets.py
git commit -m "feat: add metric-weighted drift smoothness loss"
```

---

### Task 2: Integrate into `train_stage2`

**Files:**
- Modify: `src/numeric/sde_training.py:73-128` (add lambda_smooth to train_stage2)
- Modify: `src/numeric/sde_training.py:130-169` (add lambda_smooth to train_stage2_precomputed)

**Step 1: Write the failing test**

Add to `tests/test_sde_nets.py`:

```python
class TestStage2WithSmoothing:
    def test_train_stage2_with_smoothing(self, ae, sample_data):
        """Stage 2 runs with lambda_smooth > 0."""
        from src.numeric.sde_training import SDEPipelineTrainer
        x, v, Lambda = sample_data
        drift = DriftNet(2, [8, 8])
        diff = DiffusionNet(2, [8, 8])
        trainer = SDEPipelineTrainer(ae, drift, diff)
        losses = trainer.train_stage2(
            x, v, Lambda, epochs=5, lr=1e-3,
            lambda_smooth=0.1, aug_sigma=0.1,
            print_interval=0,
        )
        assert len(losses) == 5
        assert all(l > 0 for l in losses)

    def test_smoothing_reduces_jacobian(self, ae, sample_data):
        """With smoothing, drift Jacobian norm should be smaller."""
        from src.numeric.sde_training import SDEPipelineTrainer
        from src.numeric.sde_losses import drift_smoothness_loss
        x, v, Lambda = sample_data
        torch.manual_seed(0)

        # Train without smoothing
        drift_no = DriftNet(2, [8, 8])
        diff_no = DiffusionNet(2, [8, 8])
        trainer_no = SDEPipelineTrainer(ae, drift_no, diff_no)
        trainer_no.train_stage2(x, v, Lambda, epochs=50, lr=1e-3, print_interval=0)

        # Train with smoothing
        torch.manual_seed(0)
        drift_sm = DriftNet(2, [8, 8])
        diff_sm = DiffusionNet(2, [8, 8])
        trainer_sm = SDEPipelineTrainer(ae, drift_sm, diff_sm)
        trainer_sm.train_stage2(
            x, v, Lambda, epochs=50, lr=1e-3,
            lambda_smooth=1.0, aug_sigma=0.1, print_interval=0,
        )

        z_test = torch.randn(20, 2)
        ae.eval()
        for p in ae.parameters():
            p.requires_grad_(False)
        smooth_no = drift_smoothness_loss(ae.decoder, drift_no, z_test).item()
        smooth_sm = drift_smoothness_loss(ae.decoder, drift_sm, z_test).item()
        assert smooth_sm < smooth_no, \
            f"Smoothed ({smooth_sm:.4f}) should have lower Jacobian norm than unsmoothed ({smooth_no:.4f})"
```

**Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_sde_nets.py::TestStage2WithSmoothing -v`
Expected: FAIL with "unexpected keyword argument 'lambda_smooth'"

**Step 3: Write minimal implementation**

Modify `train_stage2` in `src/numeric/sde_training.py` — add `lambda_smooth=0.0, aug_sigma=0.1` parameters and augmented-point smoothness computation inside the training loop:

```python
def train_stage2(
    self, x, v, Lambda, epochs, lr=1e-3, batch_size=32, print_interval=100,
    lambda_smooth=0.0, aug_sigma=0.1,
):
    # ... existing setup (lines 91-99) ...

    for epoch in range(epochs):
        epoch_loss = 0.0
        n_batches = 0
        for x_b, v_b, Lambda_b in loader:
            x_b = x_b.to(self.device)
            v_b = v_b.to(self.device)
            Lambda_b = Lambda_b.to(self.device)
            z = self.autoencoder.encoder(x_b).detach()
            loss = tangential_drift_loss(
                self.autoencoder.decoder, self.drift_net, z, v_b, Lambda_b,
            )

            if lambda_smooth > 0.0:
                z_aug = z + torch.randn_like(z) * aug_sigma
                loss = loss + lambda_smooth * drift_smoothness_loss(
                    self.autoencoder.decoder, self.drift_net, z_aug,
                )

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.drift_net.parameters(), max_norm=1.0)
            optimizer.step()
            epoch_loss += loss.item()
            n_batches += 1
        # ... rest unchanged ...
```

Apply the same pattern to `train_stage2_precomputed`: add `lambda_smooth=0.0, aug_sigma=0.1` params, and inside the batch loop add the smoothness term using `z_aug = z_b + randn * aug_sigma`. Note: the precomputed dphi/d2phi are only for training points; the smoothness loss computes its own dphi at augmented points via the decoder.

**Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_sde_nets.py::TestStage2WithSmoothing -v`
Expected: 2 PASS

**Step 5: Commit**

```bash
git add src/numeric/sde_training.py tests/test_sde_nets.py
git commit -m "feat: integrate drift smoothness into Stage 2 training"
```

---

### Task 3: Validation script (D=3, N=20, 10 seeds)

**Files:**
- Create: `experiments/validate_drift_smoothness.py`

**Step 1: Write the validation script**

```python
"""
Validate drift smoothness regularization on D=3 paraboloid, N=20, 10 seeds.

4 conditions:
  1. baseline     (no K, no smooth)
  2. K only       (K in AE, no smooth)
  3. smooth only  (no K, smooth in Stage 2)
  4. K + smooth   (K in AE, smooth in Stage 2)

Metrics:
  - E_mu at training points
  - E_mu at held-out points (uniform grid in [-1,1]^2, 100 points)
  - W2@1.0
"""
import argparse
import time
import torch
import pandas as pd
import numpy as np
from scipy import stats

from src.numeric.losses import LossWeights
from src.numeric.sde_nets import DriftNet, DiffusionNet
from src.numeric.sde_training import SDEPipelineTrainer
from experiments.data_driven_sde import (
    DEVICE, TRAIN_BOUND, BOUNDARY, N_TRAJ, DT, N_STEPS, LR_SDE, BATCH_SIZE,
    train_autoencoder, evaluate_pipeline,
)
from experiments.common import create_test_datasets

SURFACE = "paraboloid"
N_TRAIN = 20
EPOCHS_AE = 200
EPOCHS_SDE = 300
LAMBDA_SMOOTH = 0.5
AUG_SIGMA = 0.1

# AE loss configs
NO_K = LossWeights(tangent_bundle=1.0, diffeo=1.0, curvature=0.0)
WITH_K = LossWeights(tangent_bundle=1.0, diffeo=1.0, curvature=0.1)


def compute_held_out_drift_error(autoencoder, pipeline, sde, n_held=100):
    """E_mu at a held-out uniform grid (not training points)."""
    # Uniform grid in [-1,1]^2
    s = torch.linspace(-0.9, 0.9, int(n_held ** 0.5), device=DEVICE)
    grid = torch.stack(torch.meshgrid(s, s, indexing='ij'), dim=-1).reshape(-1, 2)

    # True ambient drift at grid points
    x_grid = sde.chart(grid)
    v_true = sde.ambient_drift(grid)

    # Predicted ambient drift via pipeline
    with torch.no_grad():
        z = autoencoder.encoder(x_grid)
        dphi = autoencoder.decoder.jacobian_network(z)
        d2phi = autoencoder.decoder.hessian_network(z)

        b_z = pipeline.drift_net(z)
        dphi_bz = (dphi @ b_z.unsqueeze(-1)).squeeze(-1)

        from src.numeric.geometry import curvature_drift_explicit_full, regularized_metric_inverse
        g = dphi.mT @ dphi
        ginv = regularized_metric_inverse(g)
        pinv = ginv @ dphi.mT
        P_hat = dphi @ pinv
        P_hat = 0.5 * (P_hat + P_hat.mT)
        Lambda_grid = sde.ambient_covariance(grid)
        Lambda_tan = P_hat @ Lambda_grid @ P_hat
        Lambda_tan = 0.5 * (Lambda_tan + Lambda_tan.mT)
        Sigma_z = pinv @ Lambda_tan @ pinv.mT
        Sigma_z = 0.5 * (Sigma_z + Sigma_z.mT)
        q = curvature_drift_explicit_full(d2phi, Sigma_z)

        v_pred = dphi_bz + q

    D = v_true.shape[-1]
    err = ((v_pred - v_true) ** 2).sum(-1).median().item() / D
    return err


def run_one(seed, ae_lw, lambda_smooth):
    """Run one seed with given AE loss weights and smoothness lambda."""
    from src.symbolic.manifold_sdes import ManifoldSDE
    from experiments.trajectory_fidelity_study import lambdify_sde

    sde_obj = ManifoldSDE.from_surface(SURFACE)
    sde = lambdify_sde(sde_obj)

    autoencoder, _ = train_autoencoder(
        SURFACE, sde, N_TRAIN, EPOCHS_AE, seed, ae_lw,
    )

    d = 2
    torch.manual_seed(seed + 100)
    drift_net = DriftNet(d).to(DEVICE)
    diffusion_net = DiffusionNet(d).to(DEVICE)
    pipeline = SDEPipelineTrainer(autoencoder, drift_net, diffusion_net, device=DEVICE)

    from experiments.data_driven_sde import sample_from_manifold
    from src.numeric.highd_manifolds import sample_from_highd_manifold

    # Get training data
    torch.manual_seed(seed)
    train_data = create_test_datasets(
        SURFACE, N_TRAIN, TRAIN_BOUND, seed, device=DEVICE
    )
    x = train_data.samples.to(DEVICE)
    v = train_data.mu.to(DEVICE)
    Lambda = train_data.cov.to(DEVICE)

    # Stage 2 with optional smoothing
    drift_losses = pipeline.train_stage2(
        x, v, Lambda, epochs=EPOCHS_SDE, lr=LR_SDE,
        batch_size=BATCH_SIZE, print_interval=0,
        lambda_smooth=lambda_smooth, aug_sigma=AUG_SIGMA,
    )

    # Stage 3
    diff_losses = pipeline.train_stage3(
        x, Lambda, epochs=EPOCHS_SDE, lr=LR_SDE,
        batch_size=BATCH_SIZE, print_interval=0,
    )

    # Evaluate
    results = evaluate_pipeline(pipeline, autoencoder, sde, seed)

    # Held-out drift error
    e_mu_held = compute_held_out_drift_error(autoencoder, pipeline, sde)
    results["E_mu_held"] = e_mu_held

    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-seeds", type=int, default=10)
    parser.add_argument("--lambda-smooth", type=float, default=LAMBDA_SMOOTH)
    args = parser.parse_args()

    seeds = [42 + i * 1000 for i in range(args.n_seeds)]

    conditions = [
        ("baseline",    NO_K,    0.0),
        ("K",           WITH_K,  0.0),
        ("smooth",      NO_K,    args.lambda_smooth),
        ("K+smooth",    WITH_K,  args.lambda_smooth),
    ]

    rows = []
    for seed in seeds:
        for cond_name, ae_lw, lam_s in conditions:
            print(f"\n{'='*50}")
            print(f"  seed={seed}  condition={cond_name}")
            print(f"{'='*50}")
            t0 = time.time()
            results = run_one(seed, ae_lw, lam_s)
            elapsed = time.time() - t0
            row = {"seed": seed, "condition": cond_name, **results}
            rows.append(row)
            print(f"  E_mu={results.get('E_mu','?'):.4f}  "
                  f"E_mu_held={results.get('E_mu_held','?'):.4f}  "
                  f"W2@1.0={results.get('W2@1.0','?'):.4f}  "
                  f"({elapsed:.0f}s)")

    df = pd.DataFrame(rows)
    csv_path = "drift_smoothness_validation.csv"
    df.to_csv(csv_path, index=False)
    print(f"\nSaved {len(df)} rows to {csv_path}")

    # Summary statistics
    metrics = ["E_mu", "E_mu_held", "W2@1.0"]
    print(f"\n{'='*70}")
    print("SUMMARY (paired t-test vs baseline)")
    print(f"{'='*70}")
    baseline = df[df["condition"] == "baseline"]
    for cond_name in ["K", "smooth", "K+smooth"]:
        cond = df[df["condition"] == cond_name]
        print(f"\n  {cond_name} vs baseline:")
        for m in metrics:
            bv = baseline.sort_values("seed")[m].values
            cv = cond.sort_values("seed")[m].values
            if len(bv) != len(cv) or len(bv) < 2:
                continue
            delta = (cv.mean() - bv.mean()) / bv.mean() * 100
            n_help = int((cv < bv).sum())
            _, p = stats.ttest_rel(cv, bv)
            sig = "**" if p < 0.01 else ("*" if p < 0.05 else ("+" if p < 0.1 else ""))
            print(f"    {m:<12} {delta:>+6.1f}%  {n_help}/{len(bv)} wins  p={p:.4f} {sig}")


if __name__ == "__main__":
    main()
```

**Step 2: Smoke test**

Run: `python -m experiments.validate_drift_smoothness --n-seeds 1`
Expected: Runs 4 conditions for seed=42, prints E_mu, E_mu_held, W2, saves CSV.

**Step 3: Full validation**

Run: `python -u -m experiments.validate_drift_smoothness --n-seeds 10`
Expected: 40 rows (4 conditions × 10 seeds). Summary table shows whether K+smooth beats baseline/K on E_mu_held and W2.

**Step 4: Commit**

```bash
git add experiments/validate_drift_smoothness.py
git commit -m "feat: add drift smoothness validation experiment"
```

---

### Task 4: Analyze and iterate

**No code — analysis task.**

After the validation run completes, check:

1. **Does smoothing reduce E_mu_held?** This is the primary validation — if the drift is smoother, the held-out drift error should decrease.
2. **Does K+smooth beat K alone on W2?** This is the paper story — K improves the chart, smoothing improves the drift generalization, together they improve trajectories.
3. **Does smoothing hurt E_mu at training points?** If so, lambda_smooth is too high.
4. **Calibrate lambda_smooth**: If the effect is too weak, increase lambda_smooth. If training E_mu degrades, decrease it. Try [0.1, 0.5, 1.0, 2.0].

**Success criteria:**
- K+smooth shows significant improvement over K alone on E_mu_held (p < 0.05)
- K+smooth shows improvement over K alone on W2@1.0 (at least directional, ideally p < 0.1)
- Training E_mu does not degrade significantly

If successful, scale to D=201 with the N×D sweep.

---

## File Summary

| File | Action |
|------|--------|
| `src/numeric/sde_losses.py` | Add `drift_smoothness_loss()` |
| `src/numeric/sde_training.py` | Add `lambda_smooth`, `aug_sigma` to `train_stage2` and `train_stage2_precomputed` |
| `tests/test_sde_nets.py` | Add `TestDriftSmoothnessLoss`, `TestStage2WithSmoothing` |
| `experiments/validate_drift_smoothness.py` | New validation script |
