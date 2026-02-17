# Full-Space Curvature Loss Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add a full-space curvature loss (`curvature_full`) that matches the entire ambient-space Ito correction vector instead of only its normal projection, then add experiment configs to test it.

**Architecture:** Thread a new `local_cov` field (true σσ^T in local coordinates) through the data pipeline (DatasetBatch → datagen → training). Add `_full` geometry functions that skip normal projection. Wire a new `curvature_full` weight into `autoencoder_loss` that compares model vs. true full Ito correction in ambient space.

**Chart dependence note:** The full Ito correction `q = 0.5 * tr(Λ·H)` is NOT chart-invariant on its own — only `Dφ·b_z + q` (the ambient drift) is. Matching `q_model` to `q_true` implicitly fixes a gauge. This is acceptable in our synthetic setting (known true chart) and when paired with diffeo (F), which constrains the parameterization. See design doc for details.

**Naming convention:** Throughout this plan, `local_cov_true` refers to the true σσ^T in local (u,v) coordinates (from data), while `local_cov_z` refers to the pulled-back ambient covariance in latent z-coordinates (computed at runtime via `pinv(dphi) @ Λ @ pinv(dphi)^T`). The DatasetBatch field is named `local_cov` (always the true σσ^T).

**Tech Stack:** PyTorch, sympy, pytest

---

### Task 1: Add `local_cov` field to DatasetBatch

**Files:**
- Modify: `src/numeric/datasets.py:7-55`
- Test: `tests/test_datagen.py`

**Step 1: Write the failing test**

Add to `tests/test_datagen.py`:

```python
def test_dataset_batch_local_cov_field():
    """DatasetBatch should carry optional local_cov field."""
    n, D, d = 4, 3, 2
    batch = DatasetBatch(
        samples=torch.randn(n, D),
        local_samples=torch.randn(n, d),
        mu=torch.randn(n, D),
        cov=torch.randn(n, D, D),
        p=torch.randn(n, D, D),
        weights=torch.randn(n),
        hessians=torch.randn(n, D, d, d),
        local_cov=torch.randn(n, d, d),
    )
    assert batch.local_cov.shape == (n, d, d)

    # as_tuple should include local_cov (8 elements)
    t = batch.as_tuple()
    assert len(t) == 8
    assert t[7].shape == (n, d, d)

    # from_tuple round-trip
    batch2 = DatasetBatch.from_tuple(t)
    assert torch.allclose(batch2.local_cov, batch.local_cov)


def test_dataset_batch_local_cov_defaults_none():
    """local_cov should default to None and as_tuple should still work (7 elements)."""
    n, D, d = 4, 3, 2
    batch = DatasetBatch(
        samples=torch.randn(n, D),
        local_samples=torch.randn(n, d),
        mu=torch.randn(n, D),
        cov=torch.randn(n, D, D),
        p=torch.randn(n, D, D),
        weights=torch.randn(n),
        hessians=torch.randn(n, D, d, d),
    )
    assert batch.local_cov is None
    t = batch.as_tuple()
    assert len(t) == 7

    # from_tuple with 7 elements still works
    batch2 = DatasetBatch.from_tuple(t)
    assert batch2.local_cov is None
```

**Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_datagen.py::test_dataset_batch_local_cov_field tests/test_datagen.py::test_dataset_batch_local_cov_defaults_none -v`
Expected: FAIL — `local_cov` is not a field of DatasetBatch yet.

**Step 3: Implement**

In `src/numeric/datasets.py`, make these changes:

1. Add field after `hessians` (before `tangent_basis`):
```python
    hessians: torch.Tensor
    local_cov: torch.Tensor = None  # Shape: (batch, d, d) — true local covariance σσ^T
    # Efficient tangent basis storage: U_d from SVD of P (optional)
    tangent_basis: torch.Tensor = None  # Shape: (batch, D, d) - top d eigenvectors
```

2. Update `as_tuple()` to conditionally include `local_cov`:
```python
    def as_tuple(self) -> Tuple[torch.Tensor, ...]:
        base = (
            self.samples,
            self.local_samples,
            self.mu,
            self.cov,
            self.p,
            self.weights,
            self.hessians,
        )
        if self.local_cov is not None:
            return base + (self.local_cov,)
        return base
```

3. Update `from_tuple()` to accept 7 or 8 elements:
```python
    @classmethod
    def from_tuple(cls, tensors: Tuple[torch.Tensor, ...]) -> "DatasetBatch":
        if len(tensors) == 7:
            return cls(*tensors)
        elif len(tensors) == 8:
            return cls(*tensors[:7], local_cov=tensors[7])
        else:
            raise ValueError(f"Expected tuple of length 7 or 8, got {len(tensors)}.")
```

**Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_datagen.py::test_dataset_batch_local_cov_field tests/test_datagen.py::test_dataset_batch_local_cov_defaults_none -v`
Expected: PASS

**Step 5: Run full test suite to check for regressions**

Run: `python -m pytest tests/ -v`
Expected: All existing tests pass (from_tuple with 7 elements still works).

**Step 6: Commit**

```bash
git add src/numeric/datasets.py tests/test_datagen.py
git commit -m "feat: add local_cov field to DatasetBatch"
```

---

### Task 2: Compute `local_cov` in datagen and thread through embedding

**Files:**
- Modify: `src/numeric/datagen.py:9-56` (sample_from_manifold)
- Modify: `src/numeric/datagen.py:90-107` (embed_dataset_with_qr_matrix)
- Test: `tests/test_datagen.py`

**Step 1: Write the failing test**

Add to `tests/test_datagen.py`:

```python
def test_sample_from_manifold_includes_local_cov():
    """sample_from_manifold should populate local_cov field."""
    man = _make_manifold()
    sde = ManifoldSDE(man)
    bounds = [(-1.0, 1.0), (-1.0, 1.0)]
    batch = sample_from_manifold(sde, bounds, n_samples=8, seed=123)

    assert batch.local_cov is not None
    assert batch.local_cov.shape == (8, 2, 2)
    # local_cov should be positive semi-definite (eigenvalues >= 0)
    # Symmetrize first to avoid numerical flake
    lc = batch.local_cov
    lc_sym = 0.5 * (lc + lc.transpose(-1, -2))
    eigvals = torch.linalg.eigvalsh(lc_sym)
    assert (eigvals >= -1e-6).all()


def test_embed_dataset_preserves_local_cov():
    """embed_dataset_with_qr_matrix should preserve local_cov (intrinsic, not embedded)."""
    from src.numeric.datagen import embed_dataset_with_qr_matrix, create_embedding_matrix

    n, d = 4, 2
    extrinsic_dim = 5
    embedding_dim = 3
    local_cov_orig = torch.randn(n, d, d)
    batch = DatasetBatch(
        samples=torch.randn(n, extrinsic_dim),
        local_samples=torch.randn(n, d),
        mu=torch.randn(n, extrinsic_dim),
        cov=torch.randn(n, extrinsic_dim, extrinsic_dim),
        p=torch.randn(n, extrinsic_dim, extrinsic_dim),
        weights=torch.randn(n),
        hessians=torch.randn(n, extrinsic_dim, d, d),
        local_cov=local_cov_orig,
    )
    emb = create_embedding_matrix(embedding_dim=embedding_dim, extrinsic_dim=extrinsic_dim, embedding_seed=11)
    embedded = embed_dataset_with_qr_matrix(batch, emb)
    # local_cov is intrinsic — must be preserved unchanged
    assert embedded.local_cov is not None
    assert torch.allclose(embedded.local_cov, local_cov_orig)
```

**Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_datagen.py::test_sample_from_manifold_includes_local_cov tests/test_datagen.py::test_embed_dataset_preserves_local_cov -v`
Expected: FAIL

**Step 3: Implement**

In `src/numeric/datagen.py`, make these changes:

1. In `sample_from_manifold()`, after the `np_phi_hessian` line (~line 33), add:
```python
    np_local_cov = manifold_sde.manifold.sympy_to_numpy(manifold_sde.local_covariance)
```

2. After the `hessians` computation (~line 52), add:
```python
    local_covs = np.array([np_local_cov(*sample) for sample in local_samples])
```

3. Update the return tuple (~line 53-56) to include `local_covs`:
```python
    return convert_samples_to_torch(
        (ambient_samples, local_samples, extrinsic_drifts, extrinsic_covariances, orthogonal_projections, weights, hessians, local_covs),
        device=device
    )
```

4. In `embed_dataset_with_qr_matrix()` (~line 90-107), pass through `local_cov`:
```python
def embed_dataset_with_qr_matrix(dataset: DatasetBatch, embedding_matrix) -> DatasetBatch:
    x_e, mu_e, cov_e, p_e, h_e = embed_data_with_qr_matrix(
        dataset.samples,
        dataset.mu,
        dataset.cov,
        dataset.p,
        dataset.hessians,
        embedding_matrix,
    )
    return DatasetBatch(
        samples=x_e,
        local_samples=dataset.local_samples,
        mu=mu_e,
        cov=cov_e,
        p=p_e,
        weights=dataset.weights,
        hessians=h_e,
        local_cov=dataset.local_cov,  # intrinsic — not embedded
    )
```

**Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_datagen.py::test_sample_from_manifold_includes_local_cov tests/test_datagen.py::test_embed_dataset_preserves_local_cov -v`
Expected: PASS

**Step 5: Run full test suite**

Run: `python -m pytest tests/ -v`
Expected: All tests pass. Note: `test_convert_samples_to_torch_length_guard` currently checks that 6 elements raises ValueError — it will still pass because 6 != 7 and 6 != 8.

**Step 6: Commit**

```bash
git add src/numeric/datagen.py tests/test_datagen.py
git commit -m "feat: compute local_cov in sample_from_manifold and preserve through embedding"
```

---

### Task 3: Add full-space curvature geometry functions

**Files:**
- Modify: `src/numeric/geometry.py`
- Test: `tests/test_improvements.py`

**Step 1: Write the failing test**

Add to `tests/test_improvements.py` (add the new imports at the top of the file alongside existing geometry imports):

```python
from src.numeric.geometry import (
    curvature_drift_explicit,
    curvature_drift_explicit_full,
    curvature_drift_hessian_free,
    curvature_drift_hessian_free_full,
    orthogonal_projection_from_jacobian,
    ambient_quadratic_variation_drift,
    transform_covariance,
)


class TestFullSpaceCurvature:
    """Tests for full-space (no normal projection) curvature functions."""

    def setup_method(self):
        B, D, d = 4, 5, 2
        self.B, self.D, self.d = B, D, d
        self.decoder_hessian = torch.randn(B, D, d, d)
        # Make local_cov symmetric positive definite
        L = torch.randn(B, d, d)
        self.local_cov = torch.bmm(L, L.mT) + 0.1 * torch.eye(d).unsqueeze(0)
        # Projection matrix
        dphi = torch.randn(B, D, d)
        self.normal_proj = torch.eye(D).unsqueeze(0) - orthogonal_projection_from_jacobian(dphi)

    def test_curvature_drift_explicit_full_shape(self):
        result = curvature_drift_explicit_full(self.decoder_hessian, self.local_cov)
        assert result.shape == (self.B, self.D)

    def test_curvature_drift_explicit_full_no_projection(self):
        """Full version should equal explicit version WITHOUT the normal projection step."""
        full = curvature_drift_explicit_full(self.decoder_hessian, self.local_cov)
        expected = 0.5 * ambient_quadratic_variation_drift(self.local_cov, self.decoder_hessian)
        assert torch.allclose(full, expected, atol=1e-6)

    def test_curvature_drift_explicit_full_vs_projected(self):
        """Full version should differ from projected version (unless normal_proj is identity)."""
        full = curvature_drift_explicit_full(self.decoder_hessian, self.local_cov)
        projected = curvature_drift_explicit(self.decoder_hessian, self.local_cov, self.normal_proj)
        # They should generally NOT be equal
        assert not torch.allclose(full, projected, atol=1e-4)

    def test_curvature_drift_hessian_free_full_shape(self):
        ae = AutoEncoder(
            extrinsic_dim=self.D, intrinsic_dim=self.d,
            hidden_dims=[8], encoder_act=nn.Tanh(), decoder_act=nn.Tanh(),
            tie_weights=False,
        )
        z = torch.randn(self.B, self.d)
        result = curvature_drift_hessian_free_full(ae.decoder, z, self.local_cov)
        assert result.shape == (self.B, self.D)

    def test_curvature_drift_hessian_free_full_matches_explicit(self):
        """Hessian-free full should match explicit full for same decoder."""
        ae = AutoEncoder(
            extrinsic_dim=self.D, intrinsic_dim=self.d,
            hidden_dims=[8], encoder_act=nn.Tanh(), decoder_act=nn.Tanh(),
            tie_weights=False,
        )
        z = torch.randn(self.B, self.d)
        d2phi = ae.hessian_decoder(z)

        explicit = curvature_drift_explicit_full(d2phi, self.local_cov)
        hfree = curvature_drift_hessian_free_full(ae.decoder, z, self.local_cov)
        assert torch.allclose(explicit, hfree, rtol=1e-3, atol=1e-4)
```

**Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_improvements.py::TestFullSpaceCurvature -v`
Expected: FAIL — `curvature_drift_explicit_full` and `curvature_drift_hessian_free_full` don't exist.

**Step 3: Implement**

Add to `src/numeric/geometry.py` after `curvature_drift_explicit` (after line 266):

```python
def curvature_drift_explicit_full(
    decoder_hessian: Tensor,
    local_cov: Tensor,
) -> Tensor:
    """
    Compute the full-space (unprojected) Ito correction using explicit Hessian.

    Unlike curvature_drift_explicit, this does NOT project onto the normal space.
    The result is the full ambient-space Ito correction vector:
        (1/2) * Σᵢⱼ Λᵢⱼ ∂²φ/∂zᵢ∂zⱼ

    Note: this quantity is chart-dependent. Only the sum Dφ·b_z + q is
    chart-invariant. See design doc for discussion.

    Args:
        decoder_hessian: Full Hessian tensor, shape (B, D, d, d)
        local_cov: Local covariance (transformed), shape (B, d, d)

    Returns:
        ito_correction: (1/2) * tr(Λ·H), shape (B, D)
    """
    q = ambient_quadratic_variation_drift(local_cov, decoder_hessian)  # (B, D)
    return 0.5 * q


def curvature_drift_hessian_free_full(
    decoder_func,
    z: Tensor,
    local_cov: Tensor,
) -> Tensor:
    """
    Compute the full-space (unprojected) Ito correction using Hessian-free JVP.

    Like curvature_drift_hessian_free but without the normal projection step.
    Uses Proposition 8: Σᵢⱼ Λᵢⱼ H_{r,ij} = Σₖ λₖ ∇²φ_r(eₖ, eₖ)

    Note: this quantity is chart-dependent. See curvature_drift_explicit_full.

    Args:
        decoder_func: Decoder network callable
        z: Latent points, shape (B, d)
        local_cov: Local covariance matrices, shape (B, d, d)

    Returns:
        ito_correction: (1/2) * tr(Λ·H), shape (B, D)
    """
    B, d = z.shape

    # Eigendecompose local covariance
    eigenvalues, eigenvectors = torch.linalg.eigh(local_cov)  # (B, d), (B, d, d)

    # Determine D from a probe forward pass
    with torch.no_grad():
        D = decoder_func(z[:1]).shape[-1]

    result = torch.zeros(B, D, device=z.device, dtype=z.dtype)

    for k in range(d):
        e_k = eigenvectors[:, :, k]  # (B, d)
        lam_k = eigenvalues[:, k]    # (B,)

        hvvp = hessian_vector_vector_product_batch(decoder_func, z, e_k)  # (B, D)
        # NO normal projection — full ambient vector
        result = result + lam_k.unsqueeze(-1) * hvvp

    return 0.5 * result
```

**Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_improvements.py::TestFullSpaceCurvature -v`
Expected: PASS

**Step 5: Run full test suite**

Run: `python -m pytest tests/ -v`
Expected: All tests pass.

**Step 6: Commit**

```bash
git add src/numeric/geometry.py tests/test_improvements.py
git commit -m "feat: add full-space curvature drift functions (no normal projection)"
```

---

### Task 4: Add `curvature_full` to LossWeights and `autoencoder_loss`

**Files:**
- Modify: `src/numeric/losses.py:16-225`
- Test: `tests/test_improvements.py`

**Step 1: Write the failing test**

Add to `tests/test_improvements.py`:

```python
from src.numeric.losses import LossWeights, autoencoder_loss
from src.numeric.datasets import DatasetBatch


class TestCurvatureFullLoss:
    """Tests for the curvature_full loss in autoencoder_loss."""

    def setup_method(self):
        self.ae = AutoEncoder(
            extrinsic_dim=3, intrinsic_dim=2,
            hidden_dims=[8], encoder_act=nn.Tanh(), decoder_act=nn.Tanh(),
            tie_weights=False,
        )
        B, D, d = 6, 3, 2
        self.x = torch.randn(B, D)
        self.mu = torch.randn(B, D)
        self.cov = torch.eye(D).unsqueeze(0).expand(B, -1, -1) * 0.1
        self.p = torch.eye(D).unsqueeze(0).expand(B, -1, -1) * 0.5
        # Make local_cov_true SPD
        L = torch.randn(B, d, d)
        self.local_cov_true = torch.bmm(L, L.mT) + 0.1 * torch.eye(d).unsqueeze(0)
        self.hessians = torch.randn(B, D, d, d)

    def test_loss_weights_has_curvature_full(self):
        lw = LossWeights(curvature_full=0.5)
        assert lw.curvature_full == 0.5

    def test_curvature_full_loss_runs(self):
        """autoencoder_loss with curvature_full > 0 should compute without error."""
        lw = LossWeights(curvature_full=1.0)
        targets = (self.x, self.mu, self.cov, self.p)
        loss = autoencoder_loss(
            self.ae, targets, lw,
            hessians=self.hessians, local_cov_true=self.local_cov_true,
        )
        assert loss.dim() == 0  # scalar
        assert not torch.isnan(loss)

    def test_curvature_full_loss_changes_total(self):
        """Adding curvature_full should change the total loss vs baseline."""
        targets = (self.x, self.mu, self.cov, self.p)
        loss_base = autoencoder_loss(self.ae, targets, LossWeights())
        loss_kf = autoencoder_loss(
            self.ae, targets, LossWeights(curvature_full=1.0),
            hessians=self.hessians, local_cov_true=self.local_cov_true,
        )
        assert loss_kf > loss_base  # penalty should add positive term

    def test_curvature_full_backward(self):
        """curvature_full loss should be differentiable."""
        lw = LossWeights(tangent_bundle=1.0, curvature_full=1.0)
        targets = (self.x, self.mu, self.cov, self.p)
        loss = autoencoder_loss(
            self.ae, targets, lw,
            hessians=self.hessians, local_cov_true=self.local_cov_true,
        )
        loss.backward()
        # Check gradients exist on decoder parameters
        for param in self.ae.decoder.parameters():
            assert param.grad is not None

    def test_curvature_full_missing_data_raises(self):
        """curvature_full > 0 without hessians/local_cov_true should raise ValueError."""
        lw = LossWeights(curvature_full=1.0)
        targets = (self.x, self.mu, self.cov, self.p)
        with pytest.raises(ValueError, match="curvature_full"):
            autoencoder_loss(self.ae, targets, lw)
        with pytest.raises(ValueError, match="curvature_full"):
            autoencoder_loss(self.ae, targets, lw, hessians=self.hessians)
        with pytest.raises(ValueError, match="curvature_full"):
            autoencoder_loss(self.ae, targets, lw, local_cov_true=self.local_cov_true)
```

**Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_improvements.py::TestCurvatureFullLoss -v`
Expected: FAIL — `curvature_full` not a field of LossWeights, `autoencoder_loss` doesn't accept `hessians`/`local_cov_true`.

**Step 3: Implement**

In `src/numeric/losses.py`:

1. Add import for new geometry functions (replace lines 6-12):
```python
from .geometry import (
    transform_covariance,
    ambient_quadratic_variation_drift,
    curvature_drift_explicit,
    curvature_drift_explicit_full,
    curvature_drift_hessian_free,
    curvature_drift_hessian_free_full,
    regularized_metric_inverse,
)
```

2. Add field to `LossWeights` (after line 22, `curvature`):
```python
    curvature_full: float = 0.
```

3. Update `autoencoder_loss` signature (line 115-118). Use `local_cov_true` to avoid name collision with the existing `local_cov_z` computed inside:
```python
def autoencoder_loss(model: AutoEncoder,
                     targets: Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
                     loss_weights: LossWeights,
                     tangent_basis: torch.Tensor = None,
                     hessians: torch.Tensor = None,
                     local_cov_true: torch.Tensor = None):
```

4. Add early guard at the top of the function body (right after `x, mu, cov, p = targets`):
```python
    if loss_weights.curvature_full > 0.:
        if hessians is None or local_cov_true is None:
            raise ValueError(
                "curvature_full > 0 requires both `hessians` and `local_cov_true` arguments"
            )
```

5. **DO NOT** add `curvature_full` to `need_encoder_jacobian` (line 145). The curvature_full computation uses `pinv(dphi)` (decoder pseudo-inverse), not `dpi` (encoder Jacobian). Keep existing line unchanged:
```python
    need_encoder_jacobian = loss_weights.contractive > 0. or loss_weights.diffeo > 0. or loss_weights.curvature > 0.
```

6. Update `need_decoder_jacobian` (line 147-153) to include `curvature_full`:
```python
    need_decoder_jacobian = (
        loss_weights.normal_decoder_jacobian > 0.
        or loss_weights.decoder_contraction > 0.
        or loss_weights.diffeo > 0.
        or loss_weights.curvature > 0.
        or loss_weights.curvature_full > 0.
        or use_efficient_tangent
    )
```

7. Update Hessian-free threshold to include `curvature_full` (after line 164):
```python
    use_hessian_free_curvature_full = (
        loss_weights.curvature_full > 0.0
        and D > 200
        and d <= 3
    )
    need_decoder_hessian = (
        (loss_weights.curvature > 0. and not use_hessian_free_curvature)
        or (loss_weights.curvature_full > 0. and not use_hessian_free_curvature_full)
    )
```

8. Replace the curvature computation section (lines 183–203) with a restructured version that computes `penrose` and `local_cov_z` once, shared by both curvature variants:

```python
    # Shared computation for curvature losses
    # local_cov_z = pulled-back ambient covariance in z-coordinates
    if loss_weights.curvature > 0. or loss_weights.curvature_full > 0.:
        penrose = torch.linalg.pinv(dphi)
        local_cov_z = transform_covariance(cov, penrose)

    if loss_weights.curvature > 0.:
        normal_drift_true = torch.bmm(normal_proj, mu.unsqueeze(-1)).squeeze(-1)
        nhat = torch.eye(p.size(1), device=p.device, dtype=p.dtype).unsqueeze(0) - phat

        if use_hessian_free_curvature:
            normal_drift_model = curvature_drift_hessian_free(
                model.decoder, z, local_cov_z, nhat
            )
        else:
            d2phi = model.hessian_decoder(z)
            normal_drift_model = curvature_drift_explicit(d2phi, local_cov_z, nhat)

    if loss_weights.curvature_full > 0.:
        # True target: full Ito correction in ambient space from true local covariance
        ito_true = 0.5 * ambient_quadratic_variation_drift(local_cov_true, hessians)

        if use_hessian_free_curvature_full:
            ito_model = curvature_drift_hessian_free_full(model.decoder, z, local_cov_z)
        else:
            # Reuse d2phi if already computed for old curvature; otherwise compute it
            if loss_weights.curvature == 0. or use_hessian_free_curvature:
                d2phi = model.hessian_decoder(z)
            ito_model = curvature_drift_explicit_full(d2phi, local_cov_z)
```

9. Add the loss accumulation (after the existing `curvature` accumulation line):
```python
    if loss_weights.curvature_full > 0.0:
        total_loss += loss_weights.curvature_full * empirical_l2_risk(ito_model, ito_true)
```

10. `need_orthogonal_projection` (line 155) stays unchanged — `curvature_full` does NOT need `phat`:
```python
    need_orthogonal_projection = (loss_weights.tangent_bundle > 0. and not use_efficient_tangent) or loss_weights.curvature > 0.
```

**Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_improvements.py::TestCurvatureFullLoss -v`
Expected: PASS

**Step 5: Run full test suite**

Run: `python -m pytest tests/ -v`
Expected: All tests pass.

**Step 6: Commit**

```bash
git add src/numeric/losses.py tests/test_improvements.py
git commit -m "feat: add curvature_full loss (full-space Ito correction matching)"
```

---

### Task 5: Thread `local_cov` and `hessians` through training pipeline

**Files:**
- Modify: `src/numeric/training.py:57-179`
- Test: `tests/test_improvements.py`

**Step 1: Write the failing test**

Add to `tests/test_improvements.py`:

```python
class TestCurvatureFullTraining:
    """Test that curvature_full loss works end-to-end through training."""

    def test_train_epoch_with_curvature_full(self):
        """Training with curvature_full should complete without error."""
        config = TrainingConfig(
            device="cpu", hidden_dim=8, latent_dim=2, input_dim=3,
            batch_size=4, epochs=1,
        )
        trainer = MultiModelTrainer(config)
        trainer.add_model(ModelConfig(
            name="kf_test",
            loss_weights=LossWeights(tangent_bundle=1.0, curvature_full=1.0),
        ))

        B, D, d = 8, 3, 2
        L = torch.randn(B, d, d)
        local_cov = torch.bmm(L, L.mT) + 0.1 * torch.eye(d).unsqueeze(0)
        dataset = DatasetBatch(
            samples=torch.randn(B, D),
            local_samples=torch.randn(B, d),
            mu=torch.randn(B, D),
            cov=torch.eye(D).unsqueeze(0).expand(B, -1, -1) * 0.1,
            p=torch.eye(D).unsqueeze(0).expand(B, -1, -1) * 0.5,
            weights=torch.ones(B) / B,
            hessians=torch.randn(B, D, d, d),
            local_cov=local_cov,
        )

        data_loader = trainer.create_data_loader(dataset)
        losses = trainer.train_epoch(data_loader)
        assert "kf_test" in losses
        assert losses["kf_test"] > 0
```

**Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_improvements.py::TestCurvatureFullTraining -v`
Expected: FAIL — training pipeline doesn't pass `hessians`/`local_cov_true` to `autoencoder_loss`.

**Step 3: Implement**

In `src/numeric/training.py`:

The data layout in the TensorDataset is determined by `dataset.as_tuple()`:
- Indices 0-6: samples, local_samples, mu, cov, p, weights, hessians (always)
- Index 7: local_cov (if present in DatasetBatch)
- Last index: tangent_basis (appended by create_data_loader if computed)

We need to track whether local_cov is in the tuple. Use a `_has_local_cov` flag.

1. In `__init__` (~line 66), add:
```python
        self._has_local_cov = False
```

2. In `create_data_loader`, set the flag for both DatasetBatch and tuple inputs (~after line 103):
```python
    def create_data_loader(self, dataset: Union[DatasetBatch, Tuple[torch.Tensor, ...]]):
        """..."""
        needs_tangent_basis = any(
            cfg.loss_weights.tangent_bundle > 0
            for cfg in self.model_configs.values()
        )

        if isinstance(dataset, DatasetBatch):
            self._has_local_cov = dataset.local_cov is not None

            if needs_tangent_basis and dataset.tangent_basis is None:
                dataset.compute_tangent_basis(self.config.latent_dim)

            if dataset.tangent_basis is not None:
                tensors = dataset.as_tuple() + (dataset.tangent_basis,)
            else:
                tensors = dataset.as_tuple()
        else:
            # Tuple input: disambiguate local_cov (d,d) from tangent_basis (D,d).
            # If 8th element is square (last two dims equal), it's local_cov.
            # tangent_basis is (B, D, d) where D > d, so not square.
            if len(dataset) >= 8 and dataset[7].shape[-1] == dataset[7].shape[-2]:
                self._has_local_cov = True
            else:
                self._has_local_cov = False
            tensors = dataset

        tensor_dataset = torch.utils.data.TensorDataset(*tensors)
        return torch.utils.data.DataLoader(
            tensor_dataset,
            batch_size=self.config.batch_size,
            shuffle=True
        )
```

3. Update `train_epoch` batch unpacking (~line 138-162):
```python
        for batch_idx, batch in enumerate(data_loader):
            x, _, mu, cov, p, _, hessians_batch = batch[:7]
            idx = 7
            if self._has_local_cov:
                local_cov_true_batch = batch[idx]
                idx += 1
            else:
                local_cov_true_batch = None
            tangent_basis = batch[idx] if len(batch) > idx else None

            # Move tensors to device
            x = x.to(self.device)
            mu = mu.to(self.device)
            cov = cov.to(self.device)
            p = p.to(self.device)
            hessians_batch = hessians_batch.to(self.device)
            if local_cov_true_batch is not None:
                local_cov_true_batch = local_cov_true_batch.to(self.device)
            if tangent_basis is not None:
                tangent_basis = tangent_basis.to(self.device)
            targets = (x, mu, cov, p)

            for model_name, model in self.models.items():
                optimizer = self.optimizers[model_name]
                if loss_weights_override and model_name in loss_weights_override:
                    loss_weights = loss_weights_override[model_name]
                else:
                    loss_weights = self.model_configs[model_name].loss_weights

                optimizer.zero_grad()
                loss = autoencoder_loss(
                    model, targets, loss_weights,
                    tangent_basis=tangent_basis,
                    hessians=hessians_batch,
                    local_cov_true=local_cov_true_batch,
                )
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), self.config.grad_clip_max_norm
                )
                optimizer.step()

                epoch_losses[model_name] += loss.item()
```

**Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_improvements.py::TestCurvatureFullTraining -v`
Expected: PASS

**Step 5: Run full test suite**

Run: `python -m pytest tests/ -v`
Expected: All tests pass.

**Step 6: Commit**

```bash
git add src/numeric/training.py tests/test_improvements.py
git commit -m "feat: thread local_cov and hessians through training pipeline for curvature_full"
```

---

### Task 6: Add new penalty configs and update `sample_ring_region`

**Files:**
- Modify: `experiments/common.py:32-39` (PENALTY_CONFIGS)
- Modify: `experiments/common.py:42-84` (sample_ring_region)

**Step 1: Implement directly (no separate test — configs are data, not logic)**

In `experiments/common.py`:

1. Add to `PENALTY_CONFIGS` (after line 38). Note: "F" = diffeo penalty in this repo (not decoder_contraction). Kf should always be paired with F for fair comparison (gauge-fixing):
```python
    "T+Kf":     LossWeights(tangent_bundle=1.0, curvature_full=1.0),
    "T+F+Kf":   LossWeights(tangent_bundle=1.0, diffeo=1.0, curvature_full=1.0),
```

2. In `sample_ring_region` (line 76-84), add `local_cov` to the returned DatasetBatch:
```python
    return DatasetBatch(
        samples=full_dataset.samples[indices],
        local_samples=full_dataset.local_samples[indices],
        weights=full_dataset.weights[indices],
        mu=full_dataset.mu[indices],
        cov=full_dataset.cov[indices],
        p=full_dataset.p[indices],
        hessians=full_dataset.hessians[indices],
        local_cov=full_dataset.local_cov[indices] if full_dataset.local_cov is not None else None,
    )
```

**Step 2: Run full test suite**

Run: `python -m pytest tests/ -v`
Expected: All tests pass.

**Step 3: Commit**

```bash
git add experiments/common.py
git commit -m "feat: add T+Kf and T+F+Kf penalty configs, thread local_cov in sample_ring_region"
```

---

### Task 7: Smoke test — end-to-end trajectory fidelity with T+F+Kf

**Step 1: Run a quick smoke test**

Run: `python -m experiments.trajectory_fidelity_study --surface paraboloid --penalty "T+F+Kf" --epochs 50`

Expected: Completes without error, prints loss values, produces CSV output.

If it fails, debug and fix before proceeding.

**Step 2: Commit any fixes**

Only if Task 7 smoke test required fixes.

---

### Task 8: Run the full experiment

**Step 1: Run trajectory fidelity study with new configs**

Run:
```bash
python -m experiments.trajectory_fidelity_study \
    --surface paraboloid \
    --penalty "T+F" "T+Kf" "T+F+Kf" \
    --epochs 500
```

**Step 2: Compare results**

Check whether T+F+Kf improves on T+F for MTE and W2 metrics per the success criteria in the design doc.
