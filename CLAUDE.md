# AEML Project

## Git Push

Conda's OpenSSL 3.6 overrides the system's 3.0. The system `ssh` was built against 3.0 and picks up conda's 3.6 at runtime, causing `OpenSSL version mismatch` errors.

Fix: temporarily point to the system libs when pushing:

```bash
LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu git push origin main
```

## Repository Layout

- `Autoencoder-Paper/` is a **separate git repo** (not a submodule). Commit and push there independently.

## Data-Driven SDE Pipeline

3-stage decoupled pipeline: Stage 1 (AE), Stage 2 (drift_net, frozen AE), Stage 3 (diffusion_net, frozen AE).

### Training Data

Training data should be **very small (20–40 points)** — sparse observations from the manifold. This is the realistic data-driven regime.

### Penalty Configuration

The paper uses **T+F penalties only** (tangent-bundle + inverse-consistency). No curvature (K) or drift smoothness (L_S) penalties in the final paper.

- **T (tangent-bundle)**: aligns decoder Jacobian with observed covariance eigenvectors
- **F (inverse-consistency)**: ||Dπ·Dφ - I_d||²_F
- **Encoder-pullback drift**: b_z = Dπ·b + ½⟨Λ, ∇²π⟩ — used for Stage 2 (not decoder-side formula)

### Two-Phase AE Training

Train Phase 1 with T+F (warmup), then Phase 2 continues T+F (fine-tune). Two-phase is used for all conditions including baseline.

### Curvature (K) Regularization (experimental, not in paper)

K was investigated but is **not included in the final SIAM MDS paper**. Multi-seed findings showed modest/mixed effects at trajectory level despite improving coefficient errors.
