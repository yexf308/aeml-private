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

### Curvature (K) Regularization

**Two-phase AE training is required when using K.** Train Phase 1 with T+F only (warmup), then Phase 2 with T+F+K (fine-tune). Single-phase K training is unstable with sparse data.

Multi-seed study findings (N=20, 10 seeds, two-phase AE, paired t-test):

- **paraboloid**: K helps MTE by 6.6% (p=0.009**), W2 by 13.8% (p=0.005**). 9/10 seeds show improvement.
- **hyperbolic_paraboloid**: K helps W2 by 9.5% (p=0.001**), MMD by 12.2% (p=0.011*), MTE by 2.5% (p=0.057+). 9/10 seeds show W2/MMD improvement.
- **sinusoidal**: K is neutral (no significant effect on any metric). This is because `z = sin(u+v)` is a **developable surface** with zero Gaussian curvature everywhere (`det(H) = 0` since the Hessian is rank-1). There is no intrinsic curvature signal for K to capture.
- With abundant data (N=2000), K has no significant effect — the geometric inductive bias only matters in the sparse regime.
