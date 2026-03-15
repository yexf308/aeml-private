# Two-Dynamics Experiment Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Run two physically clean dynamics on all 4 surfaces, each with appropriate metrics:
1. **MB Langevin** (−∇V_MB + √(2kT)·I): inter-well MFPT, transition rate
2. **Rotation + state-dependent σ** (existing): radial MFPT, W2, MTE

**Architecture:** MB experiments use a shared `mb_dynamics.py` module with isotropic diffusion (FDT holds). Rotation experiments use existing `mfpt_full_ablation.py` and `highd_N_D_sweep.py` infrastructure unchanged.

**Tech Stack:** PyTorch, NumPy, Pandas, SLURM (dgx partition, 1 GPU)

---

## Two Dynamics

### Dynamics 1: MB Langevin (overdamped)
- **Drift:** −∇V_MB(u,v) — rescaled Müller-Brown potential gradient
- **Diffusion:** √(2kT)·I₂ — isotropic, FDT-consistent
- **Physics:** Proper overdamped Langevin. Boltzmann stationary distribution. Kramers escape rates physically interpretable.
- **Metrics:** Inter-well MFPT (well 1 → any other), transition rate (#well-jumps / time)
- **Why:** Tests whether learned SDE captures metastable dynamics with correct well-jumping statistics

### Dynamics 2: Rotation + state-dependent σ (existing)
- **Drift:** (−v, u) — non-gradient rotation, no equilibrium
- **Diffusion:** σ(u,v) — anisotropic, state-dependent
- **Physics:** General SDE (no FDT). Exercises full 3-stage pipeline with non-trivial Stage 3.
- **Metrics:** Radial MFPT (exit ambient ball at r=0.5,1,2,3), W2, MTE
- **Why:** Tests whether learned SDE captures trajectory statistics with non-trivial diffusion

### Surfaces (both dynamics)
paraboloid, hyperbolic_paraboloid, quartic_dome, sinusoidal

### MB calibration results (isotropic diffusion, kT=0.15, T=20, paraboloid D=11)
```
MFPT: 6.09 ± 4.68
Transition frac: 89%
Rate: 0.4400/t (1760 transitions in 200 trajectories)
Transitions per traj: 8.8
```
No parameter adjustment needed.

---

## File Structure

### MB Langevin (new files)
| File | Status | Responsibility |
|------|--------|----------------|
| `experiments/mb_dynamics.py` | ✅ DONE | MB potential, drift, isotropic diffusion, wells, inter-well metrics, D-general two-phase AE trainer |
| `experiments/mb_contractive_sweep.py` | ✅ DONE | λ_C sweep: {0.01, 0.05, 0.1, 0.5}, paraboloid D=11, 3 seeds |
| `experiments/mb_ablation.py` | ✅ DONE | Ablation: 9 configs × paraboloid D=11 × 10 seeds |
| `experiments/mb_mfpt.py` | ✅ DONE | Inter-well MFPT: 4-5 conditions × 4 surfaces × D=11 or D=201 × 10 seeds |
| `experiments/mb_nd_sweep.py` | ✅ DONE | N×D sweep with MB dynamics |
| `run_mb_*.sh` (×5) | ✅ DONE | SLURM scripts |

### Rotation + σ(u,v) (existing files, no changes needed)
| File | Status | Responsibility |
|------|--------|----------------|
| `experiments/mfpt_full_ablation.py` | ✅ EXISTS | Radial MFPT: 5 conditions × 4 surfaces × D=11/201 × 10 seeds |
| `experiments/highd_N_D_sweep.py` | ✅ EXISTS | N×D sweep: 3 conditions × 2 surfaces × D=11+201 |
| `experiments/paper_experiments.py` | ✅ EXISTS | Ablation (D=3), extrapolation, trajectory fidelity |

---

## Task 1: MB Dynamics Module ✅ COMPLETE

- [x] Write `mb_dynamics.py` with isotropic diffusion √(2kT)·I
- [x] Verify: σ diagonal = 0.5477 = √(2×0.15), off-diagonal = 0

## Task 2: GT Calibration ✅ COMPLETE

- [x] Isotropic diffusion: 89% transition frac, 8.8 transitions/traj, MFPT≈6.09
- [x] No parameter adjustment needed

## Task 3: Contractive Sweep ✅ COMPLETE (Job 60145 with state-dep σ)

**Result: λ_C=0.01 best.** Set in mb_ablation.py and mb_mfpt.py.

**Note:** Sweep ran with state-dependent σ (stale). Rerunning with isotropic (Job 60164) for correctness, but λ_C ranking is unlikely to change since contractive penalty acts on the encoder Jacobian, not diffusion.

## Task 4: MB Ablation 🔄 RUNNING (Job 60165)

9 configs × paraboloid D=11 × 10 seeds, LAMBDA_C=0.01, isotropic diffusion.

## Task 5: MB Inter-well MFPT D=11 🔄 RUNNING (Job 60166)

5 conditions × 4 surfaces × 10 seeds. Metrics: inter-well MFPT, transition rate, E_μ, E_Σ.

## Task 6: MB Inter-well MFPT D=201 🔄 RUNNING (Job 60167)

4 conditions × 4 surfaces × 10 seeds (drops C). Same metrics.

## Task 6b: MB N×D Sweep 🔄 RUNNING (Job 60168)

3 conditions × 2 surfaces × D=11+201 × 4 N values × 10 seeds.

## Task 7: Rotation Experiments (existing)

The rotation-drift experiments already exist and have been run:
- `mfpt_full_ablation.py` → radial MFPT on all 4 surfaces
- `highd_N_D_sweep.py` → N×D sweep

These may need to be rerun if the current CSVs are from stale configs. Check before paper update.

- [ ] **Step 1:** Verify existing rotation CSVs are current (correct conditions, 10 seeds)
- [ ] **Step 2:** Rerun if needed

## Task 8: Collect Results and Update Paper ⏳ WAITING

- [ ] **Step 1: Verify CSV completeness**
  - `mb_ablation.csv` — 90 rows
  - `mb_mfpt_d11.csv` — 200 rows
  - `mb_mfpt_d201.csv` — 160 rows
  - `mb_nd_sweep.csv` — 480 rows
  - Rotation CSVs — verify row counts

- [ ] **Step 2: Paper structure for two dynamics**

  §5.1 Setup: describe both dynamics
  §5.2 Ablation: MB Langevin, paraboloid D=11 (chart quality)
  §5.3 MB inter-well MFPT + transition rate (all 4 surfaces, D=11 and D=201)
  §5.4 Rotation radial MFPT (all 4 surfaces, D=11 and D=201)
  §5.5 N×D sweep (one or both dynamics)

  **Supplement:** extrapolation, coefficient errors

- [ ] **Step 3: Update tables and text with numbers from CSVs**
- [ ] **Step 4: Add MB reference to bibliography**
- [ ] **Step 5: Compile, verify, cross-check**

---

## Execution Order

```
Task 1 (mb_dynamics.py)                         ✅ DONE
Task 2 (GT calibration)                         ✅ DONE (89% trans frac, isotropic)
Task 3 (sweep) → Job 60164                      🔄 RUNNING (~1hr)
Task 4 (ablation) → Job 60165                   🔄 RUNNING (~30min)
Task 5 (MFPT D=11) → Job 60166                  🔄 RUNNING (~6hr)
Task 6 (MFPT D=201) → Job 60167                 🔄 RUNNING (~8hr)
Task 6b (N×D sweep) → Job 60168                 🔄 RUNNING (~8hr)
Task 7 (rotation experiments)                    ⏳ check/rerun existing
Task 8 (paper update)                            ⏳ WAITING for all jobs
```

---

## Job Tracking

| Job ID | Script | Status | Notes |
|--------|--------|--------|-------|
| 60145 | sweep (old) | ✅ COMPLETED | State-dep σ, λ_C=0.01 best |
| 60146 | ablation (old) | ✅ COMPLETED | State-dep σ + λ_C=0.1, superseded |
| 60147-60149 | mfpt/nd (old) | ❌ CANCELLED | State-dep σ + stale λ_C |
| 60152-60154 | ablation/mfpt (old) | ❌ CANCELLED | State-dep σ |
| **60164** | `run_mb_sweep.sh` | 🔄 RUNNING | Isotropic σ |
| **60165** | `run_mb_ablation.sh` | 🔄 RUNNING | Isotropic σ, λ_C=0.01 |
| **60166** | `run_mb_mfpt_d11.sh` | 🔄 RUNNING | Isotropic σ |
| **60167** | `run_mb_mfpt_d201.sh` | 🔄 RUNNING | Isotropic σ |
| **60168** | `run_mb_nd_sweep.sh` | 🔄 RUNNING | Isotropic σ |

---

## Codex Review Fixes (applied)

1. **LAMBDA_C stale** → Updated to 0.01 in mb_ablation.py and mb_mfpt.py
2. **hash(cond_label) non-reproducible** → Deterministic `_COND_SEED_OFFSET` dict
3. **rate_err NaN when learned=0** → Reports 100% error when GT has transitions but learned doesn't
4. **args.N not threaded** → `run_one()` accepts `n_train` parameter
5. **Frankenstein SDE** → Fixed: MB uses isotropic √(2kT)·I (proper Langevin), rotation uses state-dependent σ (separate experiments)

---

## Key Design Decisions

1. **Two dynamics, not one:** Each dynamics tests a different aspect — MB tests metastable well-jumping, rotation tests non-trivial diffusion learning. Neither is a Frankenstein.

2. **MB = overdamped Langevin:** FDT holds. Boltzmann stationary distribution. Kramers theory applies. Physically meaningful MFPT and transition rates.

3. **Rotation = general SDE:** Not Langevin. Exercises Stage 3 with state-dependent σ. Radial MFPT measures trajectory spreading fidelity.

4. **Isotropic diffusion makes Stage 3 trivial for MB:** This is fine — the MB experiments focus on chart quality (Stage 1) and drift learning (Stage 2). The rotation experiments cover Stage 3.
