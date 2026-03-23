# Plan: Restructure Paper — Remove GC2/K, Add Encoder-Pullback Drift

## Summary

Replace the GC2/K narrative with the encoder-pullback drift approach. The paper's new story: chart-invariant T+F penalties for the autoencoder (Stage 1), then encoder-pullback drift fitting (Stage 2) derived directly from Itô's formula. Include a bias analysis showing why the old decoder-side drift target is systematically biased.

**Name**: "encoder-pullback drift" (not "enc_pull")

**Presentation**: encoder-pullback is THE method. Mention decoder-side briefly as an alternative that introduces systematic bias.

## Files

- **`Autoencoder-Paper/paper_trimmed.tex`** — all edits go here (working copy)
- **`Autoencoder-Paper/paper.tex`** — unchanged (original reference)
- New experiment scripts/CSVs as needed

---

## Step 1: §2 — Remove GC2, Add Encoder-Pullback Subsection

### Remove

| Item | Lines | Description |
|------|-------|-------------|
| GC2 bullet | 259–261 | `\item[(GC2)] the normal component...` in the enumeration |
| GC2 forward-ref | 263–265 | "derives (GC1) and (GC2) from..." → just "(GC1)" |
| Lemma 2.6 + proof | 396–422 | `curvature_from_drift`: normal projection of q(Σ) |
| Example 2.1 | 424–434 | Brownian motion — uses GC2 curvature decomposition |
| Conceptual dichotomy display | 436–446 | `covariance ↔ tangent` / `normal drift ↔ curvature` — remove the second line |
| GC2 references in coord invariance | 448–476 | Simplify: P is invariant. Remove "and the normal drift Nb" from Lemma 2.5; remove GC2 mentions |

### Modify

- **Lemma 2.5 (coord invariance)**: Keep the statement about P being invariant. Remove "and the normal drift $Nb$". Simplify proof accordingly.
- **§2.3 opening** (lines 353–380): Keep Lemma 2.4 (tangent from covariance). After it, the paragraph "Having extracted first-order geometry..." (line 379) leads into GC2 — rewrite to lead into the new §2.4 instead.

### Add: New §2.4 "Encoder-Pullback Drift Target"

**Content** (~30 lines):

The encoder-pullback drift target comes directly from Itô's formula. The proof of Lemma 2.3 already shows:

$$b_z = D\pi \cdot b + \tfrac{1}{2}\langle\Lambda, \nabla^2\pi\rangle_F$$

This is the first line of the Itô expansion of $z = \pi(X)$, before applying Lemma 2.2 to rewrite in decoder coordinates. The key observation:

**Remark (Encoder-pullback vs decoder-side drift).** The latent drift $\mu$ admits two equivalent representations:
1. **Encoder form**: $\mu = D\pi \cdot b + \frac{1}{2}\langle\Lambda, \nabla^2\pi\rangle_F$ — requires encoder Jacobian and Hessian
2. **Decoder form** (Lemma 2.3): $\mu = D\pi[b - \frac{1}{2}q(\Sigma)]$ — requires decoder Hessian and metric pseudo-inverse

For the true chart ($\pi = \phi^{-1}$), these are identical (by Lemma 2.2). For a learned autoencoder where $\pi_\theta \circ \phi_\theta \neq \mathrm{id}$, they differ: the encoder form uses the actual encoder $\pi_\theta$ via autodiff, while the decoder form substitutes the metric pseudo-inverse $g^{-1}D\phi^T$ for $D\pi$ — introducing systematic errors analyzed in §4.X.

**Algorithmic consequence**: Given observations $(x_i, b(x_i), \Lambda(x_i))$ and a trained encoder $\pi_\theta$, the encoder-pullback target $b_z(z_i) = D\pi_\theta(x_i) \cdot b(x_i) + \frac{1}{2}\langle\Lambda(x_i), \nabla^2\pi_\theta(x_i)\rangle_F$ is computable via encoder autodiff without touching the decoder. This target is used to train the latent drift network in Stage 2 (Algorithm 1).

---

## Step 2: §3 — Remove K Penalty

### Remove

| Item | Lines | Description |
|------|-------|-------------|
| K penalty definition | 676–701 | GC2 residual $R_K(\phi)$, curvature penalty $K(\phi)$ |
| K in training objective | 720–724 | `$+\lambda_K K(\phi_\theta)$` |
| K in ERM | 735–741 | `$+\lambda K(\phi_\theta)$` in ERM |
| Three roles summary | 744–755 | "GC1 and GC2 play three roles" → simplify to two roles for GC1 only |
| Prop 3.5 part (K) | 821–886 | Remove curvature penalty computation (keep tangent part T) |
| K in data requirements | 783–788 | Item 3 "Drift data needed for K" — modify: drift data still needed but for encoder-pullback target, not K |

### Modify

- **§3.1 opening** (lines 481–491): Remove "curvature regularizer" mention, just "function-space metric and tangent-bundle penalty"
- **Prop 3.5**: Keep part (T). Remove part (K). The efficient computation section becomes shorter.
- **Data requirements**: Item 3 (drift data) changes motivation: "needed for the encoder-pullback drift target (§2.4)" instead of "needed for K"

---

## Step 3: §4 — Remove K-Preserves-Rate, Add Bias Analysis

### Remove

| Item | Lines | Description |
|------|-------|-------------|
| §4.2 entirely | 1333–1441 | "K preserves ρ-rate" — Prop 4.2 and all discussion |
| Forward-ref to §4.2 | 931–932 | "Section 4.2 further shows..." |

### Add: New §4.2 "Bias of Decoder-Side Drift Fitting" (~60 lines)

**Content**: Rigorous bias decomposition. When the autoencoder is imperfect, the decoder-side target $\mu_{dec} = g^{-1}D\phi^T[b - \frac{1}{2}q(\hat\Sigma)]$ differs from the encoder-pullback target $\mu_{enc} = D\pi \cdot b + \frac{1}{2}\langle\Lambda, \nabla^2\pi\rangle_F$ by:

$$\mu_{dec} - \mu_{enc} = \underbrace{(g^{-1}D\phi^T - D\pi)(b - \tfrac{1}{2}q(\hat\Sigma))}_{\text{(I) pseudo-inverse vs encoder}} - \tfrac{1}{2}\underbrace{\hat\Sigma : D^2(\pi\circ\phi)}_{\text{(II) cycle Hessian bias}} - \tfrac{1}{2}\underbrace{(\Lambda - D\phi\,\hat\Sigma\,D\phi^T):\nabla^2\pi}_{\text{(III) covariance mismatch}}$$

where $\hat\Sigma = g^{-1}D\phi^T \Lambda D\phi\,g^{-1}$.

**Key point**: Term (II) is a deterministic bias — $D^2(\pi\circ\phi) \neq 0$ for any imperfect AE, and this doesn't vanish with more training data or larger networks. It's the fundamental reason the decoder-side approach is worse.

State as a **Proposition**: "For a learned autoencoder with $\pi\circ\phi \neq \mathrm{id}$, the decoder-side drift target has systematic bias of order $\|D^2(\pi\circ\phi)\|$."

### Keep

- §4.1 Main generalization theorem — unchanged (about ρ-ERM, doesn't involve K)
- §4.3 Error propagation — unchanged (chart quality → coeff → process convergence)

---

## Step 4: Algorithm 1

### Current (lines 1611–1641)

Step 1: T+F+K chart learning (two-phase)
Step 2: Decoder-side drift via Lemma 2.3
Step 3: Diffusion fitting

### New

```
Step 1: Chart learning (T+F)
  Train (π_θ, φ_θ) with L = L_recon + λ_T L_T + λ_F L_F

Step 2: Encoder-pullback drift fitting (frozen chart)
  Compute target: b_z(z_i) = Dπ_θ(x_i)·b(x_i) + ½⟨Λ(x_i), ∇²π_θ(x_i)⟩_F
  Train μ̂_ω to minimize Σ_i ‖μ̂_ω(z_i) - b_z(z_i)‖²_g

Step 3: Diffusion fitting (frozen chart) — unchanged
```

Remove two-phase schedule (no K warmup needed). Remove λ_K parameter.

---

## Step 5: §5 Experiments — Restructure Tables

### §5.1 Setup
- Remove K from penalty configurations
- Remove two-phase schedule description
- Update algorithm reference
- Mention drift smoothness regularization (already there)

### §5.2 Ablation (Table 1)
- **Keep as-is** — this table is about chart quality (reconstruction, tangent, curvature errors). Still valid for T+F analysis. The K column provides useful comparison showing K helps chart quality but (as we now show) hurts drift fitting.
- Add a sentence: "While K improves chart-level metrics, §5.4 shows it does not improve — and can hinder — the downstream drift fitting."

### §5.3 Extrapolation (Table 2)
- **Keep as-is** — chart extrapolation, still valid for T+F

### §5.4 Dynamics Extrapolation (Table 3)
- **Need new data**: Run enc_pull on D=3 surfaces with the same conditions
- Compare: T+F with encoder-pullback vs T+F with decoder-side vs baseline
- Or: just present T+F + enc_pull results, mention decoder-side briefly

### §5.5 Trajectory Fidelity (Table 4)
- **Need new data**: Run enc_pull on 4 surfaces × 10 seeds
- Replace Table 4 with enc_pull results
- Mention briefly: "The decoder-side approach (Lemma 2.3) yields higher W2 due to systematic bias (§4.2); see supplementary for comparison."

### §5.6 N×D Sweep (Table 5)
- **Already have data**: `highd_enc_pull.log` (480 rows, 10 seeds)
- Replace current 3-condition table (baseline/K/K+S) with 2-condition (baseline/enc_pull) or show enc_pull improvement over both
- The enc_pull data shows 22-66% W2 improvement across all 32 cells

### New subsection or merged: Decoder-side comparison
- Brief (1 paragraph + 1 small table or figure) showing enc_pull vs decoder-side W2
- Reference bias analysis in §4.2

---

## Step 6: §6 Discussion

- Remove all K-specific claims
- New framing: T+F produces geometrically consistent charts; encoder-pullback drift exploits the encoder directly via Itô's formula, avoiding the systematic bias of decoder-side fitting
- Keep: ρ-metric theory, error propagation chain
- Keep: future work (multi-chart, non-compact, statistical estimation)
- Add: the bias decomposition as a design principle — when the AE is imperfect, prefer encoder-side computation over decoder pseudo-inverse

---

## Step 7: §1 Abstract and Introduction

Rewrite last. After all structural changes are made:
- Abstract: remove K claims, add encoder-pullback drift narrative
- Contributions: (1) chart-invariant T+F, (2) ρ-metric generalization, (3) encoder-pullback drift + bias analysis, (4) experiments
- Outline: update section references

---

## Step 8: Supplementary Material

- Move current K-related proofs to supplementary (or remove if redundant)
- Move decoder-side comparison details to supplementary
- Keep ρ̃-metric appendix but note it's no longer used in main text

---

## Experiments Needed

| Experiment | Status | Description |
|-----------|--------|-------------|
| Enc_pull N×D sweep | ✅ Done | `highd_enc_pull.log`, 480 rows |
| Enc_pull capacity test | ✅ Done | `drift_capacity_test.csv`, 40 rows |
| Enc_pull D=3 dynamics | ❌ Needed | 4 surfaces × 10 seeds, for Tables 3-4 |
| Enc_pull vs decoder W2 | ✅ Have data | Can extract from existing sweep |

**One experiment to run**: D=3 encoder-pullback dynamics on paraboloid, hyp_paraboloid, quartic_dome, sinusoidal with 10 seeds each — replaces Tables 3 and 4.

---

## Execution Order

1. Run missing D=3 experiment (background, ~30 min)
2. §2 edits: remove GC2, add encoder-pullback subsection
3. §3 edits: remove K penalty
4. §4 edits: remove K-preserves-rate, add bias analysis
5. Algorithm 1: rewrite
6. §5 edits: update experiment sections with new data
7. §6 edits: rewrite discussion
8. §1 edits: rewrite abstract and intro (last, after structure is stable)
9. Compile and verify: `pdflatex`, check references, count pages

## Line Budget Estimate

| Section | Lines removed | Lines added | Net |
|---------|-------------|------------|-----|
| §2 | ~80 (GC2, Lem 2.6, Ex 2.1) | ~35 (enc_pull subsection) | −45 |
| §3 | ~100 (K penalty, K computation) | ~5 | −95 |
| §4 | ~110 (K-preserves-rate) | ~65 (bias analysis) | −45 |
| §5 | ~40 (K discussions) | ~30 (enc_pull comparison) | −10 |
| §6 | ~30 (K claims) | ~20 (enc_pull discussion) | −10 |
| §1 | ~15 (K mentions) | ~15 (enc_pull mentions) | 0 |
| **Total** | **~375** | **~170** | **−205** |

Paper shrinks by ~200 lines (~2.5 pages), becoming tighter and more focused.
